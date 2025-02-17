import os
import pathlib
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
from omegaconf import OmegaConf
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnv, GraphBuildingEnvContext
from gflownet.utils.misc import create_logger
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
from gflownet.utils.misc import prepend_keys, average_values_across_dicts
from gflownet.utils.metrics_final_eval import candidates_eval
import wandb
import omegaconf
from rdkit import Chem

from .config import Config

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


class GFNAlgorithm:
    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations

        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).

        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        raise NotImplementedError()


class GFNTask:
    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional informations (e.g. temperature)
        flat_reward: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.

        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar log-reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: FlatRewards
            A 2d tensor, a vector of scalar reward for valid each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any]):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        hps: Dict[str, Any]
            A dictionary of hyperparameters. These override default values obtained by the `set_default_hps` method.
        device: torch.device
            The torch device of the main worker.
        """
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        self.replay_buffer: Optional[ReplayBuffer]
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm

        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method, typically what is defined by a task
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources
        # NEW: option to get configs from wandb sweeps
        self.cfg: Config = OmegaConf.structured(Config())
        self.set_default_hps(self.cfg)
        # OmegaConf returns a fancy object but we can still pretend it's a Config instance
        self.cfg = OmegaConf.merge(self.cfg, hps)  # type: ignore
        # self.cfg = self.setup_sweep_config(hps) #For doing wandb sweeps

        self.device = torch.device(self.cfg.device)
        # Print the loss every `self.print_every` iterations
        self.print_every = self.cfg.print_every
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup()

    def setup_sweep_config(self, hps):
        # Params we hyperoptimize over
        wandb_config = {
            "learning_rate": self.cfg.opt.learning_rate,
            "lr_decay": self.cfg.opt.lr_decay,
            "Z_learning_rate": self.cfg.algo.tb.Z_learning_rate,
            "Z_lr_decay": self.cfg.algo.tb.Z_lr_decay,
        }
        # Get wandb to generate params we sweep over
        wandb.init(project=self.cfg.experiment_name, sync_tensorboard=True, config=wandb_config)
        print(wandb.config)

        # Convert wandb.config to one that can be merged with omegaconf
        wandb_hps = {
            "opt": {
                "learning_rate": wandb.config["learning_rate"],
                "lr_decay": wandb.config["lr_decay"],
            },
            "algo": {
                "tb": {
                    "Z_learning_rate": wandb.config["Z_learning_rate"],
                    "Z_lr_decay": wandb.config["Z_lr_decay"],
                }
            },
        }
        cfg = OmegaConf.merge(self.cfg, wandb_hps)
        return cfg

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup_env_context(self):
        raise NotImplementedError()

    def setup_task(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_algo(self):
        raise NotImplementedError()

    def setup_data(self):
        pass

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def setup(self):
        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()

    def _wrap_for_mp(self, obj, send_to_device=False):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if send_to_device:
            obj.to(self.device)
        if self.cfg.num_workers > 0 and obj is not None:
            placeholder = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            return placeholder, torch.device("cpu")
        else:
            return obj, self.device

    def build_callbacks(self):
        return {}

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)
        iterator = SamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            online_batch_size=self.cfg.algo.online_batch_size,
            replay_batch_size=self.cfg.algo.replay_batch_size,
            offline_batch_size=self.cfg.algo.offline_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=replay_buffer,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            random_traj_prob=self.cfg.algo.train_random_traj_prob,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            # The 2 here is an odd quirk of torch 1.10, it is fixed and
            # replaced by None in torch 2.
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.model, send_to_device=True)
        iterator = SamplingIterator(
            self.test_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            online_batch_size=self.cfg.algo.online_batch_size,
            replay_batch_size=0,
            offline_batch_size=self.cfg.algo.offline_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "valid"),
            sample_cond_info=self.cfg.algo.valid_sample_cond_info,
            stream=False,
            random_action_prob=self.cfg.algo.valid_random_action_prob,
        )
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def build_final_data_loader(self) -> DataLoader:
        # Final data loader is now used to generate final trajectories for evaluation
        # it is different from validation data loader in that it does not take any test_data
        model, dev = self._wrap_for_mp(self.model, send_to_device=True)  # changed to model
        iterator = SamplingIterator(
            [],  # changed
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            online_batch_size=self.cfg.algo.online_batch_size,
            replay_batch_size=0,
            offline_batch_size=0,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            log_dir=os.path.join(self.cfg.log_dir, "final"),
            sample_cond_info=self.cfg.algo.valid_sample_cond_info,  # changed
            stream=False,
            random_action_prob=self.cfg.algo.valid_random_action_prob,
            hindsight_ratio=0.0,
            # init_train_iter=self.cfg.num_training_steps,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        try:
            loss, info = self.algo.compute_batch_losses(self.model, batch)
            # print("batch len", len(batch.is_valid))
            if not torch.isfinite(loss):
                raise ValueError("loss is not finite")
            step_info = self.step(loss)
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError("parameters are not finite")
        except ValueError as e:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            torch.save([self.model.state_dict(), batch, loss, info], open(self.cfg.log_dir + "/dump.pkl", "wb"))
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(self.model, batch)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        if self.cfg.num_final_gen_steps:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        for it, (batch, _) in zip(range(start, 1 + num_training_steps), cycle(train_dl)):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            info_train = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info_train.items()))
            info_train = prepend_keys(info_train, "train")
            self.log(info_train, it)

            if (valid_freq > 0 and it % valid_freq == 0) or (it == num_training_steps):
                info_val = []
                # for batch in valid_dl:
                # validate on at least 10 batches
                for valid_it, (batch, _) in zip(range(10), cycle(valid_dl)):
                    # print("valid_it", valid_it)
                    info_val.append(self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx))
                info_val = average_values_across_dicts(info_val)
                logger.info(f"VALIDATION - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info_val.items()))
                info_val = prepend_keys(info_val, "val")
                self.log(info_val, it)
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                end_metrics = prepend_keys(end_metrics, "val_end")
                self.log(end_metrics, it)
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self._save_state(it)
        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        if num_final_gen_steps:
            gen_candidates_list = []
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, (_, gen_candidates_eval_info) in zip(
                range(num_training_steps, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                gen_candidates_list.append(gen_candidates_eval_info)

            info_final_gen = candidates_eval(gen_candidates_list)
            logger.info("Final generation steps completed.")
            self.log(info_final_gen, it)
            logger.info(f"FINAL CANDIDATE GENERATION : " + " ".join(f"{k}:{v:.2f}" for k, v in info_final_gen.items()))
            info_val = {**info_val, **info_final_gen}

        return info_val

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / "model_state.pt", "wb"),
        )

    def log(self, info, index):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            # self._summary_writer.add_scalar(f"{key}_{k}", v, index)
            self._summary_writer.add_scalar(k, v, index)


def cycle(it):
    while True:
        for i in it:
            yield i
