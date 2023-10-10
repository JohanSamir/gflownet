import os
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union
import wandb

import numpy as np
import wandb
from gflownet.utils.misc import prepend_keys
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.tasks.tdc_frag import TDCFragTrainer
from gflownet.config import Config
from gflownet.algo.config import TBVariant
import torch
import time


def main(hps, use_wandb=False):
    # hps must contain task.name, log_dir, overwrite_existing_exp

    # Measuring time
    start_time = time.time()

    if use_wandb:
        wandb.init(
            project=hps["log_dir"].split("/")[-2], name=hps["log_dir"].split("/")[-1], config=hps, sync_tensorboard=True
        )

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    Trainer = get_Trainer(hps)
    trial = Trainer(hps)
    info_final = trial.run()
    # if trial.cfg.num_final_gen_steps > 0:
    #     info_candidates = candidates_eval(final_gen_batches,path = trial.cfg.log_dir+"final", k=trial.cfg.evaluation.k, reward_thresh = trial.cfg.evaluation.reward_thresh, tanimoto_thresh=trial.cfg.evaluation.tanimoto_thresh)
    #     info_final = {**info_final,**info_candidates}

    if use_wandb:
        # wandb.log(prepend_keys(info_final,"final"))
        wandb.finish()

    print("\n\nFinal results:\n")
    print(info_final)

    print("\n\nTime elapsed: ", time.time() - start_time)
    return info_final


def get_Trainer(hps) -> Union[SEHFragTrainer, TDCFragTrainer]:
    if hps["task"]["name"] == "seh_frag":
        return SEHFragTrainer
    elif hps["task"]["name"] in ["tdc_frag"]:
        return TDCFragTrainer
    else:
        raise ValueError(f"Unknown task!")


if __name__ == "__main__":
    hps = {
        "log_dir": f"./logs/mol_eval",  # _{time.strftime('%Y-%m-%d_%H-%M-%S')
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 1000,  # 10_000,
        "print_every": 10,
        "validate_every": 100,
        "num_workers": 8,
        "num_final_gen_steps": 2,
        "opt": {
            "lr_decay": 20_000,
        },
        "algo": {
            "method": "SQL",
            "helper": "SQL",
            "sampling_tau": 0.99,
            "sample_temp": 1.0,
            "online_batch_size": 64,
            "replay_batch_size": 32,
            "offline_batch_size": 0,
            "valid_sample_cond_info": True,
            "tb": {
                "do_length_normalize": False,  ###TODO
                "variant": TBVariant.DB,
            },
        },
        "replay": {
            "use": True,
            "capacity": 100,  # 100,
            "warmup": 1,  # 10,
            "hindsight_ratio": 0.0,
            "insertion": {
                "strategy": "diversity_and_reward",  # "diversity_and_reward_fast",
                "sim_thresh": 0.7,
                "reward_thresh": 0.9,
            },
            "sampling": {
                "strategy": "uniform",
                "weighted": {
                    "reward_power": 1.0,
                },
                "quantile": {
                    "alpha": 0.1,
                    "beta": 0.5,
                },
            },
        },
        "cond": {
            # "temperature": {
            #     "sample_dist": "uniform",
            #     "dist_params": [0, 64.0],
            #     }
            "temperature": {
                "sample_dist": "discrete",  # "discrete", #"uniform" #"constant"
                "dist_params": [1 / 2, 1 / 4, 1 / 32],  # [16,32,64,96,128] ,#[0, 64.0] # [1.0]
                "num_thermometer_dim": 32,
            },
        },
        "task": {
            "name": "seh_frag",
            "helper": "qed_frag",
            "tdc": {
                "oracle": "qed",
            },
        },
        "evaluation": {
            "k": 10,
            "reward_thresh": 8.0,
            "tanimoto_thresh": 0.7,
        },
    }
    info_val = main(hps, use_wandb=False)
