from algorithms.test import test_policy
from algorithms.baselines import ConstantPolicy
from algorithms.utils import make_env
import hydra
import wandb
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg):


    # wandb_run = wandb.init(
    #     project=cfg.get('wandb_project'),
    #     config=OmegaConf.to_container(cfg, resolve=True),
	# 	sync_tensorboard=True,
    #     dir=str(Path.cwd())
    # )

    policy = ConstantPolicy()

    output_dir = Path.cwd()
    test_policy(cfg, policy, output_dir)

if __name__ == "__main__":
    main()

