from algorithms.test import test_policy
from algorithms.baselines import ConstantPolicy
import hydra
import wandb
from omegaconf import OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="test")
def main(cfg):


    wandb_run = wandb.init(
        project=cfg.get('wandb_project'),
        config=OmegaConf.to_container(cfg, resolve=True),
		sync_tensorboard=True,
        dir=str(Path.cwd())
    )

    heating = cfg.get("heating_setpoint", cfg.policy.heating_setpoint)
    delta = cfg.get("delta_setpoint", cfg.policy.delta_setpoint)

    policy = ConstantPolicy(heating_setpoint=heating, delta_setpoint=delta)
    output_dir = Path.cwd()
    test_policy(cfg, policy, output_dir)

if __name__ == "__main__":
    main()

