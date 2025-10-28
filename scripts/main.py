import hydra
import wandb
from pathlib import Path
from omegaconf import OmegaConf

from algorithms.online_trainer import online_trainer


@hydra.main(config_path="../configs", config_name="base")
def main(cfg):

	wandb_run = wandb.init(
        project=cfg.get('wandb_project'),
        config=OmegaConf.to_container(cfg, resolve=True),
		sync_tensorboard=True,
        dir=str(Path.cwd())
    )

	output_dir = Path.cwd()

	online_trainer(config=cfg, output_dir=output_dir, wandb_run=wandb_run)


if __name__ == "__main__":
	main()


# TODO: normalize obs