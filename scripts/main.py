import hydra
from pathlib import Path

from b2b.baselines.online_trainer import online_trainer


@hydra.main(config_path="../configs", config_name="base")
def main(cfg):
	output_dir = Path.cwd()
	online_trainer(config=cfg, output_dir=output_dir)


if __name__ == "__main__":
	main()

