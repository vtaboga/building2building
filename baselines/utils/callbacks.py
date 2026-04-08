"""SB3 training callbacks for experiment tracking."""

from __future__ import annotations

import logging
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TrainingEpisodeRewardCallback(BaseCallback):
    """Log episode reward and length from SB3's Monitor wrapper to W&B."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._try_wandb_log(
                    {
                        "train/episode_reward": ep["r"],
                        "train/episode_length": ep["l"],
                    }
                )
        return True

    @staticmethod
    def _try_wandb_log(payload: dict[str, Any]) -> None:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(payload)
        except Exception:
            pass
