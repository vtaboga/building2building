"""Shared SB3 training callbacks."""

import logging

import wandb
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

TRAIN_EP_REW_KEY = "train/ep_rew"
TRAIN_EP_LEN_KEY = "train/ep_len"


class TrainingEpisodeRewardCallback(BaseCallback):
    """Log individual training episode rewards under the ``train/`` namespace.

    SB3's built-in ``rollout/ep_rew_mean`` is a windowed rolling average over
    the last 100 episodes and does not distinguish training from evaluation.
    This callback reads the per-episode ``info["episode"]`` dict that SB3's
    ``Monitor`` wrapper injects at episode end and logs each completed training
    episode individually, making it straightforward to track learning progress
    separately from evaluation metrics.

    Logged keys (to wandb, stepped by ``num_timesteps``):
        ``train/ep_rew`` – total undiscounted reward of the completed episode.
        ``train/ep_len``  – number of environment steps in the episode.
    """

    def _on_step(self) -> bool:
        if wandb.run is None:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is None:
                continue
            wandb.log(
                {
                    TRAIN_EP_REW_KEY: float(ep["r"]),
                    TRAIN_EP_LEN_KEY: int(ep["l"]),
                }
            )
            if self.verbose >= 1:
                logger.debug(
                    "Training episode finished: reward=%.3f, length=%d",
                    float(ep["r"]),
                    int(ep["l"]),
                )

        return True
