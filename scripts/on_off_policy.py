import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from algorithms.baselines import OnOffSensibleLoadPolicy
from algorithms.utils import make_env

logger = logging.getLogger(__name__)


def _find_zone_air_temp_index(env) -> int:
    names = []
    if hasattr(env, "metadata") and isinstance(env.metadata, dict):
        raw = env.metadata.get("observation_names")
        if isinstance(raw, list):
            names = [str(x) for x in raw]

    if not names:
        raise RuntimeError(
            "Could not find env.metadata['observation_names']; cannot locate zone air temperature."
        )

    for i, name in enumerate(names):
        if name.lower().startswith("zone air temperature"):
            return i

    for i, name in enumerate(names):
        if "zone air temperature" in name.lower():
            return i

    raise RuntimeError(
        "Could not find a 'Zone Air Temperature' entry in observation_names."
    )


@hydra.main(version_base=None, config_path="../configs", config_name="on_off")
def main(cfg) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    out_dir = Path.cwd() / "test"
    env = make_env(config=cfg, eplus_output_dir=str(out_dir / "eplus_outputs"))

    # Find observation index for zone temperature
    tz_idx = _find_zone_air_temp_index(env)
    logger.info(f"Using Zone Air Temperature observation index: {tz_idx}")

    policy = OnOffSensibleLoadPolicy(
        target_temp_c=float(cfg.policy.target_temp_c),
        deadband_c=float(cfg.policy.deadband_c),
        q_heat_w=float(cfg.policy.q_heat_w),
        q_cool_w=float(cfg.policy.q_cool_w),
        temp_obs_index=tz_idx,
    )

    for ep in range(int(cfg.n_episodes)):
        obs, _info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=float))
            total_reward += float(reward)
            done = bool(terminated or truncated)
            steps += 1

        logger.info(
            f"Episode {ep + 1} done - steps={steps} total_reward={total_reward} last_mode={policy.last_mode}"
        )

    env.close()


if __name__ == "__main__":
    main()


