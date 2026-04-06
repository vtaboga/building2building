"""Environment factory module.

Provides :func:`make_env_from_config`, the canonical entry point for
building Gymnasium environments from :class:`~building2building.config.models.EnvBuildConfig`.
"""

from building2building.envs.factory import make_env_from_config

__all__ = ["make_env_from_config"]
