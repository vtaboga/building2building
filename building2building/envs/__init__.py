"""Environment factory module.

Provides :func:`make_env_from_config`, the canonical entry point for
building Gymnasium environments from :class:`~b2b.config.models.EnvBuildConfig`.
"""

from b2b.envs.factory import make_env_from_config

__all__ = ["make_env_from_config"]
