"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when required configuration values are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables and optional .env file.

    All three values are required. Missing variables produce a single clear error
    listing every key that was not found.
    """

    anthropic_api_key: str
    github_token: str
    github_repo: str

    @classmethod
    def from_env(cls, *, env_file: str | None = ".env") -> Settings:
        """Load settings from the environment, optionally reading a .env file first.

        Args:
            env_file: Path to a dotenv file. Pass ``None`` to skip file loading.

        Returns:
            A validated ``Settings`` instance.

        Raises:
            ConfigurationError: If any required variable is absent or blank.
        """
        if env_file is not None:
            load_dotenv(env_file)

        required_keys = (
            "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN",
            "GITHUB_REPO",
        )
        missing = [key for key in required_keys if not os.getenv(key, "").strip()]

        if missing:
            joined = ", ".join(missing)
            raise ConfigurationError(
                f"Missing required environment variable(s): {joined}. "
                "Set them in your shell or in a .env file (see .env.example)."
            )

        github_repo = os.environ["GITHUB_REPO"].strip()
        if "/" not in github_repo or github_repo.count("/") != 1:
            raise ConfigurationError(
                "GITHUB_REPO must be in 'owner/repository' format, "
                f"got: {github_repo!r}"
            )

        return cls(
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"].strip(),
            github_token=os.environ["GITHUB_TOKEN"].strip(),
            github_repo=github_repo,
        )

    @property
    def github_owner(self) -> str:
        """Repository owner segment of ``GITHUB_REPO``."""
        return self.github_repo.split("/", 1)[0]

    @property
    def github_repo_name(self) -> str:
        """Repository name segment of ``GITHUB_REPO``."""
        return self.github_repo.split("/", 1)[1]
