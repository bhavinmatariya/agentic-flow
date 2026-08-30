"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when required configuration values are missing or invalid."""


@dataclass(frozen=True)
class AgentClaudeConfig:
    """Per-agent Claude request defaults."""

    effort: str
    temperature: float
    max_tokens: int


_AGENT_DEFAULTS: dict[str, AgentClaudeConfig] = {
    "investigator": AgentClaudeConfig(effort="xhigh", temperature=0.0, max_tokens=16000),
    "proposer": AgentClaudeConfig(effort="high", temperature=0.0, max_tokens=8000),
    "response_parser": AgentClaudeConfig(effort="low", temperature=0.0, max_tokens=2000),
    "implementer": AgentClaudeConfig(effort="xhigh", temperature=0.0, max_tokens=16000),
    "reviewer": AgentClaudeConfig(effort="medium", temperature=0.0, max_tokens=8000),
    "task_decomposer": AgentClaudeConfig(effort="high", temperature=0.0, max_tokens=8000),
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables and optional .env file.

    All three core values are required. Missing variables produce a single clear
    error listing every key that was not found. Per-agent Claude settings can be
    overridden with ``{AGENT}_EFFORT``, ``{AGENT}_TEMPERATURE``, and
    ``{AGENT}_MAX_TOKENS`` environment variables.
    """

    anthropic_api_key: str
    github_token: str
    github_repo: str
    investigator: AgentClaudeConfig
    proposer: AgentClaudeConfig
    response_parser: AgentClaudeConfig
    implementer: AgentClaudeConfig
    reviewer: AgentClaudeConfig
    task_decomposer: AgentClaudeConfig
    reviewer_live_effort: str
    anthropic_fallback_model: str | None
    live_verification_enabled: bool

    @classmethod
    def from_env(cls, *, env_file: str | None = ".env") -> Settings:
        """Load settings from the environment, optionally reading a .env file first."""
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
                "Set them in their shell or in a .env file (see .env.example)."
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
            investigator=_load_agent_config("investigator"),
            proposer=_load_agent_config("proposer"),
            response_parser=_load_agent_config("response_parser"),
            implementer=_load_agent_config("implementer"),
            reviewer=_load_agent_config("reviewer"),
            task_decomposer=_load_agent_config("task_decomposer"),
            reviewer_live_effort=_env_str("REVIEWER_LIVE_EFFORT", "high"),
            anthropic_fallback_model=_optional_env_str(
                "ANTHROPIC_FALLBACK_MODEL",
                "claude-sonnet-4-20250514",
            ),
            live_verification_enabled=_env_bool("LIVE_VERIFICATION_ENABLED", False),
        )

    def agent_config(self, agent_type: str) -> AgentClaudeConfig:
        """Return Claude settings for a logical agent type."""
        normalized = agent_type.strip().lower()
        mapping = {
            "investigator": self.investigator,
            "proposer": self.proposer,
            "response_parser": self.response_parser,
            "implementer": self.implementer,
            "reviewer": self.reviewer,
            "task_decomposer": self.task_decomposer,
        }
        if normalized not in mapping:
            raise ConfigurationError(f"Unknown agent type for Claude config: {agent_type!r}")
        return mapping[normalized]

    @property
    def github_owner(self) -> str:
        """Repository owner segment of ``GITHUB_REPO``."""
        return self.github_repo.split("/", 1)[0]

    @property
    def github_repo_name(self) -> str:
        """Repository name segment of ``GITHUB_REPO``."""
        return self.github_repo.split("/", 1)[1]


def _load_agent_config(agent_type: str) -> AgentClaudeConfig:
    """Load one agent's Claude config from defaults plus env overrides."""
    defaults = _AGENT_DEFAULTS[agent_type]
    prefix = agent_type.upper()
    return AgentClaudeConfig(
        effort=_env_str(f"{prefix}_EFFORT", defaults.effort),
        temperature=_env_float(f"{prefix}_TEMPERATURE", defaults.temperature),
        max_tokens=_env_int(f"{prefix}_MAX_TOKENS", defaults.max_tokens),
    )


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key, "").strip()
    return value or default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a float, got: {raw!r}") from exc


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer, got: {raw!r}") from exc
    if parsed < 1:
        raise ConfigurationError(f"{key} must be >= 1, got: {parsed}")
    return parsed


def _optional_env_str(key: str, default: str | None) -> str | None:
    """Return env value when set; otherwise ``default`` (may be None)."""
    value = os.getenv(key, "").strip()
    if not value:
        return default
    return value


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{key} must be a boolean (true/false, 1/0, yes/no), got: {raw!r}"
    )
