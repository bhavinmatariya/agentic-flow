"""Core types shared across agentic-flow components.

Import from submodules directly (for example ``core.orchestrator``,
``core.models``). This package ``__init__`` intentionally avoids importing
``core.orchestrator`` so agent modules can load ``core.claude_client`` without
a circular import through ``agents``.
"""

from config import AgentClaudeConfig, Settings
from core.exceptions import (
    AdapterError,
    AgentError,
    EnvironmentError,
    EnvironmentSetupError,
    ToolError,
)
from core.models import (
    Approach,
    CodeMatch,
    ImplementationResult,
    Investigation,
    LinkedRepo,
    ParsedIntent,
    Proposal,
    RelevantFile,
    RepoConfig,
    ReviewResult,
)

__all__ = [
    "AdapterError",
    "AgentClaudeConfig",
    "AgentError",
    "Approach",
    "CodeMatch",
    "EnvironmentError",
    "EnvironmentSetupError",
    "ImplementationResult",
    "Investigation",
    "LinkedRepo",
    "ParsedIntent",
    "Proposal",
    "RelevantFile",
    "RepoConfig",
    "ReviewResult",
    "Settings",
    "ToolError",
]
