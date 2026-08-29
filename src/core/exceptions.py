"""Shared exception types for expected, recoverable failures.

``AdapterError`` is defined on the Step 1 adapter contract and re-exported here
so callers can import adapter and tool failures from one module. ``ToolError``
covers context-gathering operations (clone, search, file I/O). ``AgentError``
covers Claude API failures and invalid agent output. ``EnvironmentSetupError``
covers disposable test-environment setup and teardown failures that must not be
treated as code-quality review failures.
"""

from __future__ import annotations

from adapters.base import AdapterError

__all__ = [
    "AdapterError",
    "AgentError",
    "EnvironmentError",
    "EnvironmentSetupError",
    "ToolError",
]


class ToolError(Exception):
    """Raised when a context-gathering tool operation fails.

    The original exception (subprocess failure, I/O error, validation error)
    should be attached with ``raise ToolError(...) from exc`` so callers can
    inspect both the human-readable message and the underlying cause.
    """


class AgentError(Exception):
    """Raised when an agent cannot complete a Claude call or validate its output.

    Attach the original exception with ``raise AgentError(...) from exc`` for API
    failures, JSON parse errors, and pydantic validation errors so callers can
    inspect both the message and the underlying cause.
    """


class EnvironmentSetupError(Exception):
    """Raised when disposable test-environment setup or teardown fails.

    This indicates an infrastructure or tooling problem (Docker unavailable,
    migrations failed, app health check timed out, DB connection errors), not
    a defect in the code under review. Reviewers and orchestrators should catch
    this separately from real verification failures.

    Attach the original exception with ``raise EnvironmentSetupError(...) from exc``
    so callers can inspect both the human-readable step context and the cause.
    """


# Backward-compatible alias used by earlier steps.
EnvironmentError = EnvironmentSetupError
