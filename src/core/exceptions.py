"""Shared exception types for expected, recoverable failures.

``AdapterError`` is defined on the Step 1 adapter contract and re-exported here
so callers can import adapter and tool failures from one module. ``ToolError``
covers context-gathering operations (clone, search, file I/O). ``AgentError``
covers Claude API failures and invalid agent output.
"""

from __future__ import annotations

from adapters.base import AdapterError

__all__ = ["AdapterError", "AgentError", "ToolError"]


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
