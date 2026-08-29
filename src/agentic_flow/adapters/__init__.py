"""Issue-provider adapters for external ticketing platforms."""

from agentic_flow.adapters.base import AdapterError, IssueProviderAdapter
from agentic_flow.adapters.github_adapter import GitHubAdapter

__all__ = [
    "AdapterError",
    "GitHubAdapter",
    "IssueProviderAdapter",
]
