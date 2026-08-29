"""Issue-provider adapters for external ticketing platforms."""

from adapters.base import AdapterError, IssueProviderAdapter
from adapters.github_adapter import GitHubAdapter

__all__ = [
    "AdapterError",
    "GitHubAdapter",
    "IssueProviderAdapter",
]
