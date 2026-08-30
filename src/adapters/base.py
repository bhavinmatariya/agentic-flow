"""Abstract base class for issue-provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AdapterError(Exception):
    """Raised when an adapter operation against an external provider fails."""


class IssueProviderAdapter(ABC):
    """Contract for interacting with an issue-tracking platform.

    Concrete implementations (e.g. ``GitHubAdapter``, a future ``JiraAdapter``)
    must supply every method below so orchestration code can remain provider-agnostic.
    """

    @abstractmethod
    def get_issue(self, issue_number: int) -> dict[str, Any]:
        """Fetch a single issue by its numeric identifier.

        Returns:
            A dictionary with at least ``number``, ``title``, ``body``, ``state``,
            and ``labels`` keys.
        """

    @abstractmethod
    def list_comments(self, issue_number: int) -> list[dict[str, Any]]:
        """Return all comments on the given issue, oldest first.

        Each comment dict contains at least ``id``, ``body``, ``author``, and
        ``created_at`` keys.
        """

    @abstractmethod
    def post_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        """Post a new comment on the issue.

        Returns:
            A dictionary describing the created comment (``id``, ``body``, etc.).
        """

    @abstractmethod
    def add_label(self, issue_number: int, label: str) -> None:
        """Add a label to the issue, creating the label on the repo if needed."""

    @abstractmethod
    def remove_label(self, issue_number: int, label: str) -> None:
        """Remove a label from the issue."""

    @abstractmethod
    def has_label(self, issue_number: int, label: str) -> bool:
        """Return ``True`` if the issue currently carries the given label."""

    @abstractmethod
    def create_branch(self, branch_name: str, *, from_ref: str = "main") -> None:
        """Create a new branch pointing at the tip of ``from_ref``."""

    @abstractmethod
    def get_default_branch(self) -> str:
        """Return the repository's default branch name (e.g. ``main`` or ``master``)."""

    @abstractmethod
    def get_file_content(self, repo_full_name: str, path: str, ref: str) -> str:
        """Fetch a file's live text content from a repository at ``ref``.

        Args:
            repo_full_name: Repository slug in ``owner/repository`` form.
            path: Repository-relative path to the file.
            ref: Branch name, tag, or commit SHA to read from.

        Returns:
            The file contents decoded as UTF-8 text.
        """

    @abstractmethod
    def commit_file(
        self,
        branch_name: str,
        file_path: str,
        content: str,
        message: str,
    ) -> dict[str, Any]:
        """Create or update a single file on ``branch_name``.

        Returns:
            A dictionary with commit metadata (``sha``, ``url``, etc.).
        """

    @abstractmethod
    def open_pr(
        self,
        title: str,
        body: str,
        head: str,
        *,
        base: str = "main",
    ) -> dict[str, Any]:
        """Open a pull request from ``head`` into ``base``.

        Returns:
            A dictionary with PR metadata (``number``, ``url``, ``title``, etc.).
        """
