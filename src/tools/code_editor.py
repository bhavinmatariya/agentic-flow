"""Surgical file edits on GitHub branches via an issue-provider adapter."""

from __future__ import annotations

import logging

from adapters.base import AdapterError, IssueProviderAdapter
from core.exceptions import ToolError
from utils.logger import get_logger


class CodeEditTool:
    """Create fix branches and apply exact-match edits through an adapter.

    Edits fetch live file content from GitHub, replace a single unique
    ``old_string`` occurrence, and commit through
    :meth:`IssueProviderAdapter.commit_file`.
    """

    def __init__(
        self,
        adapter: IssueProviderAdapter,
        logger: logging.Logger | None = None,
    ) -> None:
        """Bind this tool to an issue-provider adapter.

        Args:
            adapter: Provider used to read files, create branches, and commit.
            logger: Optional logger. When omitted, the shared module logger
                is used.
        """
        self._adapter = adapter
        self._logger = logger or get_logger(__name__)

    def start_branch(self, repo: str, issue_number: int) -> str:
        """Create or reuse ``agent/fix-issue-{issue_number}`` off ``main``.

        Args:
            repo: GitHub slug in ``owner/repository`` form (used for logging
                context; branch creation uses the adapter's configured repo).
            issue_number: GitHub issue number that names the branch.

        Returns:
            The branch name.

        Raises:
            ToolError: If branch creation fails for a reason other than the
                branch already existing.
        """
        branch_name = f"agent/fix-issue-{issue_number}"
        self._logger.info(
            "Ensuring branch %r exists for issue #%s in %s",
            branch_name,
            issue_number,
            repo,
        )
        try:
            self._adapter.create_branch(branch_name, from_ref="main")
            self._logger.info("Created branch %r", branch_name)
        except AdapterError as exc:
            if not _is_existing_branch_error(exc):
                raise ToolError(
                    f"Could not create branch {branch_name!r}: {exc}"
                ) from exc
            self._logger.info("Branch %r already exists; reusing it", branch_name)
        return branch_name

    def edit_file(
        self,
        repo: str,
        branch: str,
        path: str,
        old_string: str,
        new_string: str,
        commit_message: str,
    ) -> None:
        """Replace exactly one ``old_string`` occurrence and commit the file.

        Args:
            repo: GitHub slug containing the file.
            branch: Branch to read from and commit onto.
            path: Repository-relative file path.
            old_string: Exact text to find; must occur exactly once.
            new_string: Replacement text for that single occurrence.
            commit_message: Git commit message for the change.

        Raises:
            ToolError: If the adapter repo does not match, ``old_string`` is
                missing or ambiguous, or the commit fails.
        """
        self._ensure_commit_target(repo)
        normalized_path = path.replace("\\", "/").lstrip("/")
        if not normalized_path:
            raise ToolError("path must be a non-empty repository-relative file path")
        if not old_string:
            raise ToolError("old_string must not be empty")

        try:
            content = self._adapter.get_file_content(repo, normalized_path, branch)
        except AdapterError as exc:
            raise ToolError(
                f"Could not read {repo}:{normalized_path}@{branch}: {exc}"
            ) from exc

        match_count = content.count(old_string)
        if match_count == 0:
            raise ToolError(
                f"old_string was not found in {normalized_path!r}. Read the "
                "current file content and provide an exact substring to replace."
            )
        if match_count > 1:
            raise ToolError(
                f"old_string matched {match_count} times in {normalized_path!r}. "
                "Provide more surrounding context so the match is unique."
            )

        updated_content = content.replace(old_string, new_string, 1)
        try:
            self._adapter.commit_file(
                branch,
                normalized_path,
                updated_content,
                commit_message,
            )
        except AdapterError as exc:
            raise ToolError(
                f"Could not commit {normalized_path!r} on branch {branch!r}: {exc}"
            ) from exc

        self._logger.info(
            "Committed surgical edit to %r on branch %r",
            normalized_path,
            branch,
        )

    def _ensure_commit_target(self, repo: str) -> None:
        """Reject commits when the adapter is bound to a different repository."""
        settings = getattr(self._adapter, "_settings", None)
        configured_repo = getattr(settings, "github_repo", None)
        if configured_repo and repo != configured_repo:
            raise ToolError(
                f"Cannot commit to {repo!r}; adapter is bound to {configured_repo!r}"
            )


def _is_existing_branch_error(exc: AdapterError) -> bool:
    """Return True when ``exc`` indicates the branch ref already exists."""
    message = str(exc).lower()
    return "already exists" in message or "reference already exists" in message
