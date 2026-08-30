"""Surgical file edits on GitHub branches via an issue-provider adapter."""

from __future__ import annotations

import logging

from typing import Final

from adapters.base import AdapterError, IssueProviderAdapter
from core.exceptions import ToolError
from utils.logger import get_logger


_NEW_FILE_OLD_STRING: Final[str] = "__NEW_FILE__"


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
        """Create or reuse ``agent/fix-issue-{issue_number}`` off the default branch.

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
            default_branch = self._adapter.get_default_branch()
            self._adapter.create_branch(branch_name, from_ref=default_branch)
            self._logger.info("Created branch %r from %r", branch_name, default_branch)
        except AdapterError as exc:
            if not _is_existing_branch_error(exc):
                raise ToolError(
                    _format_tool_error(
                        "create_branch",
                        exc,
                        context=f"branch={branch_name!r}",
                    )
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
        """Replace exactly one ``old_string`` occurrence, or create a new file.

        Pass ``old_string`` as an empty string to create a brand-new file whose
        full content is ``new_string``. For edits to existing files,
        ``old_string`` must occur exactly once in the live file on GitHub.

        Args:
            repo: GitHub slug containing the file.
            branch: Branch to read from and commit onto.
            path: Repository-relative file path.
            old_string: Exact text to find, or ``""`` to create a new file.
            new_string: Replacement text, or full file content when creating.
            commit_message: Git commit message for the change.

        Raises:
            ToolError: If the adapter repo does not match, ``old_string`` is
                missing or ambiguous, or the commit fails.
        """
        self._ensure_commit_target(repo)
        normalized_path = path.replace("\\", "/").lstrip("/")
        if not normalized_path:
            raise ToolError("path must be a non-empty repository-relative file path")

        if _is_new_file_edit(old_string):
            try:
                self._adapter.commit_file(
                    branch,
                    normalized_path,
                    new_string,
                    commit_message,
                )
            except AdapterError as exc:
                raise ToolError(
                    _format_tool_error(
                        "commit_file",
                        exc,
                        context=(
                            f"create {repo}:{normalized_path}@{branch}"
                        ),
                    )
                ) from exc
            self._logger.info(
                "Committed new file %r on branch %r",
                normalized_path,
                branch,
            )
            return

        try:
            content = self._adapter.get_file_content(repo, normalized_path, branch)
        except AdapterError as exc:
            raise ToolError(
                _format_tool_error(
                    "get_file_content",
                    exc,
                    context=f"{repo}:{normalized_path}@{branch}",
                )
            ) from exc

        updated_content = _apply_exact_replace(
            content,
            old_string=old_string,
            new_string=new_string,
            path=normalized_path,
        )
        self._commit_with_conflict_retry(
            repo=repo,
            branch=branch,
            path=normalized_path,
            content=updated_content,
            commit_message=commit_message,
            old_string=old_string,
            new_string=new_string,
        )

        self._logger.info(
            "Committed surgical edit to %r on branch %r",
            normalized_path,
            branch,
        )

    def _commit_with_conflict_retry(
        self,
        *,
        repo: str,
        branch: str,
        path: str,
        content: str,
        commit_message: str,
        old_string: str,
        new_string: str,
    ) -> None:
        """Commit once, re-fetch and retry on SHA/conflict errors."""
        try:
            self._adapter.commit_file(branch, path, content, commit_message)
            return
        except AdapterError as first_exc:
            if not _is_sha_conflict_error(first_exc):
                raise ToolError(
                    _format_tool_error(
                        "commit_file",
                        first_exc,
                        context=f"{path!r} on branch {branch!r}",
                    )
                ) from first_exc

        self._logger.warning(
            "Commit conflict on %r@%r; re-fetching file and retrying once: %s",
            path,
            branch,
            first_exc,
        )

        try:
            fresh_content = self._adapter.get_file_content(repo, path, branch)
        except AdapterError as read_exc:
            raise ToolError(
                "Commit failed with a SHA/conflict error and the automatic "
                f"re-fetch failed for {repo}:{path}@{branch}: "
                f"{_format_tool_error('get_file_content', read_exc)} "
                f"(original commit error: "
                f"{_format_tool_error('commit_file', first_exc)})"
            ) from read_exc

        try:
            retried_content = _apply_exact_replace(
                fresh_content,
                old_string=old_string,
                new_string=new_string,
                path=path,
            )
        except ToolError as replace_exc:
            raise ToolError(
                "Commit failed with a SHA/conflict error on "
                f"{path!r}; after re-fetch, the edit could not be reapplied: "
                f"{replace_exc} (original commit error: "
                f"{_format_tool_error('commit_file', first_exc)})"
            ) from replace_exc

        try:
            self._adapter.commit_file(branch, path, retried_content, commit_message)
        except AdapterError as retry_exc:
            raise ToolError(
                "Commit retry failed after re-fetching fresh content for "
                f"{path!r} on branch {branch!r}: "
                f"{_format_tool_error('commit_file', retry_exc)} "
                f"(original conflict: {_format_tool_error('commit_file', first_exc)})"
            ) from retry_exc

    def _ensure_commit_target(self, repo: str) -> None:
        """Reject commits when the adapter is bound to a different repository."""
        settings = getattr(self._adapter, "_settings", None)
        configured_repo = getattr(settings, "github_repo", None)
        if configured_repo and repo != configured_repo:
            raise ToolError(
                f"Cannot commit to {repo!r}; adapter is bound to {configured_repo!r}"
            )

    def file_exists_on_branch(self, repo: str, branch: str, path: str) -> bool:
        """Return True when ``path`` exists on ``branch`` in ``repo``."""
        normalized_path = path.replace("\\", "/").lstrip("/")
        if not normalized_path:
            return False
        try:
            self._adapter.get_file_content(repo, normalized_path, branch)
            return True
        except AdapterError:
            return False


def _is_new_file_edit(old_string: str) -> bool:
    """Return True when ``edit_file`` should create a file instead of patching."""
    return old_string == "" or old_string == _NEW_FILE_OLD_STRING


def _apply_exact_replace(
    content: str,
    *,
    old_string: str,
    new_string: str,
    path: str,
) -> str:
    """Replace exactly one ``old_string`` occurrence in ``content``."""
    match_count = content.count(old_string)
    if match_count == 0:
        raise ToolError(
            f"old_string was not found in {path!r}. Read the current file "
            "content and provide an exact substring to replace."
        )
    if match_count > 1:
        raise ToolError(
            f"old_string matched {match_count} times in {path!r}. Provide more "
            "surrounding context so the match is unique."
        )
    return content.replace(old_string, new_string, 1)


def _format_tool_error(
    operation: str,
    exc: Exception,
    *,
    context: str | None = None,
) -> str:
    """Format an exception for tool_result payloads."""
    prefix = f"{operation} failed"
    if context:
        prefix += f" for {context}"
    return f"{prefix}: {type(exc).__name__}: {exc}"


def _is_existing_branch_error(exc: AdapterError) -> bool:
    """Return True when ``exc`` indicates the branch ref already exists."""
    message = str(exc).lower()
    return "already exists" in message or "reference already exists" in message


def _is_sha_conflict_error(exc: AdapterError) -> bool:
    """Return True when ``exc`` looks like a stale-SHA or merge conflict."""
    message = str(exc).lower()
    return (
        "http 409" in message
        or "status 409" in message
        or "sha" in message
        or "conflict" in message
        or "does not match" in message
    )
