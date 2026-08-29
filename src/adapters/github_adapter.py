"""GitHub implementation of :class:`IssueProviderAdapter`."""

from __future__ import annotations

import logging
from typing import Any

from github import Auth, Github, GithubException, UnknownObjectException

from adapters.base import AdapterError, IssueProviderAdapter
from config import Settings

logger = logging.getLogger(__name__)


class GitHubAdapter(IssueProviderAdapter):
    """Issue-provider adapter backed by the GitHub REST API via PyGithub."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the adapter with credentials and target repository.

        Args:
            settings: Application settings containing token and repo slug.
        """
        self._settings = settings
        self._github = Github(auth=Auth.Token(settings.github_token))
        self._repo = self._wrap(
            "connect",
            lambda: self._github.get_repo(settings.github_repo),
        )
        logger.info("GitHubAdapter connected to repo %s", settings.github_repo)

    def _format_github_error(self, action: str, exc: GithubException | UnknownObjectException) -> str:
        """Build a human-readable error message with actionable hints."""
        detail = self._github_exception_detail(exc)
        status = getattr(exc, "status", None)
        message = f"GitHub API error during {action} on {self._settings.github_repo}"
        if status is not None:
            message += f" (HTTP {status})"
        message += f": {detail}"
        if status == 404:
            message += (
                " Verify GITHUB_REPO is 'owner/repository' (case-sensitive name), "
                "the repository exists, and your GITHUB_TOKEN has access (repo scope "
                "for private repos)."
            )
        elif status == 401:
            message += " Verify GITHUB_TOKEN is valid and not expired."
        elif status == 403:
            message += (
                " Verify GITHUB_TOKEN has the required scopes (repo for private "
                "repositories, public_repo for public ones)."
            )
        return message

    @staticmethod
    def _github_exception_detail(exc: GithubException | UnknownObjectException) -> str:
        """Extract a safe human-readable message from a PyGithub exception."""
        data = getattr(exc, "data", None)
        if isinstance(data, dict):
            api_message = data.get("message")
            if api_message:
                return str(api_message)
        return str(exc)

    @staticmethod
    def _extract_commit_metadata(
        result: Any,
        *,
        file_path: str,
        branch_name: str,
    ) -> dict[str, Any]:
        """Return commit metadata from a PyGithub create/update_file response."""
        commit = result.get("commit") if isinstance(result, dict) else getattr(result, "commit", None)
        if commit is None:
            raise AdapterError(
                f"GitHub commit_file response for {file_path!r} on branch "
                f"{branch_name!r} did not include a commit object"
            )

        sha = getattr(commit, "sha", None)
        if not sha:
            raise AdapterError(
                f"GitHub commit_file response for {file_path!r} on branch "
                f"{branch_name!r} did not include a commit SHA"
            )

        url = getattr(commit, "html_url", None) or ""
        commit_message = ""
        nested_commit = getattr(commit, "commit", None)
        if nested_commit is not None:
            nested_message = getattr(nested_commit, "message", None)
            if nested_message:
                commit_message = str(nested_message)

        return {
            "sha": sha,
            "url": url,
            "message": commit_message,
        }

    def _wrap(self, action: str, fn: Any) -> Any:
        """Execute ``fn`` and translate GitHub API failures into ``AdapterError``."""
        try:
            return fn()
        except (GithubException, UnknownObjectException) as exc:
            message = self._format_github_error(action, exc)
            logger.error(message, exc_info=True)
            raise AdapterError(message) from exc

    def _issue(self, issue_number: int) -> Any:
        return self._wrap(
            f"get_issue({issue_number})",
            lambda: self._repo.get_issue(issue_number),
        )

    @staticmethod
    def _issue_to_dict(issue: Any) -> dict[str, Any]:
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "",
            "state": issue.state,
            "labels": [label.name for label in issue.labels],
            "url": issue.html_url,
        }

    @staticmethod
    def _comment_to_dict(comment: Any) -> dict[str, Any]:
        return {
            "id": comment.id,
            "body": comment.body,
            "author": comment.user.login if comment.user else "unknown",
            "created_at": comment.created_at.isoformat(),
            "url": comment.html_url,
        }

    def get_issue(self, issue_number: int) -> dict[str, Any]:
        """Fetch a GitHub issue by number."""
        issue = self._issue(issue_number)
        logger.debug("Fetched issue #%s: %s", issue_number, issue.title)
        return self._issue_to_dict(issue)

    def list_comments(self, issue_number: int) -> list[dict[str, Any]]:
        """List all comments on a GitHub issue."""
        issue = self._issue(issue_number)

        def _fetch() -> list[dict[str, Any]]:
            comments = issue.get_comments()
            return [self._comment_to_dict(c) for c in comments]

        result = self._wrap(f"list_comments({issue_number})", _fetch)
        logger.debug("Listed %d comment(s) on issue #%s", len(result), issue_number)
        return result

    def post_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        """Post a comment on a GitHub issue."""
        issue = self._issue(issue_number)
        comment = self._wrap(
            f"post_comment({issue_number})",
            lambda: issue.create_comment(body),
        )
        logger.info("Posted comment on issue #%s", issue_number)
        return self._comment_to_dict(comment)

    def add_label(self, issue_number: int, label: str) -> None:
        """Add a label to a GitHub issue."""
        issue = self._issue(issue_number)
        self._wrap(
            f"add_label({issue_number}, {label!r})",
            lambda: issue.add_to_labels(label),
        )
        logger.info("Added label %r to issue #%s", label, issue_number)

    def remove_label(self, issue_number: int, label: str) -> None:
        """Remove a label from a GitHub issue."""
        issue = self._issue(issue_number)
        self._wrap(
            f"remove_label({issue_number}, {label!r})",
            lambda: issue.remove_from_labels(label),
        )
        logger.info("Removed label %r from issue #%s", label, issue_number)

    def has_label(self, issue_number: int, label: str) -> bool:
        """Check whether a GitHub issue has a specific label."""
        issue = self._issue(issue_number)
        present = any(existing.name == label for existing in issue.labels)
        logger.debug(
            "Issue #%s has label %r: %s", issue_number, label, present
        )
        return present

    def create_branch(self, branch_name: str, *, from_ref: str = "main") -> None:
        """Create a branch from an existing ref."""
        def _create() -> None:
            source = self._repo.get_branch(from_ref)
            self._repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=source.commit.sha,
            )

        self._wrap(f"create_branch({branch_name!r}, from_ref={from_ref!r})", _create)
        logger.info("Created branch %r from %r", branch_name, from_ref)

    def get_file_content(self, repo_full_name: str, path: str, ref: str) -> str:
        """Fetch a file's current live content from GitHub at ``ref``."""
        normalized_path = path.replace("\\", "/").lstrip("/")
        if not normalized_path:
            raise AdapterError("path must be a non-empty repository-relative file path")

        def _fetch() -> str:
            repo = (
                self._repo
                if repo_full_name == self._settings.github_repo
                else self._github.get_repo(repo_full_name)
            )
            contents = repo.get_contents(normalized_path, ref=ref)
            if isinstance(contents, list):
                raise AdapterError(
                    f"Path {normalized_path!r} is a directory, not a file, "
                    f"in {repo_full_name}@{ref}"
                )
            try:
                return contents.decoded_content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AdapterError(
                    f"File {normalized_path!r} in {repo_full_name}@{ref} "
                    "is not valid UTF-8 text"
                ) from exc

        return self._wrap(
            f"get_file_content({repo_full_name}, {normalized_path!r}, ref={ref!r})",
            _fetch,
        )

    def commit_file(
        self,
        branch_name: str,
        file_path: str,
        content: str,
        message: str,
    ) -> dict[str, Any]:
        """Create or update a file on the given branch."""
        normalized_path = file_path.replace("\\", "/").lstrip("/")
        if not normalized_path:
            raise AdapterError("file_path must be a non-empty repository-relative path")

        def _commit() -> dict[str, Any]:
            try:
                existing = self._repo.get_contents(normalized_path, ref=branch_name)
                if isinstance(existing, list):
                    raise AdapterError(
                        f"Path {normalized_path!r} is a directory, not a file, "
                        f"on branch {branch_name!r}"
                    )
                result = self._repo.update_file(
                    path=normalized_path,
                    message=message,
                    content=content,
                    sha=existing.sha,
                    branch=branch_name,
                )
            except UnknownObjectException:
                result = self._repo.create_file(
                    path=normalized_path,
                    message=message,
                    content=content,
                    branch=branch_name,
                )

            return self._extract_commit_metadata(
                result,
                file_path=normalized_path,
                branch_name=branch_name,
            )

        metadata = self._wrap(
            f"commit_file({normalized_path!r}, branch={branch_name!r})",
            _commit,
        )
        logger.info(
            "Committed %r on branch %r (sha=%s)",
            normalized_path,
            branch_name,
            metadata["sha"],
        )
        return metadata

    def open_pr(
        self,
        title: str,
        body: str,
        head: str,
        *,
        base: str = "main",
    ) -> dict[str, Any]:
        """Open a pull request."""
        pr = self._wrap(
            f"open_pr({title!r}, head={head!r}, base={base!r})",
            lambda: self._repo.create_pull(title=title, body=body, head=head, base=base),
        )
        logger.info("Opened PR #%s: %s", pr.number, title)
        return {
            "number": pr.number,
            "url": pr.html_url,
            "title": pr.title,
            "state": pr.state,
        }
