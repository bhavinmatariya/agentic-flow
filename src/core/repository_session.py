"""Single clone + branch checkout reused for an entire orchestrator run."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.exceptions import EnvironmentSetupError
from tools.code_search import CodeSearchTool
from tools.environment_manager import checkout_git_branch
from utils.logger import get_logger


@dataclass
class RepositorySession:
    """One local checkout of ``repo_full_name`` at ``branch_name`` for a pipeline run."""

    local_repo_path: str
    repo_full_name: str
    branch_name: str
    github_token: str

    @classmethod
    def open(
        cls,
        *,
        code_search: CodeSearchTool,
        repo_full_name: str,
        branch_name: str,
        github_token: str,
        logger: logging.Logger | None = None,
    ) -> RepositorySession:
        """Clone (or reuse) the repo once and check out ``branch_name``."""
        log = logger or get_logger(__name__)
        local_repo_path = code_search.clone_repo(repo_full_name, github_token)
        checkout_git_branch(
            local_repo_path,
            branch_name,
            repo_full_name,
            github_token,
            logger=log,
        )
        log.info(
            "RepositorySession ready at %r (branch=%r, repo=%s)",
            local_repo_path,
            branch_name,
            repo_full_name,
        )
        return cls(
            local_repo_path=local_repo_path,
            repo_full_name=repo_full_name,
            branch_name=branch_name,
            github_token=github_token,
        )

    def sync(self, logger: logging.Logger | None = None) -> None:
        """Fetch the latest ``branch_name`` from GitHub into the local checkout."""
        log = logger or get_logger(__name__)
        try:
            checkout_git_branch(
                self.local_repo_path,
                self.branch_name,
                self.repo_full_name,
                self.github_token,
                logger=log,
            )
        except EnvironmentSetupError as exc:
            raise EnvironmentSetupError(
                f"Could not sync branch {self.branch_name!r} in "
                f"{self.local_repo_path!r}: {exc}"
            ) from exc
