"""Implementer agent: apply an approved fix approach via surgical code edits."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.exceptions import AgentError, ToolError
from core.models import Approach, ImplementationResult, Investigation, Subtask
from core.repository_session import RepositorySession
from tools.code_editor import CodeEditTool
from tools.code_search import CodeSearchTool
from tools.environment_manager import checkout_git_branch

IMPLEMENTER_SYSTEM_PROMPT: Final[str] = (
    "You are implementing an already-approved fix. You have been given "
    "the issue, the investigation, and the specific approach a human "
    "approved. Follow these rules strictly:\n"
    "1. Change ONLY what is necessary to implement the approved approach. "
    "Do not touch, reformat, or 'clean up' any code unrelated to this "
    "specific fix, even if you notice other issues nearby.\n"
    "2. Match the existing codebase's style exactly — indentation, "
    "naming conventions, quote style, comment style. Your change "
    "should look like it was written by the same person who wrote "
    "the surrounding code, not a rewrite.\n"
    "3. Before editing, read the current file content to confirm it "
    "still matches what the investigation described — code may have "
    "changed since then.\n"
    "4. Make each distinct change as its own precise edit_file call with "
    "a unique old_string, rather than one large replacement.\n"
    "5. To create a brand-new file, call edit_file with old_string set to "
    "the empty string \"\" and new_string set to the full file content. "
    "Do not use any other sentinel for new files.\n"
    "6. If something described in the approach no longer matches the "
    "current code, stop and explain what's different rather than "
    "guessing.\n"
    "7. If you are given issues_found from a previous review round, you "
    "must address each one with a concrete code change via edit_file. If "
    "read_file shows the branch already satisfies a finding, you may skip "
    "that edit but must still call edit_file for any finding that is not "
    "yet satisfied — never return final JSON with open findings and zero "
    "successful edit_file calls unless a prior round already committed the "
    "required files and you verified each finding with read_file.\n"
    "8. Before finishing any change, check whether your new code creates "
    "two competing implementations of the same behavior (conflicting CSS "
    "rules, duplicate state, overlapping logic, etc.). If so, resolve the "
    "conflict directly — remove or override the losing one — rather than "
    "leaving both in place and hoping yours wins.\n"
    "9. When a CURRENT SUBTASK block is provided, implement ONLY that subtask. "
    "Do not implement later subtasks or out-of-scope parts of the full "
    "approach in this session.\n"
    "10. Every path in files_changed MUST be committed on GitHub via a "
    "successful edit_file call in this session before you return final JSON. "
    "Never claim a file was added or changed without a successful edit_file "
    "response {\"status\": \"committed\"}. Never describe work in summary that "
    "is not reflected in files_changed AND on the branch via edit_file. If "
    "edit_file returns an error, fix and retry — do not return success JSON anyway.\n"
    "11. read_file returns LIVE content from the working branch on GitHub — "
    "use that exact text as old_string for edit_file. If edit_file says "
    "old_string was not found, call read_file again on that path before retrying.\n\n"
    "When finished, respond with ONLY JSON: {\"branch_name\": str, "
    "\"files_changed\": [str], \"summary\": str}"
)

_FORCE_EDIT_RETRY_SUFFIX: Final[str] = (
    "\n\nCRITICAL RETRY: Your previous JSON was rejected because no edit_file "
    "call returned {\"status\": \"committed\"} while review findings were still "
    "open. Call edit_file now for each remaining finding before returning JSON."
)


IMPLEMENTER_TOOL_DEFINITIONS: Final[list[dict[str, Any]]] = [
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the local checkout of the primary "
            "repository. Use this to verify current code before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the local checkout provided in your "
                        "instructions."
                    ),
                },
                "relative_path": {
                    "type": "string",
                    "description": (
                        "Path to the file inside the checkout, relative to the "
                        "repository root (POSIX separators preferred)."
                    ),
                },
            },
            "required": ["local_repo_path", "relative_path"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search the local checkout for a literal (non-regex) string. "
            "Returns a JSON array of objects with file, line_number, and line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the local checkout provided in your "
                        "instructions."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Literal text to find. Special characters are matched "
                        "as written, not as a regular expression."
                    ),
                },
            },
            "required": ["local_repo_path", "query"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Surgically replace exactly one occurrence of old_string in an "
            "existing file on the working branch, or create a brand-new file. "
            "For existing files, old_string must appear exactly once in the "
            "current live file content on GitHub. To create a new file, set "
            "old_string to the empty string \"\" and put the full file content "
            "in new_string. Never rewrite a whole existing file in one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": (
                        "GitHub repository slug in 'owner/repository' form."
                    ),
                },
                "branch": {
                    "type": "string",
                    "description": "Working branch name provided in your instructions.",
                },
                "path": {
                    "type": "string",
                    "description": "Repository-relative path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "Exact substring to replace in an existing file. Must "
                        "match exactly once in the current file. Use the empty "
                        "string \"\" to create a brand-new file; then "
                        "new_string must contain the full file content."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text for that single occurrence.",
                },
                "commit_message": {
                    "type": "string",
                    "description": "Short commit message describing this edit.",
                },
            },
            "required": [
                "repo",
                "branch",
                "path",
                "old_string",
                "new_string",
                "commit_message",
            ],
        },
    },
]


class ImplementerAgent(BaseAgent):
    """Implement an approved :class:`Approach` using read/search and surgical edits."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        settings: Settings,
        code_search: CodeSearchTool,
        code_edit: CodeEditTool,
        github_token: str,
    ) -> None:
        """Create an implementer bound to search and edit tools.

        Args:
            client: Authenticated Anthropic SDK client.
            model: Model identifier passed to the centralized Claude client.
            settings: Application settings with per-agent Claude defaults.
            code_search: Tool for reading and searching a local checkout.
            code_edit: Tool for branch creation and GitHub commits.
            github_token: Token with read access for cloning the primary repo.
        """
        super().__init__(client, model, settings, "implementer")
        self._code_search = code_search
        self._code_edit = code_edit
        self._github_token = github_token.strip()
        if not self._github_token:
            raise AgentError("github_token must not be empty")
        self._max_tool_turns = 40
        self.system_prompt = IMPLEMENTER_SYSTEM_PROMPT
        self.tool_definitions = list(IMPLEMENTER_TOOL_DEFINITIONS)
        self._committed_paths: set[str] = set()
        self._last_committed_paths: frozenset[str] = frozenset()
        self._working_repo: str = ""
        self._working_branch: str = ""
        self._local_repo_path: str = ""
        self._edit_failures: dict[str, int] = {}
        self._active_repo_session: RepositorySession | None = None

    @property
    def last_committed_paths(self) -> frozenset[str]:
        """Paths successfully committed via edit_file in the last implement() call."""
        return self._last_committed_paths

    def create_repository_session(
        self,
        primary_repo: str,
        branch_name: str,
    ) -> RepositorySession:
        """Open a single local checkout for an orchestrator run."""
        return RepositorySession.open(
            code_search=self._code_search,
            repo_full_name=primary_repo,
            branch_name=branch_name,
            github_token=self._github_token,
            logger=self._logger,
        )

    def files_exist_on_branch(
        self,
        repo: str,
        branch: str,
        paths: list[str],
    ) -> bool:
        """Return True when every path exists on the GitHub branch."""
        if not paths:
            return False
        return all(
            self._code_edit.file_exists_on_branch(
                repo,
                branch,
                _normalize_repo_path(path),
            )
            for path in paths
        )

    def implement(
        self,
        issue: dict[str, Any],
        investigation: Investigation,
        approach: Approach,
        primary_repo: str,
        *,
        human_approval_text: str | None = None,
        review_findings: list[str] | None = None,
        attempt_failure_note: str | None = None,
        existing_branch: str | None = None,
        subtask: Subtask | None = None,
        subtask_index: int | None = None,
        subtask_total: int | None = None,
        repo_session: RepositorySession | None = None,
        baseline_implementation: ImplementationResult | None = None,
        require_fresh_commits: bool = False,
    ) -> ImplementationResult:
        """Apply ``approach`` on a dedicated branch and return the outcome.

        Args:
            issue: Issue dict from the adapter with at least ``number``,
                ``title``, and ``body``.
            investigation: Prior investigation findings.
            approach: Human-approved approach to implement.
            primary_repo: GitHub slug of the repository that owns the issue.
            human_approval_text: Optional human approval comment text.
            review_findings: Exact reviewer findings from the prior round when
                re-implementing after a failed review.
            attempt_failure_note: Optional note from the immediately prior round
                when implement or review raised an error.
            existing_branch: When resuming after ``agent:needs-human``, reuse this
                branch instead of creating a new fix branch from main.

        Returns:
            Validated implementation metadata including branch and files changed.

        Raises:
            AgentError: If branch setup, cloning, Claude, or validation fails.
        """
        issue_number = int(issue["number"])
        branch_name = self._code_edit.start_branch(primary_repo, issue_number)
        if existing_branch and existing_branch.strip():
            requested = existing_branch.strip()
            if requested != branch_name:
                self._logger.warning(
                    "Ignoring existing_branch %r; using %r",
                    requested,
                    branch_name,
                )
            else:
                self._logger.info(
                    "Continuing implementation on branch %r for issue #%s",
                    branch_name,
                    issue_number,
                )
        session_parts = [f"issue #{issue_number}"]
        if subtask_index is not None and subtask_total is not None:
            session_parts.append(f"subtask {subtask_index}/{subtask_total}")
        self._run_session_label = " · ".join(session_parts)
        self._committed_paths = set()
        self._edit_failures = {}
        self._working_repo = primary_repo
        self._working_branch = branch_name
        self._active_repo_session = repo_session
        try:
            if repo_session is not None:
                local_repo_path = repo_session.local_repo_path
                self._local_repo_path = local_repo_path
            else:
                local_repo_path = self._code_search.clone_repo(
                    primary_repo,
                    self._github_token,
                )
                self._local_repo_path = local_repo_path
                checkout_git_branch(
                    local_repo_path,
                    branch_name,
                    primary_repo,
                    self._github_token,
                    logger=self._logger,
                )
            user_message = self._build_user_message(
                issue=issue,
                investigation=investigation,
                approach=approach,
                primary_repo=primary_repo,
                branch_name=branch_name,
                local_repo_path=local_repo_path,
                human_approval_text=human_approval_text,
                review_findings=review_findings,
                attempt_failure_note=attempt_failure_note,
                subtask=subtask,
                subtask_index=subtask_index,
                subtask_total=subtask_total,
                baseline_implementation=baseline_implementation,
            )
            result = self.run(user_message, ImplementationResult)
            if result.branch_name != branch_name:
                self._logger.warning(
                    "Model returned branch_name=%r; using %r",
                    result.branch_name,
                    branch_name,
                )
                result = result.model_copy(update={"branch_name": branch_name})
            try:
                return self._validate_implementation(
                    result,
                    review_findings=review_findings,
                    subtask=subtask,
                    attempt_failure_note=attempt_failure_note,
                    baseline_implementation=baseline_implementation,
                    require_fresh_commits=require_fresh_commits,
                )
            except AgentError as exc:
                if not (
                    _is_no_edit_file_error(exc)
                    and review_findings
                    and not _is_reviewer_infra_retry(attempt_failure_note)
                ):
                    raise
                if (
                    baseline_implementation is not None
                    and self._baseline_files_on_branch(baseline_implementation)
                ):
                    self._logger.info(
                        "No new edit_file commits; prior round files remain on branch "
                        "— allowing re-review."
                    )
                    return self._merge_with_baseline(result, baseline_implementation)
                self._logger.warning(
                    "Implementer returned no edit_file commits with open findings; "
                    "retrying once with forced edit_file instruction."
                )
                retry_result = self.run(
                    user_message + _FORCE_EDIT_RETRY_SUFFIX,
                    ImplementationResult,
                )
                if retry_result.branch_name != branch_name:
                    retry_result = retry_result.model_copy(
                        update={"branch_name": branch_name}
                    )
                return self._validate_implementation(
                    retry_result,
                    review_findings=review_findings,
                    subtask=subtask,
                    attempt_failure_note=attempt_failure_note,
                    baseline_implementation=baseline_implementation,
                    strict_commits=True,
                    require_fresh_commits=require_fresh_commits,
                )
        finally:
            self._last_committed_paths = frozenset(self._committed_paths)
            self._run_session_label = None
            self._committed_paths = set()
            self._edit_failures = {}
            self._working_repo = ""
            self._working_branch = ""
            self._local_repo_path = ""
            self._active_repo_session = None

    def _build_user_message(
        self,
        *,
        issue: dict[str, Any],
        investigation: Investigation,
        approach: Approach,
        primary_repo: str,
        branch_name: str,
        local_repo_path: str,
        human_approval_text: str | None = None,
        review_findings: list[str] | None = None,
        attempt_failure_note: str | None = None,
        subtask: Subtask | None = None,
        subtask_index: int | None = None,
        subtask_total: int | None = None,
        baseline_implementation: ImplementationResult | None = None,
    ) -> str:
        """Assemble the user turn from issue, investigation, and approach context."""
        issue_body = str(issue.get("body") or "").strip() or "(empty)"
        if investigation.evidence:
            evidence_block = "\n".join(f"- {item}" for item in investigation.evidence)
        else:
            evidence_block = "(none)"
        if investigation.relevant_files:
            files_block = "\n".join(
                f"- {item.repo}:{item.path}\n  {item.reason}"
                for item in investigation.relevant_files
            )
        else:
            files_block = "(none)"
        if investigation.open_questions:
            questions_block = "\n".join(
                f"- {question}" for question in investigation.open_questions
            )
        else:
            questions_block = "(none)"
        approval_block = (
            str(human_approval_text or "").strip()
            or "(no human approval comment provided)"
        )

        message = (
            "Implement the approved fix approach.\n\n"
            f"Primary repository: {primary_repo}\n"
            f"Working branch: {branch_name}\n"
            f"Local checkout path: {local_repo_path}\n\n"
            f"Issue #{issue['number']}: {issue['title']}\n\n"
            f"Issue body:\n{issue_body}\n\n"
            f"Human approval / feedback (apply every literal detail from this text):\n"
            f"{approval_block}\n\n"
        )
        if review_findings:
            message += (
                "Previous review issues_found (address every item with edit_file, "
                "or read_file + verify on branch before concluding no change needed):\n"
                f"{json.dumps(review_findings, ensure_ascii=False, indent=2)}\n\n"
            )
        if baseline_implementation is not None and baseline_implementation.files_changed:
            prior_files = "\n".join(
                f"- {path}" for path in baseline_implementation.files_changed
            )
            message += (
                "Files already committed on this branch from a prior implement round:\n"
                f"{prior_files}\n\n"
                "If these commits already satisfy the open findings, verify with "
                "read_file and explain in summary — otherwise call edit_file for "
                "each remaining gap.\n\n"
            )
        if attempt_failure_note and attempt_failure_note.strip():
            message += (
                "Previous attempt failure (from the last round only — fix this "
                "before proceeding):\n"
                f"{attempt_failure_note.strip()}\n\n"
            )
        if subtask is not None:
            index_text = (
                f"{subtask_index}/{subtask_total}"
                if subtask_index is not None and subtask_total is not None
                else "?"
            )
            message += (
                f"CURRENT SUBTASK ({index_text}) — implement ONLY this step now:\n"
                f"- Name: {subtask.name}\n"
                f"- Description: {subtask.description}\n"
                f"- Scope: {subtask.scope}\n\n"
                "Do not implement other subtasks in this session.\n\n"
            )
        message += (
            f"Issue nature:\n{investigation.issue_nature}\n\n"
            f"Root cause:\n{investigation.root_cause}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Relevant files:\n{files_block}\n\n"
            f"Investigation confidence: {investigation.confidence}\n\n"
            f"Open questions:\n{questions_block}\n\n"
            "Approved approach:\n"
            f"- Name: {approach.name}\n"
            f"- Nature: {approach.nature}\n"
            f"- Description: {approach.description}\n"
            f"- Why it works: {approach.why_it_works}\n"
            f"- Risk: {approach.risk}\n"
            f"- Tradeoffs: {approach.tradeoffs}\n"
            f"- Estimated scope: {approach.estimated_scope}\n\n"
            "Use read_file to fetch LIVE file content from the working branch on "
            "GitHub (required for accurate old_string values). Use search_code "
            "against the local checkout to explore the codebase. Use edit_file "
            "values to commit surgical changes on GitHub. To create a new file, "
            "pass old_string as \"\" and new_string as the full file content.\n\n"
            f"Set branch_name to {branch_name!r} in your final JSON. "
            "Respond with only the JSON object specified in your instructions."
        )
        return message

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Route Claude tool calls to search/edit helpers."""
        try:
            if tool_name == "read_file":
                path = _normalize_repo_path(_require_str(tool_input, "relative_path"))
                if self._working_repo and self._working_branch:
                    try:
                        return self._code_edit.read_branch_file(
                            self._working_repo,
                            self._working_branch,
                            path,
                        )
                    except ToolError as exc:
                        return json.dumps(
                            {"error": f"{type(exc).__name__}: {exc}"},
                            ensure_ascii=False,
                        )
                return self._code_search.read_file(
                    _require_str(tool_input, "local_repo_path"),
                    path,
                )

            if tool_name == "search_code":
                matches = self._code_search.search_code(
                    _require_str(tool_input, "local_repo_path"),
                    _require_str(tool_input, "query"),
                )
                return json.dumps(
                    [match.model_dump() for match in matches],
                    ensure_ascii=False,
                )

            if tool_name == "edit_file":
                path = _normalize_repo_path(_require_str(tool_input, "path"))
                try:
                    self._code_edit.edit_file(
                        _require_str(tool_input, "repo"),
                        _require_str(tool_input, "branch"),
                        path,
                        _require_str(tool_input, "old_string"),
                        tool_input.get("new_string", ""),
                        _require_str(tool_input, "commit_message"),
                    )
                except ToolError as exc:
                    failures = self._edit_failures.get(path, 0) + 1
                    self._edit_failures[path] = failures
                    hint = (
                        " Call read_file on this path to load current GitHub "
                        "branch content, then retry edit_file with an exact "
                        "old_string match."
                    )
                    if failures >= 3:
                        hint += (
                            " This path failed 3+ times — simplify the edit "
                            "or use smaller old_string snippets with more context."
                        )
                    return json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}{hint}"},
                        ensure_ascii=False,
                    )
                self._committed_paths.add(path)
                self._edit_failures.pop(path, None)
                self._refresh_local_checkout()
                return json.dumps(
                    {"status": "committed", "path": path},
                    ensure_ascii=False,
                )
        except ToolError as exc:
            return json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

        return json.dumps(
            {"error": f"Unknown tool: {tool_name!r}"},
            ensure_ascii=False,
        )

    def _refresh_local_checkout(self) -> None:
        """Sync the shared checkout after a GitHub commit."""
        if self._active_repo_session is not None:
            try:
                self._active_repo_session.sync(logger=self._logger)
            except Exception as exc:
                self._logger.warning(
                    "Could not sync repository session after commit: %s",
                    exc,
                )
            return
        if not self._local_repo_path or not self._working_branch:
            return
        try:
            checkout_git_branch(
                self._local_repo_path,
                self._working_branch,
                self._working_repo,
                self._github_token,
                logger=self._logger,
            )
        except Exception as exc:
            self._logger.warning(
                "Could not refresh local checkout after commit: %s",
                exc,
            )

    def _validate_implementation(
        self,
        result: ImplementationResult,
        *,
        review_findings: list[str] | None,
        subtask: Subtask | None,
        attempt_failure_note: str | None = None,
        baseline_implementation: ImplementationResult | None = None,
        strict_commits: bool = False,
        require_fresh_commits: bool = False,
    ) -> ImplementationResult:
        """Ensure claimed files exist on GitHub and commits match reality."""
        repo = self._working_repo
        branch = self._working_branch
        if not repo or not branch:
            return result

        claimed = [_normalize_repo_path(path) for path in result.files_changed]
        missing = [
            path
            for path in claimed
            if not self._code_edit.file_exists_on_branch(repo, branch, path)
        ]
        if missing:
            raise AgentError(
                "Implementation JSON lists file(s) not present on GitHub branch "
                f"{branch!r}: {missing}. Call edit_file until each path returns "
                '{"status": "committed"} before returning final JSON.'
            )

        if (
            subtask is not None
            and not self._committed_paths
            and not claimed
            and (require_fresh_commits or strict_commits or bool(review_findings))
        ):
            raise AgentError(
                f"Subtask {subtask.name!r} returned empty files_changed with no "
                "successful edit_file commits. Call edit_file for each required "
                "file before returning final JSON."
            )

        if claimed and not self._committed_paths:
            uncommitted = [
                path
                for path in claimed
                if not self._code_edit.file_exists_on_branch(repo, branch, path)
            ]
            if uncommitted:
                raise AgentError(
                    "files_changed lists paths that are not on GitHub and were not "
                    f"committed this session: {uncommitted}. Call edit_file for each."
                )

        if require_fresh_commits and not self._committed_paths and not strict_commits:
            if not claimed or not self.files_exist_on_branch(repo, branch, claimed):
                raise AgentError(
                    f"Subtask {subtask.name if subtask else 'work'} requires at least "
                    "one successful edit_file commit this session."
                )

        require_commit_for_findings = bool(review_findings) and not _is_reviewer_infra_retry(
            attempt_failure_note
        ) and not _findings_indicate_phantom_implement(review_findings)
        if (
            require_commit_for_findings
            and not self._committed_paths
            and not strict_commits
            and baseline_implementation is not None
            and self._baseline_files_on_branch(baseline_implementation)
        ):
            self._logger.info(
                "Skipping edit_file requirement; prior round commits exist on branch."
            )
            require_commit_for_findings = False

        if require_commit_for_findings and not self._committed_paths:
            raise AgentError(
                "Reviewer findings require code changes but no successful edit_file "
                "commit was made this session. Address each finding with edit_file."
            )

        if (
            subtask is not None
            and not claimed
            and not self._committed_paths
            and require_commit_for_findings
        ):
            raise AgentError(
                f"Subtask {subtask.name!r} still has open reviewer findings but "
                "no edit_file commit was made. Fix each finding with edit_file."
            )

        if self._committed_paths:
            merged_paths = sorted(set(claimed) | self._committed_paths)
            if set(merged_paths) != set(result.files_changed):
                result = result.model_copy(update={"files_changed": merged_paths})
        elif claimed:
            result = result.model_copy(update={"files_changed": sorted(set(claimed))})
        if baseline_implementation is not None:
            result = self._merge_with_baseline(result, baseline_implementation)
        return result

    def _baseline_files_on_branch(
        self,
        baseline: ImplementationResult,
    ) -> bool:
        """Return True when every baseline file exists on the working branch."""
        repo = self._working_repo
        branch = self._working_branch
        if not repo or not branch or not baseline.files_changed:
            return False
        return all(
            self._code_edit.file_exists_on_branch(
                repo,
                branch,
                _normalize_repo_path(path),
            )
            for path in baseline.files_changed
        )

    @staticmethod
    def _merge_with_baseline(
        result: ImplementationResult,
        baseline: ImplementationResult,
    ) -> ImplementationResult:
        """Union file lists from the current and baseline implement rounds."""
        merged_files = sorted(
            set(result.files_changed) | set(baseline.files_changed)
        )
        summary = result.summary.strip() or baseline.summary
        return result.model_copy(update={"files_changed": merged_files, "summary": summary})


def _normalize_repo_path(path: str) -> str:
    """Normalize a repository-relative path."""
    return path.replace("\\", "/").lstrip("/")


def _findings_indicate_phantom_implement(findings: list[str] | None) -> bool:
    """Return True when reviewer reported claimed files/commits are missing."""
    if not findings:
        return False
    text = " ".join(findings).lower()
    markers = (
        "does not exist",
        "file not found",
        "not present",
        "not exist in",
        "hallucin",
        "not contain",
        "was not committed",
        "never committed",
        "not actually",
        "does not contain",
        "falsely",
        "incorrect/hallucinated",
    )
    return any(marker in text for marker in markers)


def _is_reviewer_infra_retry(note: str | None) -> bool:
    """Return True when the prior round failed in review infrastructure, not code."""
    if not note:
        return False
    lowered = note.lower()
    markers = (
        "reviewer could not finish",
        "stop_reason='refusal'",
        "stop_reason=\"refusal\"",
        "stop_reason=refusal",
        "content_filter",
        "reviewer failed",
        "exceeded the maximum",
    )
    return any(marker in lowered for marker in markers)


def _is_no_edit_file_error(exc: BaseException) -> bool:
    """Return True when validation failed due to missing edit_file commits."""
    message = str(exc).lower()
    return (
        "no successful edit_file" in message
        or "no edit_file commit" in message
    )


def _require_str(tool_input: dict[str, Any], key: str) -> str:
    """Return a required string argument from a tool-input dict."""
    value = tool_input.get(key)
    if not isinstance(value, str):
        raise ToolError(f"Tool argument {key!r} must be a string")
    if key == "old_string":
        return value
    if not value.strip():
        raise ToolError(f"Tool argument {key!r} must be a non-empty string")
    return value.strip()
