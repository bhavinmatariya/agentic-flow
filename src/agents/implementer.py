"""Implementer agent: apply an approved fix approach via surgical code edits."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.exceptions import AgentError, ToolError
from core.models import Approach, ImplementationResult, Investigation
from tools.code_editor import CodeEditTool
from tools.code_search import CodeSearchTool

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
    "must address each one with a concrete code change. You may only "
    "conclude that no changes are needed if issues_found is empty — never "
    "override or dismiss a specific finding from the reviewer without "
    "fixing it.\n"
    "8. Before finishing any change, check whether your new code creates "
    "two competing implementations of the same behavior (conflicting CSS "
    "rules, duplicate state, overlapping logic, etc.). If so, resolve the "
    "conflict directly — remove or override the losing one — rather than "
    "leaving both in place and hoping yours wins.\n\n"
    "When finished, respond with ONLY JSON: {\"branch_name\": str, "
    "\"files_changed\": [str], \"summary\": str}"
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
        self.system_prompt = IMPLEMENTER_SYSTEM_PROMPT
        self.tool_definitions = list(IMPLEMENTER_TOOL_DEFINITIONS)

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
        if existing_branch and existing_branch.strip():
            branch_name = existing_branch.strip()
            self._logger.info(
                "Resuming implementation on existing branch %r for issue #%s",
                branch_name,
                issue_number,
            )
        else:
            branch_name = self._code_edit.start_branch(primary_repo, issue_number)
        local_repo_path = self._code_search.clone_repo(
            primary_repo,
            self._github_token,
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
        )
        result = self.run(user_message, ImplementationResult)
        if result.branch_name != branch_name:
            self._logger.warning(
                "Model returned branch_name=%r; using %r",
                result.branch_name,
                branch_name,
            )
            result = result.model_copy(update={"branch_name": branch_name})
        return result

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
                "Previous review issues_found (address every item with a concrete "
                "code change):\n"
                f"{json.dumps(review_findings, ensure_ascii=False, indent=2)}\n\n"
            )
        if attempt_failure_note and attempt_failure_note.strip():
            message += (
                "Previous attempt failure (from the last round only — fix this "
                "before proceeding):\n"
                f"{attempt_failure_note.strip()}\n\n"
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
            "Use read_file and search_code against the local checkout to inspect "
            "code. Use edit_file with repo, branch, path, and exact old_string "
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
                return self._code_search.read_file(
                    _require_str(tool_input, "local_repo_path"),
                    _require_str(tool_input, "relative_path"),
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
                self._code_edit.edit_file(
                    _require_str(tool_input, "repo"),
                    _require_str(tool_input, "branch"),
                    _require_str(tool_input, "path"),
                    _require_str(tool_input, "old_string"),
                    tool_input.get("new_string", ""),
                    _require_str(tool_input, "commit_message"),
                )
                return json.dumps({"status": "committed"}, ensure_ascii=False)
        except ToolError as exc:
            return json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

        return json.dumps(
            {"error": f"Unknown tool: {tool_name!r}"},
            ensure_ascii=False,
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
