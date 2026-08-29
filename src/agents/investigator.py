"""Investigator agent: gather code evidence for a GitHub issue, do not propose a fix."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.exceptions import AgentError, ToolError
from core.models import Investigation
from tools.code_search import CodeSearchTool

INVESTIGATOR_SYSTEM_PROMPT: Final[str] = (
    "You are a senior software engineer investigating a GitHub issue "
    "before anyone proposes a fix. You have tools to clone a repository, "
    "list its files, read a file, and search for text across it. "
    "\n\n"
    "Do this: 1) Describe what this issue actually is, in your own "
    "words — do not force it into bug/feature/spike or any fixed "
    "category. 2) Investigate like a senior engineer: find the relevant "
    "code, trace what calls it, check related files, and check whether "
    "the real cause lives in a linked repo instead of this one if one is "
    "configured. 3) Form a root-cause hypothesis and verify it against "
    "what you actually find in the code — do not guess without evidence. "
    "4) When an issue involves UI, check what patterns the codebase already "
    "uses for similar things — existing dialog/modal components, "
    "notification/toast systems, form validation approaches, naming "
    "conventions, or architectural patterns (e.g. how DB access, error "
    "handling, or API routes are structured elsewhere). Note these "
    "established conventions explicitly in your evidence, since they "
    "constrain what a good solution looks like. The same applies to "
    "backend/DB conventions — how migrations are structured, how similar "
    "tables/queries are written elsewhere. "
    "5) Do not propose a fix. Only investigate."
    "\n\n"
    "Respond with ONLY a JSON object matching this shape: "
    '{"issue_nature": str, "root_cause": str, "evidence": [str], '
    '"relevant_files": [{"repo": str, "path": str, "reason": str}], '
    '"confidence": "high"|"medium"|"low", "open_questions": [str]}'
)

INVESTIGATOR_TOOL_DEFINITIONS: Final[list[dict[str, Any]]] = [
    {
        "name": "clone_repo",
        "description": (
            "Shallow-clone a GitHub repository identified by owner/repo into "
            "the local working directory. If that repo is already cloned, the "
            "existing local path is returned. Call this before list_files, "
            "read_file, or search_code. Returns the absolute local path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_full_name": {
                    "type": "string",
                    "description": (
                        "GitHub repository slug in 'owner/repository' form, "
                        "for example 'acme/payments'."
                    ),
                },
            },
            "required": ["repo_full_name"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in a local checkout previously returned by clone_repo. "
            "Optionally restrict to file extensions. Returns a JSON array of "
            "repository-relative paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the local checkout, as returned by "
                        "clone_repo."
                    ),
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional file suffixes to keep, with or without a "
                        "leading dot (e.g. ['.py', 'ts']). Omit to list every "
                        "file."
                    ),
                },
            },
            "required": ["local_repo_path"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from a local checkout. The path must stay "
            "inside the checkout. Returns the file contents as text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the local checkout, as returned by "
                        "clone_repo."
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
            "Search a local checkout for a literal (non-regex) string. Returns "
            "a JSON array of objects with file, line_number, and line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the local checkout, as returned by "
                        "clone_repo."
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
]


class InvestigatorAgent(BaseAgent):
    """Investigate a GitHub issue using code-search tools; do not propose a fix.

    ``github_token`` is injected into ``clone_repo`` locally so the token is
    never part of the Anthropic tool schema or model-visible arguments.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        settings: Settings,
        code_search: CodeSearchTool,
        github_token: str,
        *,
        linked_config_path: str = "repos.json",
    ) -> None:
        """Create an investigator bound to a search tool and clone credentials.

        Args:
            client: Authenticated Anthropic SDK client.
            model: Model identifier passed to the centralized Claude client.
            settings: Application settings with per-agent Claude defaults.
            code_search: Context-gathering tool used to clone and inspect repos.
            github_token: Token with read access for ``clone_repo``. Not sent
                to Claude.
            linked_config_path: Optional ``repos.json`` path. A missing file
                means no linked repos (cross-repo search is optional).
        """
        super().__init__(client, model, settings, "investigator")
        self._code_search = code_search
        self._github_token = github_token.strip()
        if not self._github_token:
            raise AgentError("github_token must not be empty")
        self._linked_config_path = linked_config_path
        self.system_prompt = INVESTIGATOR_SYSTEM_PROMPT
        self.tool_definitions = list(INVESTIGATOR_TOOL_DEFINITIONS)

    def investigate(
        self,
        issue_title: str,
        issue_body: str,
        primary_repo: str,
    ) -> Investigation:
        """Investigate an issue and return a validated :class:`Investigation`.

        Args:
            issue_title: GitHub issue title.
            issue_body: GitHub issue body (may be empty).
            primary_repo: GitHub slug of the repository that owns the issue
                (from ``Settings.github_repo``, not from ``repos.json``).

        Returns:
            Validated investigation findings.

        Raises:
            AgentError: If Claude fails or the structured output is invalid.
        """
        user_message = self._build_user_message(
            issue_title=issue_title,
            issue_body=issue_body,
            primary_repo=primary_repo,
        )
        return self.run(user_message, Investigation)

    def _build_user_message(
        self,
        *,
        issue_title: str,
        issue_body: str,
        primary_repo: str,
    ) -> str:
        """Assemble the user turn from issue text, primary repo, and linked repos."""
        config = self._code_search.load_linked_repos(self._linked_config_path)
        if config.linked:
            linked_lines = "\n".join(
                f"- {item.name}: {item.repo}" for item in config.linked
            )
        else:
            linked_lines = "None configured."

        body = issue_body.strip() if issue_body else "(empty)"
        return (
            "Investigate this GitHub issue.\n\n"
            f"Primary repository: {primary_repo}\n\n"
            f"Title: {issue_title}\n\n"
            f"Body:\n{body}\n\n"
            "Linked repositories (optional extra context; clone only if the "
            "evidence suggests the cause may live there):\n"
            f"{linked_lines}\n\n"
            "Start by cloning the primary repository. Use the tools to gather "
            "evidence. Do not propose a fix. Respond with only the JSON object "
            "specified in your instructions."
        )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Route a Claude tool call to :class:`CodeSearchTool` and return text."""
        try:
            if tool_name == "clone_repo":
                local_path = self._code_search.clone_repo(
                    _require_str(tool_input, "repo_full_name"),
                    self._github_token,
                )
                return json.dumps({"local_path": local_path}, ensure_ascii=False)

            if tool_name == "list_files":
                extensions = _optional_str_list(tool_input.get("extensions"))
                files = self._code_search.list_files(
                    _require_str(tool_input, "local_repo_path"),
                    extensions,
                )
                return json.dumps(files, ensure_ascii=False)

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
        except ToolError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        return json.dumps(
            {"error": f"Unknown tool: {tool_name!r}"},
            ensure_ascii=False,
        )


def _require_str(tool_input: dict[str, Any], key: str) -> str:
    """Return a non-empty string argument from a tool-input dict.

    Raises:
        ToolError: If ``key`` is missing, not a string, or blank.
    """
    value = tool_input.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Tool argument {key!r} must be a non-empty string")
    return value.strip()


def _optional_str_list(value: Any) -> list[str] | None:
    """Normalize an optional list of extension strings.

    Returns:
        ``None`` when the argument is omitted, otherwise a list of strings.

    Raises:
        ToolError: If the value is present but not a list of strings.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ToolError("Tool argument 'extensions' must be a list of strings when set")
    return value
