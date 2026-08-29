"""Reviewer agent: self-review changes and optional full-stack live verification."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.exceptions import AgentError, EnvironmentError, ToolError
from core.models import ImplementationResult, Investigation, ReviewResult
from tools.browser_test import BrowserTestTool
from tools.code_search import CodeSearchTool
from tools.db_verifier import DBVerifierTool
from tools.environment_manager import EnvironmentManager
from utils.logger import get_logger

_FRONTEND_MARKERS: Final[tuple[str, ...]] = (
    "frontend/",
    "client/",
    "web/",
    "ui/",
    "src/components/",
    "pages/",
    "app/",
)
_FRONTEND_EXTENSIONS: Final[tuple[str, ...]] = (
    ".tsx",
    ".jsx",
    ".vue",
    ".html",
    ".css",
    ".scss",
)
_DB_MARKERS: Final[tuple[str, ...]] = (
    "migration",
    "migrations/",
    "alembic/",
    "schema.sql",
    "models.py",
    "model.py",
    "database/",
    "db/",
    "prisma/",
    "sequelize/",
    ".sql",
)

REVIEWER_SYSTEM_PROMPT: Final[str] = (
    "You are reviewing an implemented fix before it becomes a pull request. "
    "Compare the change against the original issue, investigation, and "
    "approved approach. Use read_file and search_code to inspect the branch "
    "checkout. Use run_command only for project-native checks such as tests "
    "or linters when those commands exist.\n\n"
    "First call detect_change_layers with the provided files_changed list. "
    "Run standard review checks for every change. Only when BOTH frontend "
    "and database layers are true should you run the full live verification "
    "tier for a user-facing flow that also writes to the database.\n\n"
    "If the change involves a user-facing flow that also writes to the "
    "database (e.g. a form submission), after the standard backend/frontend "
    "checks pass, do a full live verification: call detect_stack, "
    "start_test_database, run_migrations, start the backend and frontend "
    "with dummy env vars via start_process, then write a one-off Playwright "
    "script that exercises the specific flow using a clearly unique test "
    "marker value (e.g. a random string prefixed 'AGENT_TEST_'), run it with "
    "run_playwright_check, then use query() to independently verify the "
    "correct row was stored with correct data — report these as two SEPARATE "
    "results (ui_passed, db_passed), not one combined guess. Always call "
    "delete_by_marker for cleanup and stop_process for both processes and "
    "the test database, even if the check failed. Never run this against "
    "anything other than the disposable test database you just created.\n\n"
    "When live verification does not apply, set ui_verification and "
    "db_verification to null.\n\n"
    "Respond with ONLY JSON: {\"approved\": bool, \"summary\": str, "
    "\"findings\": [str], \"layers_detected\": {\"frontend\": bool, "
    "\"database\": bool, \"backend\": bool}, \"ui_verification\": object|null, "
    "\"db_verification\": object|null}. Live verification dicts should "
    "include ui_passed/db_passed booleans plus details."
)


class ReviewerAgent(BaseAgent):
    """Review an implementation and optionally run disposable full-stack checks."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        settings: Settings,
        code_search: CodeSearchTool,
        environment: EnvironmentManager,
        browser_test: BrowserTestTool,
        db_verifier: DBVerifierTool,
        github_token: str,
    ) -> None:
        """Create a reviewer bound to inspection and live-verification tools."""
        super().__init__(client, model, settings, "reviewer")
        self._code_search = code_search
        self._environment = environment
        self._browser_test = browser_test
        self._db_verifier = db_verifier
        self._github_token = github_token.strip()
        if not self._github_token:
            raise AgentError("github_token must not be empty")
        self._process_handles: dict[int, subprocess.Popen[Any]] = {}
        self._active_connection_string: str | None = None
        self.system_prompt = REVIEWER_SYSTEM_PROMPT
        self.tool_definitions = _build_tool_definitions()

    def review(
        self,
        issue: dict[str, Any],
        investigation: Investigation,
        implementation: ImplementationResult,
        primary_repo: str,
    ) -> ReviewResult:
        """Review ``implementation`` on its branch and return a :class:`ReviewResult`."""
        local_repo_path = self._code_search.clone_repo(primary_repo, self._github_token)
        self._checkout_branch(local_repo_path, implementation.branch_name)
        layers = detect_change_layers(implementation.files_changed)
        default_effort = self._settings.agent_config("reviewer").effort
        if layers.get("frontend") and layers.get("database"):
            self._effort = self._settings.reviewer_live_effort
        user_message = self._build_user_message(
            issue=issue,
            investigation=investigation,
            implementation=implementation,
            primary_repo=primary_repo,
            local_repo_path=local_repo_path,
            layers=layers,
        )
        try:
            return self.run(user_message, ReviewResult)
        finally:
            self._effort = default_effort
            self._cleanup_runtime()

    def _build_user_message(
        self,
        *,
        issue: dict[str, Any],
        investigation: Investigation,
        implementation: ImplementationResult,
        primary_repo: str,
        local_repo_path: str,
        layers: dict[str, bool],
    ) -> str:
        issue_body = str(issue.get("body") or "").strip() or "(empty)"
        files_block = "\n".join(f"- {path}" for path in implementation.files_changed) or "(none)"
        return (
            "Review this implemented fix.\n\n"
            f"Primary repository: {primary_repo}\n"
            f"Branch: {implementation.branch_name}\n"
            f"Local checkout path: {local_repo_path}\n\n"
            f"Issue #{issue['number']}: {issue['title']}\n\n"
            f"Issue body:\n{issue_body}\n\n"
            f"Root cause:\n{investigation.root_cause}\n\n"
            f"Implementation summary:\n{implementation.summary}\n\n"
            f"Files changed:\n{files_block}\n\n"
            f"Precomputed layers_detected: {json.dumps(layers)}\n\n"
            "Start by calling detect_change_layers with files_changed. "
            "Run live verification only when both frontend and database are true."
        )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
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

            if tool_name == "run_command":
                return json.dumps(
                    self._run_command(
                        _require_str(tool_input, "local_repo_path"),
                        _require_str(tool_input, "command"),
                        int(tool_input.get("timeout_seconds", 120)),
                    ),
                    ensure_ascii=False,
                )

            if tool_name == "detect_change_layers":
                files = tool_input.get("files_changed", [])
                if not isinstance(files, list):
                    raise ToolError("files_changed must be a list of strings")
                return json.dumps(
                    detect_change_layers([str(item) for item in files]),
                    ensure_ascii=False,
                )

            if tool_name == "detect_stack":
                return json.dumps(
                    self._environment.detect_stack(
                        _require_str(tool_input, "local_repo_path")
                    ),
                    ensure_ascii=False,
                )

            if tool_name == "start_test_database":
                db_type = _require_str(tool_input, "db_type")
                connection_string = self._environment.start_test_database(db_type)
                self._active_connection_string = connection_string
                return json.dumps(
                    {"connection_string": connection_string},
                    ensure_ascii=False,
                )

            if tool_name == "run_migrations":
                ran = self._environment.run_migrations(
                    _require_str(tool_input, "local_repo_path"),
                    _require_str(tool_input, "connection_string"),
                )
                return json.dumps({"migrations_ran": ran}, ensure_ascii=False)

            if tool_name == "generate_dummy_env":
                env_vars = tool_input.get("detected_env_vars", [])
                if not isinstance(env_vars, list):
                    raise ToolError("detected_env_vars must be a list of strings")
                env = self._environment.generate_dummy_env(
                    [str(item) for item in env_vars],
                    _require_str(tool_input, "db_connection_string"),
                )
                return json.dumps(env, ensure_ascii=False)

            if tool_name == "start_process":
                env_payload = tool_input.get("env", {})
                if not isinstance(env_payload, dict):
                    raise ToolError("env must be an object of string key/value pairs")
                env = {str(key): str(value) for key, value in env_payload.items()}
                process = self._environment.start_process(
                    _require_str(tool_input, "command"),
                    _require_str(tool_input, "cwd"),
                    env,
                    _require_str(tool_input, "ready_url"),
                    int(tool_input.get("timeout", 120)),
                )
                self._process_handles[process.pid] = process
                return json.dumps({"pid": process.pid, "ready": True}, ensure_ascii=False)

            if tool_name == "stop_process":
                pid = int(tool_input.get("pid"))
                handle = self._process_handles.pop(pid, None)
                self._environment.stop_process(handle)
                return json.dumps({"stopped_pid": pid}, ensure_ascii=False)

            if tool_name == "stop_test_database":
                self._environment.stop_test_database()
                self._active_connection_string = None
                return json.dumps({"stopped_test_database": True}, ensure_ascii=False)

            if tool_name == "run_playwright_check":
                result = self._browser_test.run_playwright_check(
                    _require_str(tool_input, "script_code"),
                    _require_str(tool_input, "base_url"),
                )
                return json.dumps(result, ensure_ascii=False)

            if tool_name == "query":
                rows = self._db_verifier.query(
                    _require_str(tool_input, "connection_string"),
                    _require_str(tool_input, "db_type"),
                    _require_str(tool_input, "table"),
                    _require_str(tool_input, "marker_column"),
                    _require_str(tool_input, "marker_value"),
                )
                return json.dumps(rows, ensure_ascii=False, default=str)

            if tool_name == "delete_by_marker":
                deleted = self._db_verifier.delete_by_marker(
                    _require_str(tool_input, "connection_string"),
                    _require_str(tool_input, "db_type"),
                    _require_str(tool_input, "table"),
                    _require_str(tool_input, "marker_column"),
                    _require_str(tool_input, "marker_value"),
                )
                return json.dumps({"deleted_rows": deleted}, ensure_ascii=False)
        except EnvironmentError as exc:
            return json.dumps({"error": f"EnvironmentError: {exc}"}, ensure_ascii=False)
        except ToolError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {tool_name!r}"}, ensure_ascii=False)

    def _run_command(
        self,
        local_repo_path: str,
        command: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            cwd=local_repo_path,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
        }

    def _checkout_branch(self, local_repo_path: str, branch_name: str) -> None:
        for args in (
            ["git", "fetch", "origin", branch_name, "--depth", "1"],
            ["git", "checkout", branch_name],
        ):
            completed = subprocess.run(
                args,
                cwd=local_repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AgentError(
                    f"Could not checkout branch {branch_name!r} in {local_repo_path}: "
                    f"{completed.stderr or completed.stdout}"
                )

    def _cleanup_runtime(self) -> None:
        for pid in list(self._process_handles):
            handle = self._process_handles.pop(pid, None)
            self._environment.stop_process(handle)
        try:
            self._environment.stop_test_database()
        except EnvironmentError as exc:
            self._logger.warning("Cleanup failed to stop test database: %s", exc)
        self._active_connection_string = None


def detect_change_layers(files_changed: list[str]) -> dict[str, bool]:
    """Detect whether a diff touches frontend, database, or backend layers."""
    frontend = False
    database = False
    backend = False
    for raw_path in files_changed:
        path = raw_path.replace("\\", "/").lower()
        if any(marker in path for marker in _FRONTEND_MARKERS) or path.endswith(
            _FRONTEND_EXTENSIONS
        ):
            frontend = True
        if any(marker in path for marker in _DB_MARKERS):
            database = True
        if path.endswith((".py", ".go", ".rs", ".java")) and not database:
            backend = True
        if any(marker in path for marker in ("api/", "server/", "backend/", "routes/")):
            backend = True
    return {"frontend": frontend, "database": database, "backend": backend}


def _build_tool_definitions() -> list[dict[str, Any]]:
    local_repo_path_schema = {
        "type": "string",
        "description": "Absolute path to the local checkout provided in your instructions.",
    }
    return [
        {
            "name": "read_file",
            "description": "Read a UTF-8 file from the local checkout.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "local_repo_path": local_repo_path_schema,
                    "relative_path": {"type": "string"},
                },
                "required": ["local_repo_path", "relative_path"],
            },
        },
        {
            "name": "search_code",
            "description": "Search the local checkout for a literal string.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "local_repo_path": local_repo_path_schema,
                    "query": {"type": "string"},
                },
                "required": ["local_repo_path", "query"],
            },
        },
        {
            "name": "run_command",
            "description": "Run a shell command in the repository root (tests, lint, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "local_repo_path": local_repo_path_schema,
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                },
                "required": ["local_repo_path", "command"],
            },
        },
        {
            "name": "detect_change_layers",
            "description": "Detect frontend/database/backend layers touched by files_changed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "files_changed": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["files_changed"],
            },
        },
        {
            "name": "detect_stack",
            "description": "Detect DB type, start commands, frontend port, and env vars.",
            "input_schema": {
                "type": "object",
                "properties": {"local_repo_path": local_repo_path_schema},
                "required": ["local_repo_path"],
            },
        },
        {
            "name": "start_test_database",
            "description": "Start a disposable Docker or sqlite test database.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "db_type": {
                        "type": "string",
                        "description": "postgres, mysql, mongo, or sqlite",
                    }
                },
                "required": ["db_type"],
            },
        },
        {
            "name": "run_migrations",
            "description": "Run Alembic/Django/schema.sql migrations when detected.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "local_repo_path": local_repo_path_schema,
                    "connection_string": {"type": "string"},
                },
                "required": ["local_repo_path", "connection_string"],
            },
        },
        {
            "name": "generate_dummy_env",
            "description": "Build placeholder env vars for booting the app in tests.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "detected_env_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "db_connection_string": {"type": "string"},
                },
                "required": ["detected_env_vars", "db_connection_string"],
            },
        },
        {
            "name": "start_process",
            "description": "Start backend or frontend and wait for ready_url.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "env": {"type": "object"},
                    "ready_url": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command", "cwd", "env", "ready_url"],
            },
        },
        {
            "name": "stop_process",
            "description": "Stop a process previously returned by start_process.",
            "input_schema": {
                "type": "object",
                "properties": {"pid": {"type": "integer"}},
                "required": ["pid"],
            },
        },
        {
            "name": "stop_test_database",
            "description": "Stop the disposable test database created by start_test_database.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "run_playwright_check",
            "description": "Run a one-off Playwright Python script against base_url.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "script_code": {"type": "string"},
                    "base_url": {"type": "string"},
                },
                "required": ["script_code", "base_url"],
            },
        },
        {
            "name": "query",
            "description": "Read rows matching an exact test marker value.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "connection_string": {"type": "string"},
                    "db_type": {"type": "string"},
                    "table": {"type": "string"},
                    "marker_column": {"type": "string"},
                    "marker_value": {"type": "string"},
                },
                "required": [
                    "connection_string",
                    "db_type",
                    "table",
                    "marker_column",
                    "marker_value",
                ],
            },
        },
        {
            "name": "delete_by_marker",
            "description": "Delete only rows matching the exact test marker value.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "connection_string": {"type": "string"},
                    "db_type": {"type": "string"},
                    "table": {"type": "string"},
                    "marker_column": {"type": "string"},
                    "marker_value": {"type": "string"},
                },
                "required": [
                    "connection_string",
                    "db_type",
                    "table",
                    "marker_column",
                    "marker_value",
                ],
            },
        },
    ]


def _require_str(tool_input: dict[str, Any], key: str) -> str:
    value = tool_input.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Tool argument {key!r} must be a non-empty string")
    return value.strip()
