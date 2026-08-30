"""Reviewer agent: self-review changes and optional full-stack live verification."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.exceptions import AgentError, EnvironmentSetupError, ToolError
from core.models import ImplementationResult, Investigation, ReviewResult
from tools.browser_test import (
    BrowserLaunchError,
    BrowserTestTool,
    ScriptExecutionError,
    extract_test_marker,
)
from tools.code_search import CodeSearchTool
from tools.db_verifier import DBVerifierTool
from tools.environment_manager import EnvironmentManager, checkout_git_branch
from tools.test_runner import AutomatedCheckResult, run_automated_checks

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
_LIVE_VERIFICATION_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "detect_stack",
        "start_test_database",
        "run_migrations",
        "generate_dummy_env",
        "start_process",
        "stop_process",
        "stop_test_database",
        "run_playwright_check",
        "query",
        "delete_by_marker",
    }
)
_LIVE_VERIFICATION_BUDGET_SECONDS: Final[int] = 20 * 60

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
    "results (ui_passed, db_passed), not one combined guess. A successful "
    "query that returns zero rows or wrong field values is a REAL code "
    "verification failure. Only connection/query infrastructure errors are "
    "environment problems. Always call delete_by_marker for cleanup and "
    "stop_process for both processes and the test database, even if the check "
    "failed. Never run this against anything other than the disposable test "
    "database you just created.\n\n"
    "When live verification does not apply, set ui_verification and "
    "db_verification to null.\n\n"
    "Extract every concrete literal detail mentioned in the human's request or "
    "approved approach (exact colors, copy/text, specific values, named "
    "behaviors). Check the actual diff against each one individually. If any "
    "literal detail was not applied, set approved=false and name the missing "
    "detail explicitly in findings — do not approve just because the code runs "
    "or looks reasonable.\n\n"
    "If a competing-implementation conflict is found (duplicate CSS rules, "
    "overlapping state, two code paths for the same behavior), require the "
    "next round's fix to remove the conflict, not add a third implementation "
    "on top. Report this clearly in findings.\n\n"
    "Set making_progress=false only when you believe another implement/review "
    "round cannot help (blocked requirement, fundamental mismatch, or the "
    "same failure would repeat with no new lever). Otherwise keep "
    "making_progress=true so the agent can self-heal.\n\n"
    "Respond with ONLY JSON: {\"approved\": bool, \"summary\": str, "
    "\"findings\": [str], \"making_progress\": bool, \"layers_detected\": "
    "{\"frontend\": bool, "
    "\"database\": bool, \"backend\": bool}, \"ui_verification\": object|null, "
    "\"db_verification\": object|null}. Live verification dicts should "
    "include ui_passed/db_passed booleans plus details."
)


class LiveVerificationAbort(Exception):
    """Live verification could not produce a trustworthy result."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _LiveVerificationState:
    """Runtime tracking for disposable live verification and cleanup."""

    deadline: float
    marker_value: str | None = None
    marker_table: str | None = None
    marker_column: str | None = None
    db_type: str | None = None
    test_row_created: bool = False
    script_retry_used: bool = False


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
        self._live_state: _LiveVerificationState | None = None
        self._review_context: dict[str, Any] | None = None
        self.system_prompt = REVIEWER_SYSTEM_PROMPT
        self.tool_definitions = _build_tool_definitions(include_live=True)

    def review(
        self,
        issue: dict[str, Any],
        investigation: Investigation,
        implementation: ImplementationResult,
        primary_repo: str,
        *,
        human_approval_text: str | None = None,
    ) -> ReviewResult:
        """Review ``implementation`` on its branch and return a :class:`ReviewResult`."""
        local_repo_path = self._code_search.clone_repo(primary_repo, self._github_token)
        checkout_git_branch(
            local_repo_path,
            implementation.branch_name,
            primary_repo,
            self._github_token,
            logger=self._logger,
        )
        layers = detect_change_layers(implementation.files_changed)
        automated_checks = run_automated_checks(local_repo_path)
        default_effort = self._settings.agent_config("reviewer").effort
        self._review_context = {
            "issue": issue,
            "investigation": investigation,
            "implementation": implementation,
            "primary_repo": primary_repo,
            "local_repo_path": local_repo_path,
            "layers": layers,
            "human_approval_text": human_approval_text,
            "automated_checks": automated_checks,
        }

        if layers.get("frontend") and layers.get("database"):
            self._effort = self._settings.reviewer_live_effort
            self._live_state = _LiveVerificationState(
                deadline=time.monotonic() + _LIVE_VERIFICATION_BUDGET_SECONDS,
            )

        user_message = self._build_user_message(
            issue=issue,
            investigation=investigation,
            implementation=implementation,
            primary_repo=primary_repo,
            local_repo_path=local_repo_path,
            layers=layers,
            human_approval_text=human_approval_text,
            automated_checks=automated_checks,
        )
        try:
            review = self.run(user_message, ReviewResult)
            return _merge_automated_checks(review, automated_checks)
        except EnvironmentSetupError as exc:
            self._logger.warning(
                "Live verification environment setup failed: %s",
                exc,
            )
            return self._review_without_live_verification(str(exc))
        except LiveVerificationAbort as exc:
            self._logger.warning(
                "Live verification inconclusive; falling back to standard review: %s",
                exc.reason,
            )
            return self._review_without_live_verification(exc.reason)
        finally:
            self._effort = default_effort
            self._safe_cleanup_live_verification()
            self._live_state = None
            self._review_context = None
            self.tool_definitions = _build_tool_definitions(include_live=True)

    def _run_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Execute tools, propagating environment aborts to the review fallback."""
        results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = str(block.name)
            raw_input = block.input if isinstance(block.input, dict) else {}
            tool_input: dict[str, Any] = dict(raw_input)
            self._logger.info("Tool call: %s input=%s", name, tool_input)
            try:
                content = self._execute_tool(name, tool_input)
            except (EnvironmentSetupError, LiveVerificationAbort):
                raise
            except Exception as exc:
                self._logger.error(
                    "Tool %s raised %s: %s", name, type(exc).__name__, exc
                )
                content = json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                }
            )
        if not results:
            raise AgentError(
                "stop_reason was tool_use but the response contained no tool_use blocks."
            )
        return results

    def _build_user_message(
        self,
        *,
        issue: dict[str, Any],
        investigation: Investigation,
        implementation: ImplementationResult,
        primary_repo: str,
        local_repo_path: str,
        layers: dict[str, bool],
        human_approval_text: str | None = None,
        live_verification_note: str | None = None,
        automated_checks: AutomatedCheckResult | None = None,
    ) -> str:
        issue_body = str(issue.get("body") or "").strip() or "(empty)"
        files_block = "\n".join(f"- {path}" for path in implementation.files_changed) or "(none)"
        approval_block = (
            str(human_approval_text or "").strip()
            or "(no human approval comment provided)"
        )
        message = (
            "Review this implemented fix.\n\n"
            f"Primary repository: {primary_repo}\n"
            f"Branch: {implementation.branch_name}\n"
            f"Local checkout path: {local_repo_path}\n\n"
            f"Issue #{issue['number']}: {issue['title']}\n\n"
            f"Issue body:\n{issue_body}\n\n"
            f"Human approval / feedback (verify every literal detail from this text):\n"
            f"{approval_block}\n\n"
            f"Root cause:\n{investigation.root_cause}\n\n"
            f"Implementation summary:\n{implementation.summary}\n\n"
            f"Files changed:\n{files_block}\n\n"
            f"Precomputed layers_detected: {json.dumps(layers)}\n\n"
            "Start by calling detect_change_layers with files_changed. "
            "Run live verification only when both frontend and database are true."
        )
        if automated_checks is not None:
            if automated_checks.findings:
                message += (
                    "\n\nAutomated test/build failures already observed "
                    "(include these in findings if still applicable):\n"
                    f"{json.dumps(automated_checks.findings, ensure_ascii=False, indent=2)}"
                )
            if automated_checks.skipped:
                skipped_block = "\n".join(
                    f"- {item}" for item in automated_checks.skipped
                )
                message += (
                    "\n\nAutomated checks skipped (informational — not failures):\n"
                    f"{skipped_block}"
                )
        if live_verification_note:
            message += (
                "\n\nIMPORTANT: Live verification could not be completed.\n"
                f"Reason: {live_verification_note}\n"
                "Do NOT call live verification tools. Finish the review using only "
                "read_file, search_code, run_command, and detect_change_layers. "
                "Set ui_verification and db_verification to null and explain in "
                "summary that live verification was skipped for the reason above."
            )
        return message

    def _review_without_live_verification(self, reason: str) -> ReviewResult:
        """Re-run review using standard checks only after live verification aborts."""
        context = self._review_context
        if context is None:
            raise AgentError(
                "Cannot fall back from live verification without review context."
            )

        self._live_state = None
        self.tool_definitions = _build_tool_definitions(include_live=False)
        fallback_message = self._build_user_message(
            issue=context["issue"],
            investigation=context["investigation"],
            implementation=context["implementation"],
            primary_repo=context["primary_repo"],
            local_repo_path=context["local_repo_path"],
            layers=context["layers"],
            human_approval_text=context.get("human_approval_text"),
            live_verification_note=reason,
            automated_checks=context.get("automated_checks"),
        )
        review = self.run(fallback_message, ReviewResult)
        automated = context.get("automated_checks")
        if isinstance(automated, AutomatedCheckResult):
            return _merge_automated_checks(review, automated)
        return review

    def _check_live_verification_budget(self) -> None:
        if self._live_state is None:
            return
        if time.monotonic() > self._live_state.deadline:
            raise EnvironmentSetupError(
                "Live verification exceeded overall time budget of "
                f"{_LIVE_VERIFICATION_BUDGET_SECONDS}s"
            )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name in _LIVE_VERIFICATION_TOOLS:
            self._check_live_verification_budget()

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
                if self._live_state is not None:
                    self._live_state.db_type = db_type
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
                    int(tool_input.get("timeout", 90)),
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
                result = self._run_playwright_resilient(
                    script_code=_require_str(tool_input, "script_code"),
                    base_url=_require_str(tool_input, "base_url"),
                    api_fallback_url=_optional_str(tool_input, "api_fallback_url"),
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
                if self._live_state is not None:
                    self._live_state.marker_table = _require_str(tool_input, "table")
                    self._live_state.marker_column = _require_str(
                        tool_input, "marker_column"
                    )
                    self._live_state.marker_value = _require_str(
                        tool_input, "marker_value"
                    )
                    self._live_state.db_type = _require_str(tool_input, "db_type")
                    if rows:
                        self._live_state.test_row_created = True
                return json.dumps(
                    {"rows": rows, "row_count": len(rows), "infra_ok": True},
                    ensure_ascii=False,
                    default=str,
                )

            if tool_name == "delete_by_marker":
                deleted = self._db_verifier.delete_by_marker(
                    _require_str(tool_input, "connection_string"),
                    _require_str(tool_input, "db_type"),
                    _require_str(tool_input, "table"),
                    _require_str(tool_input, "marker_column"),
                    _require_str(tool_input, "marker_value"),
                )
                if self._live_state is not None:
                    self._live_state.test_row_created = False
                return json.dumps({"deleted_rows": deleted}, ensure_ascii=False)
        except EnvironmentSetupError:
            raise
        except ToolError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {tool_name!r}"}, ensure_ascii=False)

    def _run_playwright_resilient(
        self,
        *,
        script_code: str,
        base_url: str,
        api_fallback_url: str | None,
    ) -> dict[str, Any]:
        marker = extract_test_marker(script_code)
        try:
            result = self._browser_test.run_playwright_check(script_code, base_url)
            if marker and self._live_state is not None:
                self._live_state.marker_value = marker
            return result
        except BrowserLaunchError as exc:
            self._logger.warning(
                "Playwright browser unavailable; falling back to HTTP check: %s",
                exc,
            )
            fallback_url = api_fallback_url or base_url
            return self._browser_test.run_http_check(
                fallback_url,
                marker_value=marker,
            )
        except ScriptExecutionError as exc:
            if self._live_state is not None and not self._live_state.script_retry_used:
                self._live_state.script_retry_used = True
                return {
                    "retry_script": True,
                    "script_error": str(exc),
                    "feedback": (
                        "The generated verification script crashed unexpectedly. "
                        "Fix the Python/Playwright error and call run_playwright_check "
                        "once more with a corrected script."
                    ),
                }
            raise LiveVerificationAbort(
                "Live verification inconclusive due to test tooling, not the "
                f"implementation: {exc}"
            ) from exc

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

    def _safe_cleanup_live_verification(self) -> None:
        """Best-effort cleanup for processes, DB container, marker rows, and scripts."""
        for pid in list(self._process_handles):
            try:
                handle = self._process_handles.pop(pid, None)
                self._environment.stop_process(handle)
            except Exception as exc:
                self._logger.warning("Cleanup failed to stop process pid=%s: %s", pid, exc)

        try:
            self._environment.stop_test_database()
        except Exception as exc:
            self._logger.warning("Cleanup failed to stop test database: %s", exc)

        state = self._live_state
        connection_string = self._active_connection_string
        if (
            state is not None
            and state.test_row_created
            and connection_string
            and state.marker_table
            and state.marker_column
            and state.marker_value
            and state.db_type
        ):
            try:
                self._db_verifier.delete_by_marker(
                    connection_string,
                    state.db_type,
                    state.marker_table,
                    state.marker_column,
                    state.marker_value,
                )
            except Exception as exc:
                self._logger.warning("Cleanup failed to delete test marker row: %s", exc)

        self._active_connection_string = None


def _merge_automated_checks(
    review: ReviewResult,
    automated: AutomatedCheckResult,
) -> ReviewResult:
    """Fold deterministic test/build failures into the model's review result."""
    findings = list(review.findings)
    for item in automated.findings:
        if item not in findings:
            findings.append(item)

    approved = review.approved and not automated.findings
    summary = review.summary
    if automated.skipped:
        skipped_note = "; ".join(automated.skipped)
        summary = f"{summary.rstrip()} ({skipped_note})"

    return review.model_copy(
        update={
            "approved": approved,
            "findings": findings,
            "summary": summary,
        }
    )


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


def _build_tool_definitions(*, include_live: bool = True) -> list[dict[str, Any]]:
    local_repo_path_schema = {
        "type": "string",
        "description": "Absolute path to the local checkout provided in your instructions.",
    }
    tools: list[dict[str, Any]] = [
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
    ]

    if not include_live:
        return tools

    tools.extend(
        [
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
                        "api_fallback_url": {
                            "type": "string",
                            "description": "Optional backend API URL used if the browser cannot launch.",
                        },
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
    )
    return tools


def _require_str(tool_input: dict[str, Any], key: str) -> str:
    value = tool_input.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"Tool argument {key!r} must be a non-empty string")
    return value.strip()


def _optional_str(tool_input: dict[str, Any], key: str) -> str | None:
    value = tool_input.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()
