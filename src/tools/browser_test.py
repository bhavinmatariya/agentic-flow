"""Run one-off Playwright verification scripts in an isolated subprocess."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

from core.exceptions import ToolError
from utils.logger import get_logger

_Browser_LAUNCH_MARKERS: tuple[str, ...] = (
    "browserType.launch",
    "Executable doesn't exist",
    "Failed to launch",
    "playwright install",
    "No module named 'playwright'",
    "cannot find Chrome",
    "cannot find Chromium",
)

SKIPPED_AUTH_UI_VERIFICATION: Final[dict[str, Any]] = {
    "ui_passed": None,
    "skipped": True,
    "details": "skipped: could not establish authenticated session",
}

_EXAMPLE_SCRIPT = '''\
"""Example Playwright script for :meth:`BrowserTestTool.run_playwright_check`.

The reviewer agent should emit a complete script like this. The script must
print a single JSON object to stdout with at least ``passed``, ``details``, and
``test_marker`` keys. When TEST_USER_EMAIL/TEST_USER_PASSWORD env vars are set,
log in before exercising the changed page.
"""

import json
import os
import sys
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:3000")
TEST_USER_EMAIL = os.environ.get("TEST_USER_EMAIL", "")
TEST_USER_PASSWORD = os.environ.get("TEST_USER_PASSWORD", "")
TEST_MARKER = "AGENT_TEST_unique-value-here"


def _maybe_login(page) -> None:
    if not TEST_USER_EMAIL or not TEST_USER_PASSWORD:
        return
    for email_selector in ('input[name="email"]', 'input[type="email"]', "#email"):
        if page.locator(email_selector).count():
            page.fill(email_selector, TEST_USER_EMAIL)
            break
    for password_selector in (
        'input[name="password"]',
        'input[type="password"]',
        "#password",
    ):
        if page.locator(password_selector).count():
            page.fill(password_selector, TEST_USER_PASSWORD)
            break
    for submit_selector in (
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Log in")',
        'button:has-text("Sign in")',
    ):
        if page.locator(submit_selector).count():
            page.click(submit_selector)
            page.wait_for_timeout(1500)
            break


def main() -> None:
    details = []
    passed = False
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL, wait_until="networkidle")
            _maybe_login(page)
            page.fill('input[name="email"]', f"{TEST_MARKER}@example.com")
            page.click('button[type="submit"]')
            page.wait_for_timeout(1000)
            passed = TEST_MARKER in page.content()
            details.append("Logged in when credentials provided, then verified marker")
            browser.close()
    except Exception as exc:
        details.append(str(exc))
    print(json.dumps({"passed": passed, "details": "; ".join(details), "test_marker": TEST_MARKER}))


if __name__ == "__main__":
    main()
'''


class BrowserLaunchError(ToolError):
    """Playwright could not launch a browser — not an assertion failure."""


class ScriptExecutionError(ToolError):
    """Generated verification script crashed unexpectedly before a clean result."""


def playwright_script_guidance(
    *,
    base_url: str,
    test_email: str | None = None,
    test_password: str | None = None,
) -> str:
    """Return instructions for generating an ephemeral Playwright verification script."""
    lines = [
        "Generate a one-off Playwright Python script for live UI verification.",
        f"Base URL: {base_url}",
        "The script must print ONE JSON object to stdout with keys: "
        "passed (bool), details (str), test_marker (str).",
        "Use a unique TEST_MARKER prefixed with AGENT_TEST_.",
    ]
    if test_email and test_password:
        lines.extend(
            [
                "",
                "Authenticated session — REQUIRED:",
                f"- TEST_USER_EMAIL is available via os.environ (value: {test_email!r}).",
                f"- TEST_USER_PASSWORD is available via os.environ (do not hardcode secrets).",
                "- If the page under test requires authentication, log in FIRST using "
                "these test credentials, then navigate to the actual changed "
                "page/behavior under test.",
                "- Do NOT stop at the login screen or treat an unauthenticated redirect "
                "as a successful verification of the changed feature.",
                "- After login, exercise the specific user flow from the issue/fix and "
                "include the TEST_MARKER in the data you submit or verify.",
            ]
        )
    else:
        lines.append(
            "No test credentials were seeded; only use public/unauthenticated flows."
        )
    lines.append("")
    lines.append("Reference pattern:")
    lines.append(BrowserTestTool.example_script())
    return "\n".join(lines)


class BrowserTestTool:
    """Execute dynamically generated Playwright scripts against a running app."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Create the tool with optional logging."""
        self._logger = logger or get_logger(__name__)

    def run_playwright_check(
        self,
        script_code: str,
        base_url: str,
        *,
        test_email: str | None = None,
        test_password: str | None = None,
    ) -> dict[str, Any]:
        """Run ``script_code`` and parse the JSON object it prints to stdout.

        Prerequisites:
            ``pip install playwright`` and ``playwright install chromium``

        Args:
            script_code: Complete Python Playwright script source code.
            base_url: Base URL of the running frontend under test.
            test_email: Optional disposable test user email for authenticated flows.
            test_password: Optional disposable test user password.

        Returns:
            Parsed JSON result dict from the script stdout. Expected keys include
            ``passed``, ``details``, ``test_marker``, and ``check_type`` (``browser``).

        Raises:
            BrowserLaunchError: If the browser cannot launch.
            ScriptExecutionError: If the script crashes unexpectedly.
            ToolError: If stdout does not contain valid JSON after a clean run.
        """
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="agentic-flow-playwright-",
                delete=False,
                encoding="utf-8",
            ) as handle:
                handle.write(script_code)
                temp_path = Path(handle.name)

            env = os.environ.copy()
            env["BASE_URL"] = base_url
            if test_email:
                env["TEST_USER_EMAIL"] = test_email
            if test_password:
                env["TEST_USER_PASSWORD"] = test_password
            completed = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
                check=False,
            )
            combined_output = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )

            if completed.returncode != 0:
                if _is_browser_launch_failure(combined_output):
                    raise BrowserLaunchError(
                        "Playwright browser could not launch: "
                        f"{completed.stderr or completed.stdout}"
                    )
                parsed = _try_parse_json_result(completed.stdout)
                if parsed is not None:
                    parsed.setdefault("check_type", "browser")
                    return parsed
                raise ScriptExecutionError(
                    "Playwright script crashed unexpectedly "
                    f"(exit={completed.returncode}): {completed.stderr or completed.stdout}"
                )

            result = self._parse_json_result(completed.stdout)
            result.setdefault("check_type", "browser")
            self._logger.info(
                "Playwright check finished: passed=%s marker=%r",
                result.get("passed"),
                result.get("test_marker"),
            )
            return result
        except BrowserLaunchError:
            raise
        except ScriptExecutionError:
            raise
        except ToolError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise ScriptExecutionError(
                f"Playwright script timed out after 180s: {exc}"
            ) from exc
        except Exception as exc:
            if _is_browser_launch_failure(str(exc)):
                raise BrowserLaunchError(f"Playwright browser could not launch: {exc}") from exc
            raise ToolError(f"run_playwright_check failed: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def run_http_check(
        self,
        url: str,
        *,
        method: str = "GET",
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        expected_status: int = 200,
        marker_value: str | None = None,
    ) -> dict[str, Any]:
        """Call a backend/frontend HTTP endpoint when a real browser is unavailable.

        Returns an API-level verification result with ``check_type`` set to ``api``.
        """
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        payload_bytes: bytes | None = None
        if json_body is not None:
            payload_bytes = json.dumps(json_body).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload_bytes,
            headers=request_headers,
            method=method.upper(),
        )
        details: list[str] = [
            f"{method.upper()} {url} (API-level check; browser unavailable)",
        ]
        passed = False
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
                status_ok = response.status == expected_status
                marker_ok = True
                if marker_value is not None:
                    marker_ok = marker_value in body
                passed = status_ok and marker_ok
                details.append(f"status={response.status}")
                if marker_value is not None:
                    details.append(
                        f"marker {'found' if marker_ok else 'missing'} in response body"
                    )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            marker_ok = marker_value is None or marker_value in body
            passed = exc.code == expected_status and marker_ok
            details.append(f"HTTPError status={exc.code}")
        except Exception as exc:
            details.append(str(exc))

        result = {
            "passed": passed,
            "details": "; ".join(details),
            "test_marker": marker_value,
            "check_type": "api",
            "fallback_reason": "browser_unavailable",
        }
        self._logger.info(
            "HTTP fallback check finished: passed=%s url=%r",
            passed,
            url,
        )
        return result

    @staticmethod
    def example_script() -> str:
        """Return an example Playwright script showing the required stdout JSON pattern."""
        return _EXAMPLE_SCRIPT

    @staticmethod
    def _parse_json_result(stdout: str) -> dict[str, Any]:
        payload = _try_parse_json_result(stdout)
        if payload is None:
            raise ToolError(
                "Playwright script did not print a JSON object to stdout. "
                f"Output was: {stdout[:500]!r}"
            )
        return payload


def _try_parse_json_result(stdout: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _is_browser_launch_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker.lower() in lowered for marker in _BROWSER_LAUNCH_MARKERS)


def extract_test_marker(script_code: str) -> str | None:
    """Best-effort extraction of TEST_MARKER from generated Playwright script text."""
    match = re.search(
        r"""TEST_MARKER\s*=\s*(['"])(.+?)\1""",
        script_code,
        flags=re.DOTALL,
    )
    if match:
        return match.group(2)
    return None
