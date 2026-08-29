"""Run one-off Playwright verification scripts in an isolated subprocess."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.exceptions import ToolError
from utils.logger import get_logger

_EXAMPLE_SCRIPT = '''\
"""Example Playwright script for :meth:`BrowserTestTool.run_playwright_check`.

The reviewer agent should emit a complete script like this. The script must
print a single JSON object to stdout with at least ``passed``, ``details``, and
``test_marker`` keys.
"""

import json
import os
import sys
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:3000")
TEST_MARKER = "AGENT_TEST_unique-value-here"


def main() -> None:
    details = []
    passed = False
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL, wait_until="networkidle")
            page.fill('input[name="email"]', f"{TEST_MARKER}@example.com")
            page.click('button[type="submit"]')
            page.wait_for_timeout(1000)
            passed = TEST_MARKER in page.content()
            details.append("Submitted form and searched page content for marker")
            browser.close()
    except Exception as exc:
        details.append(str(exc))
    print(json.dumps({"passed": passed, "details": "; ".join(details), "test_marker": TEST_MARKER}))


if __name__ == "__main__":
    main()
'''


class BrowserTestTool:
    """Execute dynamically generated Playwright scripts against a running app."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Create the tool with optional logging."""
        self._logger = logger or get_logger(__name__)

    def run_playwright_check(self, script_code: str, base_url: str) -> dict[str, Any]:
        """Run ``script_code`` and parse the JSON object it prints to stdout.

        Prerequisites:
            ``pip install playwright`` and ``playwright install chromium``

        Args:
            script_code: Complete Python Playwright script source code.
            base_url: Base URL of the running frontend under test.

        Returns:
            Parsed JSON result dict from the script stdout. Expected keys include
            ``passed``, ``details``, and ``test_marker``.

        Raises:
            ToolError: If the subprocess fails or stdout does not contain valid JSON.
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
            completed = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise ToolError(
                    "Playwright script failed "
                    f"(exit={completed.returncode}): {completed.stderr or completed.stdout}"
                )

            result = self._parse_json_result(completed.stdout)
            self._logger.info(
                "Playwright check finished: passed=%s marker=%r",
                result.get("passed"),
                result.get("test_marker"),
            )
            return result
        except ToolError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"Playwright script timed out after 180s: {exc}"
            ) from exc
        except Exception as exc:
            raise ToolError(f"run_playwright_check failed: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def example_script() -> str:
        """Return an example Playwright script showing the required stdout JSON pattern."""
        return _EXAMPLE_SCRIPT

    @staticmethod
    def _parse_json_result(stdout: str) -> dict[str, Any]:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ToolError(
            "Playwright script did not print a JSON object to stdout. "
            f"Output was: {stdout[:500]!r}"
        )
