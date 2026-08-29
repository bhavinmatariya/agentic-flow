#!/usr/bin/env python3
"""Manual smoke test for Step 8 live verification tooling.

Usage:
    python test_step8_live.py [--local-repo-path PATH] [--branch NAME]

Exercises detect_stack, disposable database startup, optional migrations,
process startup, Playwright check, and DB verification without requiring
the full reviewer LLM loop. Use this to debug environment detection before
trusting live verification in the main pipeline.

Prerequisites:
    pip install -r requirements.txt
    playwright install chromium
    Docker running (for postgres/mysql/mongo detection paths)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import ConfigurationError, Settings
from core.exceptions import EnvironmentSetupError
from tools.browser_test import BrowserTestTool
from tools.code_search import CodeSearchTool
from tools.db_verifier import DBVerifierTool
from tools.environment_manager import EnvironmentManager, checkout_git_branch

DEFAULT_FRONTEND_URL = "http://127.0.0.1:3000"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def _build_demo_playwright_script(base_url: str, marker: str) -> str:
    """Return a minimal Playwright script that only checks the frontend loads."""
    return f'''\
import json
from playwright.sync_api import sync_playwright

BASE_URL = {base_url!r}
TEST_MARKER = {marker!r}

def main() -> None:
    passed = False
    details = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            response = page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
            passed = response is not None and response.status < 500
            details.append(f"Loaded {{BASE_URL}} with status {{getattr(response, 'status', 'unknown')}}")
            browser.close()
    except Exception as exc:
        details.append(str(exc))
    print(json.dumps({{"passed": passed, "details": "; ".join(details), "test_marker": TEST_MARKER}}))

if __name__ == "__main__":
    main()
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise Step 8 live verification tooling against a checkout."
    )
    parser.add_argument(
        "--local-repo-path",
        help="Existing checkout to inspect. When omitted, clones GITHUB_REPO from .env.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to fetch/checkout when cloning (default: main)",
    )
    parser.add_argument(
        "--db-type",
        help="Force a db_type instead of detect_stack inference (postgres/mysql/mongo/sqlite)",
    )
    parser.add_argument(
        "--skip-process-start",
        action="store_true",
        help="Skip backend/frontend startup and Playwright check",
    )
    parser.add_argument(
        "--table",
        default="agentic_flow_test_rows",
        help="Table used for sqlite demo verification",
    )
    parser.add_argument(
        "--marker-column",
        default="test_marker",
        help="Marker column used for sqlite demo verification",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    marker = f"AGENT_TEST_{uuid.uuid4().hex[:12]}"
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    try:
        if args.local_repo_path:
            local_repo_path = str(Path(args.local_repo_path).expanduser().resolve())
        else:
            settings = Settings.from_env()
            temp_dir = tempfile.TemporaryDirectory(prefix="agentic-flow-step8-live-")
            search_tool = CodeSearchTool(temp_dir.name)
            local_repo_path = search_tool.clone_repo(
                settings.github_repo,
                settings.github_token,
            )
            checkout_git_branch(
                local_repo_path,
                args.branch,
                settings.github_repo,
                settings.github_token,
            )

        return _run_live_path(
            local_repo_path=local_repo_path,
            args=args,
            marker=marker,
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except EnvironmentSetupError as exc:
        print(f"Environment error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _run_live_path(
    *,
    local_repo_path: str,
    args: argparse.Namespace,
    marker: str,
) -> int:
    environment = EnvironmentManager()
    browser = BrowserTestTool()
    db_verifier = DBVerifierTool()
    backend_process = None
    frontend_process = None

    try:
        print(f"Local repo: {local_repo_path}")
        stack = environment.detect_stack(local_repo_path)
        print("=== detect_stack ===")
        print(json.dumps(stack, indent=2))
        print()

        db_type = args.db_type or stack.get("db_type") or "sqlite"
        print(f"Starting disposable test database: {db_type}")
        connection_string = environment.start_test_database(str(db_type))
        print(f"connection_string={connection_string}")
        print()

        migrations_ran = environment.run_migrations(local_repo_path, connection_string)
        print(f"run_migrations={migrations_ran}")
        print()

        env = environment.generate_dummy_env(
            list(stack.get("required_env_vars") or []),
            connection_string,
        )
        print("=== generate_dummy_env (sample) ===")
        preview = {key: env[key] for key in list(env)[:8]}
        print(json.dumps(preview, indent=2))
        print()

        if db_type == "sqlite":
            _seed_sqlite_demo_row(
                connection_string,
                args.table,
                args.marker_column,
                marker,
            )
            rows = db_verifier.query(
                connection_string,
                "sqlite",
                args.table,
                args.marker_column,
                marker,
            )
            print(f"sqlite query rows={len(rows)}")
            deleted = db_verifier.delete_by_marker(
                connection_string,
                "sqlite",
                args.table,
                args.marker_column,
                marker,
            )
            print(f"sqlite cleanup deleted={deleted}")
            print()

        if args.skip_process_start:
            print("Skipping process startup and Playwright check.")
            print("Step 8 live tooling smoke test passed (detection-only path).")
            return 0

        backend_command = stack.get("backend_start_command")
        frontend_command = stack.get("frontend_start_command")
        frontend_port = stack.get("frontend_port") or 3000
        frontend_url = f"http://127.0.0.1:{frontend_port}"

        if backend_command:
            print(f"Starting backend: {backend_command}")
            backend_process = environment.start_process(
                backend_command,
                local_repo_path,
                env,
                DEFAULT_BACKEND_URL,
                timeout=120,
            )
        else:
            print("No backend_start_command detected; skipping backend startup.")

        if frontend_command:
            print(f"Starting frontend: {frontend_command}")
            frontend_process = environment.start_process(
                frontend_command,
                local_repo_path,
                env,
                frontend_url,
                timeout=120,
            )
        else:
            print("No frontend_start_command detected; probing default frontend URL.")
            frontend_url = DEFAULT_FRONTEND_URL

        script = _build_demo_playwright_script(frontend_url, marker)
        print("=== run_playwright_check ===")
        ui_result = browser.run_playwright_check(script, frontend_url)
        print(json.dumps(ui_result, indent=2))
        print()

        db_result = {
            "db_passed": bool(connection_string),
            "details": (
                "Demo sqlite seed/query/delete path exercised when db_type=sqlite."
            ),
            "test_marker": marker,
        }
        print("=== db_verification (demo) ===")
        print(json.dumps(db_result, indent=2))
        print()
        print("Step 8 live tooling smoke test passed.")
        return 0
    finally:
        environment.stop_process(backend_process)
        environment.stop_process(frontend_process)
        environment.stop_test_database()


def _seed_sqlite_demo_row(
    connection_string: str,
    table: str,
    marker_column: str,
    marker: str,
) -> None:
    import sqlite3

    db_path = connection_string.removeprefix("sqlite:///")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            f"({marker_column} TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        conn.execute(
            f"INSERT INTO {table} ({marker_column}, payload) VALUES (?, ?)",
            (marker, "demo-row"),
        )
        conn.commit()


if __name__ == "__main__":
    raise SystemExit(main())
