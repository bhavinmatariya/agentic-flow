#!/usr/bin/env python3
"""Manual smoke test for Step 3: context-gathering code search tools.

Usage:
    python test_step3.py [--query KEYWORD]

Requires a populated .env file (see .env.example). The primary repository is
``GITHUB_REPO`` from Settings. Optional linked repos are loaded from
``repos.json`` when that file exists (see ``repos.json.example``).

The checkout is created in a temporary directory and removed on exit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import ConfigurationError, Settings
from core.exceptions import ToolError
from tools.code_search import CodeSearchTool

DEFAULT_QUERY = "import"
MAX_LISTED_FILES = 30
MAX_SEARCH_HITS = 20


def _pick_file_to_read(files: list[str]) -> str | None:
    """Choose a text-looking file from the listing for the read_file exercise."""
    if not files:
        return None
    preferred_names = ("readme.md", "readme", "readme.txt")
    for path in files:
        if Path(path).name.lower() in preferred_names:
            return path
    return files[0]


def _exercise_repo(
    tool: CodeSearchTool,
    repo: str,
    github_token: str,
    query: str,
    *,
    label: str,
) -> int:
    """Clone ``repo``, list files, read one file, and search ``query``.

    Returns:
        ``0`` on success, ``1`` if the clone has no files to read.
    """
    print(f"=== {label}: {repo} ===")
    print(f"Cloning {repo} ...")
    local_path = tool.clone_repo(repo, github_token)
    print(f"Local path: {local_path}")
    print()

    files = tool.list_files(local_path)
    print(f"File count: {len(files)}")
    preview = files[:MAX_LISTED_FILES]
    for path in preview:
        print(f"  {path}")
    if len(files) > MAX_LISTED_FILES:
        print(f"  ... ({len(files) - MAX_LISTED_FILES} more)")
    print()

    relative = _pick_file_to_read(files)
    if relative is None:
        print(f"ERROR: clone of {repo} contains no files to read", file=sys.stderr)
        return 1

    content = tool.read_file(local_path, relative)
    print(f"Read {relative!r} ({len(content)} characters)")
    snippet = content.splitlines()[:8]
    for line in snippet:
        print(f"  | {line}")
    if len(content.splitlines()) > 8:
        print("  | ...")
    print()

    matches = tool.search_code(local_path, query)
    print(f"Search {query!r}: {len(matches)} match(es)")
    for hit in matches[:MAX_SEARCH_HITS]:
        print(f"  {hit.file}:{hit.line_number}: {hit.line}")
    if len(matches) > MAX_SEARCH_HITS:
        print(f"  ... ({len(matches) - MAX_SEARCH_HITS} more)")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test CodeSearchTool against Settings.github_repo."
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Literal keyword to search for (default: {DEFAULT_QUERY!r})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    repos_json = Path(__file__).resolve().parent / "repos.json"
    primary = settings.github_repo

    with tempfile.TemporaryDirectory(prefix="agentic-flow-step3-") as tmp:
        tool = CodeSearchTool(tmp)

        try:
            config = tool.load_linked_repos(str(repos_json))
            print(f"Primary (from Settings): {primary}")
            linked = ", ".join(
                f"{item.name}={item.repo}" for item in config.linked
            ) or "(none)"
            print(f"Linked (from repos.json): {linked}")
            print()

            if _exercise_repo(
                tool,
                primary,
                settings.github_token,
                args.query,
                label="primary",
            ):
                return 1

            for item in config.linked:
                if _exercise_repo(
                    tool,
                    item.repo,
                    settings.github_token,
                    args.query,
                    label=f"linked:{item.name}",
                ):
                    return 1
        except ToolError as exc:
            print(f"Tool error: {exc}", file=sys.stderr)
            return 1

    print("Step 3 smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
