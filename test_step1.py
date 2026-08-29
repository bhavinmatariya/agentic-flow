#!/usr/bin/env python3
"""Manual smoke test for Step 1: config layer and GitHub adapter.

Usage:
    python test_step1.py <issue_number> [--label LABEL]

Requires a populated .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adapters.base import AdapterError
from adapters.github_adapter import GitHubAdapter
from config import ConfigurationError, Settings

DEFAULT_TEST_LABEL = "agentic-flow-test"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test GitHubAdapter against a real issue."
    )
    parser.add_argument(
        "issue_number",
        type=int,
        help="GitHub issue number to fetch and exercise",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_TEST_LABEL,
        help=f"Label to add/remove during the test (default: {DEFAULT_TEST_LABEL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        adapter = GitHubAdapter(settings)
    except AdapterError as exc:
        print(f"GitHub adapter error: {exc}", file=sys.stderr)
        return 1

    issue_number = args.issue_number
    label = args.label

    print(f"Repository: {settings.github_repo}")
    print(f"Issue number: {issue_number}")
    print()

    issue = adapter.get_issue(issue_number)
    print(f"Title: {issue['title']}")
    print(f"State: {issue['state']}")
    print(f"Labels: {', '.join(issue['labels']) or '(none)'}")
    print()

    comment = adapter.post_comment(
        issue_number,
        "🤖 **agentic-flow Step 1 smoke test** — comment posted by `test_step1.py`.",
    )
    print(f"Posted comment id={comment['id']} by {comment['author']}")

    adapter.add_label(issue_number, label)
    print(f"Added label: {label!r}")

    if adapter.has_label(issue_number, label):
        print(f"Confirmed: issue #{issue_number} has label {label!r}")
    else:
        print(f"ERROR: label {label!r} not found after add_label()", file=sys.stderr)
        return 1

    adapter.remove_label(issue_number, label)
    print(f"Removed label: {label!r}")

    if adapter.has_label(issue_number, label):
        print(f"ERROR: label {label!r} still present after remove_label()", file=sys.stderr)
        return 1

    print(f"Confirmed: label {label!r} removed")
    print()
    print("Step 1 smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
