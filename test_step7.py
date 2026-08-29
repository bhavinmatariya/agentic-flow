#!/usr/bin/env python3
"""Manual smoke test for Step 7: full pipeline through implementer.

Usage:
    python test_step7.py <issue_number> [--approach-index N]

Requires a populated .env file (see .env.example). Runs investigator,
proposer, picks one approach by index, then implementer. Commits to a real
GitHub branch — verify the diff manually on GitHub afterward.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from anthropic import Anthropic

from adapters.base import AdapterError
from adapters.github_adapter import GitHubAdapter
from agents.implementer import ImplementerAgent
from agents.investigator import InvestigatorAgent
from agents.proposer import ProposerAgent
from config import ConfigurationError, Settings
from core.exceptions import AgentError
from core.models import ImplementationResult, Investigation, Proposal
from tools.code_editor import CodeEditTool
from tools.code_search import CodeSearchTool

DEFAULT_MODEL = "claude-sonnet-5"


def _print_investigation(result: Investigation) -> None:
    """Pretty-print investigation fields."""
    print("=== Investigation ===")
    print(f"issue_nature: {result.issue_nature}")
    print(f"root_cause: {result.root_cause}")
    print(f"confidence: {result.confidence}")
    print()


def _print_proposal(result: Proposal, selected_index: int) -> None:
    """Pretty-print proposal and the selected approach."""
    print("=== Proposal ===")
    print(f"approach_count: {len(result.approaches)}")
    for index, approach in enumerate(result.approaches, start=1):
        marker = " <-- selected" if index - 1 == selected_index else ""
        print(f"  {index}. {approach.name}{marker}")
    print()


def _print_implementation(result: ImplementationResult) -> None:
    """Pretty-print implementation result fields."""
    print("=== ImplementationResult ===")
    print(f"branch_name: {result.branch_name}")
    print(f"summary: {result.summary}")
    print("files_changed:")
    if result.files_changed:
        for path in result.files_changed:
            print(f"  - {path}")
    else:
        print("  (none listed)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run investigator, proposer, and implementer against a real issue."
        )
    )
    parser.add_argument(
        "issue_number",
        type=int,
        help="GitHub issue number to fix",
    )
    parser.add_argument(
        "--approach-index",
        type=int,
        default=0,
        help="Zero-based index of the proposed approach to implement (default: 0)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        help=f"Anthropic model id (default: {DEFAULT_MODEL})",
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

    try:
        adapter = GitHubAdapter(settings)
        issue = adapter.get_issue(args.issue_number)
    except AdapterError as exc:
        print(f"GitHub adapter error: {exc}", file=sys.stderr)
        return 1

    print(f"Repository: {settings.github_repo}")
    print(f"Issue #{issue['number']}: {issue['title']}")
    print()

    with tempfile.TemporaryDirectory(prefix="agentic-flow-step7-") as tmp:
        search_tool = CodeSearchTool(tmp)
        edit_tool = CodeEditTool(adapter)
        client = Anthropic(api_key=settings.anthropic_api_key)
        investigator = InvestigatorAgent(
            client,
            args.model,
            settings,
            search_tool,
            settings.github_token,
            linked_config_path=str(repos_json),
        )
        proposer = ProposerAgent(client, args.model, settings)
        implementer = ImplementerAgent(
            client,
            args.model,
            settings,
            search_tool,
            edit_tool,
            settings.github_token,
        )

        try:
            investigation = investigator.investigate(
                issue["title"],
                issue["body"],
                settings.github_repo,
            )
        except AgentError as exc:
            print(f"Investigator error: {exc}", file=sys.stderr)
            return 1

        _print_investigation(investigation)

        try:
            proposal = proposer.propose(investigation)
        except AgentError as exc:
            print(f"Proposer error: {exc}", file=sys.stderr)
            return 1

        if not proposal.approaches:
            print("Proposer returned no approaches.", file=sys.stderr)
            return 1
        if args.approach_index < 0 or args.approach_index >= len(proposal.approaches):
            print(
                f"Invalid --approach-index {args.approach_index}; "
                f"proposal has {len(proposal.approaches)} approach(es).",
                file=sys.stderr,
            )
            return 1

        selected = proposal.approaches[args.approach_index]
        _print_proposal(proposal, args.approach_index)
        print(f"Implementing approach: {selected.name}")
        print()

        try:
            implementation = implementer.implement(
                issue,
                investigation,
                selected,
                settings.github_repo,
            )
        except AgentError as exc:
            print(f"Implementer error: {exc}", file=sys.stderr)
            return 1

    _print_implementation(implementation)
    print(
        "Inspect the branch on GitHub and confirm only the intended diff was committed."
    )
    print()
    print("Step 7 smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
