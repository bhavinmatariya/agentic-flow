#!/usr/bin/env python3
"""Manual smoke test for Step 5: investigator + proposer agents.

Usage:
    python test_step5.py <issue_number>

Requires a populated .env file (see .env.example). Clones into a temporary
directory for the investigator, then runs the proposer on the investigation.
The formatted comment is printed only — nothing is posted to GitHub.
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
from agents.investigator import InvestigatorAgent
from agents.proposer import ProposerAgent
from config import ConfigurationError, Settings
from core.exceptions import AgentError
from core.models import Investigation, Proposal
from tools.code_search import CodeSearchTool

DEFAULT_MODEL = "claude-sonnet-5"


def _print_investigation(result: Investigation) -> None:
    """Pretty-print every Investigation field to stdout."""
    print("=== Investigation ===")
    print()
    print("issue_nature:")
    print(f"  {result.issue_nature}")
    print()
    print("root_cause:")
    print(f"  {result.root_cause}")
    print()
    print("evidence:")
    if result.evidence:
        for item in result.evidence:
            print(f"  - {item}")
    else:
        print("  (none)")
    print()
    print("relevant_files:")
    if result.relevant_files:
        for file in result.relevant_files:
            print(f"  - {file.repo}:{file.path}")
            print(f"      {file.reason}")
    else:
        print("  (none)")
    print()
    print(f"confidence: {result.confidence}")
    print()
    print("open_questions:")
    if result.open_questions:
        for question in result.open_questions:
            print(f"  - {question}")
    else:
        print("  (none)")
    print()


def _print_proposal(result: Proposal) -> None:
    """Pretty-print every Approach in a Proposal to stdout."""
    print("=== Proposal (raw) ===")
    print()
    print(f"approach_count: {len(result.approaches)}")
    print()
    for index, approach in enumerate(result.approaches, start=1):
        print(f"--- Approach {index}: {approach.name} ---")
        print(f"nature: {approach.nature}")
        print(f"description: {approach.description}")
        print(f"why_it_works: {approach.why_it_works}")
        print(f"risk: {approach.risk}")
        print(f"tradeoffs: {approach.tradeoffs}")
        print(f"estimated_scope: {approach.estimated_scope}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run InvestigatorAgent then ProposerAgent on a real issue."
    )
    parser.add_argument(
        "issue_number",
        type=int,
        help="GitHub issue number to investigate and propose fixes for",
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

    with tempfile.TemporaryDirectory(prefix="agentic-flow-step5-") as tmp:
        tool = CodeSearchTool(tmp)
        client = Anthropic(api_key=settings.anthropic_api_key)
        investigator = InvestigatorAgent(
            client,
            args.model,
            tool,
            settings.github_token,
            linked_config_path=str(repos_json),
        )
        proposer = ProposerAgent(client, args.model)

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

    _print_proposal(proposal)

    comment = proposer.format_as_comment(proposal)
    print("=== Formatted GitHub comment (not posted) ===")
    print()
    print(comment)
    print()
    print("Step 5 smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
