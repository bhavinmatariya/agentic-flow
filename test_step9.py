#!/usr/bin/env python3
"""Manual smoke test for Step 9: full pipeline through PR creation.

Usage:
    python test_step9.py <issue_number> [--approach-index N]

Requires a populated .env file (see .env.example). Runs investigator,
proposer, picks one approach, then ImplementationOrchestrator.run().
Opens a real PR on success or prints the stall diagnostic on failure.
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
from agents.reviewer import ReviewerAgent
from agents.task_decomposer import TaskDecomposerAgent
from config import ConfigurationError, Settings
from core.exceptions import AgentError
from core.models import Investigation, Proposal
from core.orchestrator import ImplementationOrchestrator
from tools.browser_test import BrowserTestTool
from tools.code_editor import CodeEditTool
from tools.code_search import CodeSearchTool
from tools.db_verifier import DBVerifierTool
from tools.environment_manager import EnvironmentManager

DEFAULT_MODEL = "claude-sonnet-5"


def _print_investigation(result: Investigation) -> None:
    print("=== Investigation ===")
    print(f"issue_nature: {result.issue_nature}")
    print(f"root_cause: {result.root_cause}")
    print()


def _print_proposal(result: Proposal, selected_index: int) -> None:
    print("=== Proposal ===")
    for index, approach in enumerate(result.approaches, start=1):
        marker = " <-- selected" if index - 1 == selected_index else ""
        print(f"  {index}. {approach.name}{marker}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full agent pipeline through PR creation."
    )
    parser.add_argument("issue_number", type=int, help="GitHub issue number")
    parser.add_argument(
        "--approach-index",
        type=int,
        default=0,
        help="Zero-based approach index to implement (default: 0)",
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

    with tempfile.TemporaryDirectory(prefix="agentic-flow-step9-") as tmp:
        search_tool = CodeSearchTool(tmp)
        edit_tool = CodeEditTool(adapter)
        environment = EnvironmentManager()
        browser = BrowserTestTool()
        db_verifier = DBVerifierTool()
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
        reviewer = ReviewerAgent(
            client,
            args.model,
            settings,
            search_tool,
            environment,
            browser,
            db_verifier,
            settings.github_token,
        )
        decomposer = TaskDecomposerAgent(client, args.model, settings)
        orchestrator = ImplementationOrchestrator(
            adapter, implementer, reviewer, decomposer
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

        if args.approach_index < 0 or args.approach_index >= len(proposal.approaches):
            print("Invalid --approach-index.", file=sys.stderr)
            return 1

        selected = proposal.approaches[args.approach_index]
        _print_proposal(proposal, args.approach_index)
        print(f"Running orchestrator with approach: {selected.name}")
        print()

        try:
            result = orchestrator.run(
                issue,
                investigation,
                selected,
                settings.github_repo,
                args.issue_number,
                proposal=proposal,
            )
        except (AdapterError, AgentError) as exc:
            print(f"Orchestrator error: {exc}", file=sys.stderr)
            return 1

    if result.passed and result.pr_url:
        print("=== Success ===")
        print(f"PR URL: {result.pr_url}")
        print()
        print("Step 9 smoke test passed.")
        return 0

    print("=== Orchestrator stalled ===")
    print(result.diagnostic_comment or "(no diagnostic comment)")
    print()
    print("Step 9 completed with failure diagnostics (no PR opened).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
