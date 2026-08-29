#!/usr/bin/env python3
"""Entry point for the agentic-flow GitHub Action and local CLI runs."""

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
from tools.code_search import CodeSearchTool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
AWAITING_APPROVAL_LABEL = "agent:awaiting-approval"


def _failure_comment(exc: BaseException) -> str:
    """Build a GitHub comment body when the agent pipeline fails."""
    return (
        "## Agent error\n\n"
        "The agent encountered an error while investigating this issue and "
        "could not post a proposal.\n\n"
        f"**Error:** {exc}\n\n"
        "Please check the workflow logs for details, or re-trigger the "
        "workflow when ready."
    )


def _post_failure_comment(
    adapter: GitHubAdapter | None,
    issue_number: int,
    exc: BaseException,
) -> None:
    """Best-effort error comment so failures are visible on the issue."""
    if adapter is None:
        return
    try:
        adapter.post_comment(issue_number, _failure_comment(exc))
        logger.info("Posted failure notice on issue #%s", issue_number)
    except Exception as post_exc:
        logger.exception(
            "Could not post failure comment on issue #%s: %s",
            issue_number,
            post_exc,
        )


def main() -> int:
    """Investigate an issue, propose fixes, and post the proposal as a comment."""
    parser = argparse.ArgumentParser(
        description=(
            "Investigate a GitHub issue, propose fix approaches, and post "
            "the proposal for human approval."
        )
    )
    parser.add_argument(
        "--issue-number",
        type=int,
        required=True,
        help="GitHub issue number to investigate and propose fixes for",
    )
    args = parser.parse_args()
    issue_number = args.issue_number

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    adapter: GitHubAdapter | None = None

    try:
        settings = Settings.from_env()
        adapter = GitHubAdapter(settings)
        issue = adapter.get_issue(issue_number)

        model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        repos_json = Path(__file__).resolve().parent / "repos.json"

        logger.info(
            "Starting pipeline for issue #%s in %s: %s",
            issue_number,
            settings.github_repo,
            issue["title"],
        )

        with tempfile.TemporaryDirectory(prefix="agentic-flow-") as tmp:
            tool = CodeSearchTool(tmp)
            client = Anthropic(api_key=settings.anthropic_api_key)
            investigator = InvestigatorAgent(
                client,
                model,
                tool,
                settings.github_token,
                linked_config_path=str(repos_json),
            )
            investigation = investigator.investigate(
                issue["title"],
                issue["body"],
                settings.github_repo,
            )
            proposer = ProposerAgent(client, model)
            proposal = proposer.propose(investigation)

        comment_body = proposer.format_as_comment(proposal)
        comment = adapter.post_comment(issue_number, comment_body)
        adapter.add_label(issue_number, AWAITING_APPROVAL_LABEL)

        logger.info(
            "Posted proposal on issue #%s (comment id=%s, approaches=%d)",
            issue_number,
            comment["id"],
            len(proposal.approaches),
        )
        return 0

    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except (AdapterError, AgentError) as exc:
        logger.exception("Agent pipeline failed: %s", exc)
        _post_failure_comment(adapter, issue_number, exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during agent pipeline: %s", exc)
        _post_failure_comment(adapter, issue_number, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
