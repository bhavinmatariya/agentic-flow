#!/usr/bin/env python3
"""Single entrypoint for the agentic-flow GitHub Action and local CLI runs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from anthropic import Anthropic

from adapters.base import AdapterError
from adapters.github_adapter import GitHubAdapter
from agents.implementer import ImplementerAgent
from agents.investigator import InvestigatorAgent
from agents.proposer import PROPOSAL_SECTION_HEADER, ProposerAgent
from agents.response_parser import ResponseParserAgent
from agents.reviewer import ReviewerAgent
from config import ConfigurationError, Settings
from core.exceptions import AgentError
from core.models import ParsedIntent
from core.orchestrator import (
    DONE_LABEL,
    IN_PROGRESS_LABEL,
    ImplementationOrchestrator,
    resolve_approach,
)
from tools.browser_test import BrowserTestTool
from tools.code_editor import CodeEditTool
from tools.code_search import CodeSearchTool
from tools.db_verifier import DBVerifierTool
from tools.environment_manager import EnvironmentManager
from utils.logger import RunReporter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
AWAITING_APPROVAL_LABEL = "agent:awaiting-approval"
PROPOSAL_COMMENT_HEADER = PROPOSAL_SECTION_HEADER


@dataclass
class _PipelineAgents:
    """Agents and tools shared across the post-approval pipeline."""

    client: Anthropic
    investigator: InvestigatorAgent
    proposer: ProposerAgent
    implementer: ImplementerAgent
    reviewer: ReviewerAgent
    orchestrator: ImplementationOrchestrator


def _error_summary(exc: BaseException) -> str:
    """Return a short, user-facing error summary without a stack trace."""
    message = str(exc).strip().replace("\r", " ")
    first_line = message.split("\n", 1)[0].strip()
    return first_line[:500] if first_line else type(exc).__name__


def _agent_error_comment(exc: BaseException) -> str:
    """Build a GitHub comment for an unhandled pipeline failure."""
    return (
        "## Agent error\n\n"
        "The agent hit an error and could not finish this run.\n\n"
        f"**Summary:** {_error_summary(exc)}\n\n"
        "Please check the workflow logs for full details and re-trigger when ready."
    )


def _post_agent_error(
    adapter: GitHubAdapter | None,
    issue_number: int,
    exc: BaseException,
) -> None:
    """Post a short failure comment and log the full exception."""
    logger.exception("Pipeline failed for issue #%s: %s", issue_number, exc)
    if adapter is None:
        return
    try:
        adapter.post_comment(issue_number, _agent_error_comment(exc))
    except Exception as post_exc:
        logger.exception(
            "Could not post agent error comment on issue #%s: %s",
            issue_number,
            post_exc,
        )


def _find_latest_proposal_comment(
    comments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the most recent agent proposal comment, if any."""
    proposals = [
        comment
        for comment in comments
        if isinstance(comment.get("body"), str)
        and PROPOSAL_COMMENT_HEADER in comment["body"]
    ]
    if not proposals:
        return None
    return max(proposals, key=lambda comment: str(comment.get("created_at", "")))


def _build_pipeline_agents(
    *,
    settings: Settings,
    adapter: GitHubAdapter,
    model: str,
    tmp_dir: str,
    repos_json: Path,
) -> _PipelineAgents:
    """Construct agents for investigate/propose/implement/review orchestration."""
    search_tool = CodeSearchTool(tmp_dir)
    edit_tool = CodeEditTool(adapter)
    client = Anthropic(api_key=settings.anthropic_api_key)

    investigator = InvestigatorAgent(
        client,
        model,
        settings,
        search_tool,
        settings.github_token,
        linked_config_path=str(repos_json),
    )
    proposer = ProposerAgent(client, model, settings)
    implementer = ImplementerAgent(
        client,
        model,
        settings,
        search_tool,
        edit_tool,
        settings.github_token,
    )

    # Live-verification tools are always available to the reviewer; it uses them
    # only when both frontend and database layers are detected during review.
    environment = EnvironmentManager()
    browser = BrowserTestTool()
    db_verifier = DBVerifierTool()
    reviewer = ReviewerAgent(
        client,
        model,
        settings,
        search_tool,
        environment,
        browser,
        db_verifier,
        settings.github_token,
    )
    orchestrator = ImplementationOrchestrator(adapter, implementer, reviewer)

    return _PipelineAgents(
        client=client,
        investigator=investigator,
        proposer=proposer,
        implementer=implementer,
        reviewer=reviewer,
        orchestrator=orchestrator,
    )


def _handle_issue_opened(
    adapter: GitHubAdapter,
    settings: Settings,
    issue_number: int,
    model: str,
    repos_json: Path,
    reporter: RunReporter,
) -> int:
    """Investigate an issue, propose fixes, and post the proposal comment."""
    issue = adapter.get_issue(issue_number)
    logger.info(
        "Issue opened pipeline for #%s in %s: %s",
        issue_number,
        settings.github_repo,
        issue["title"],
    )

    with tempfile.TemporaryDirectory(prefix="agentic-flow-open-") as tmp:
        agents = _build_pipeline_agents(
            settings=settings,
            adapter=adapter,
            model=model,
            tmp_dir=tmp,
            repos_json=repos_json,
        )
        with reporter.stage("INVESTIGATING", 40):
            investigation = agents.investigator.investigate(
                issue["title"],
                issue["body"],
                settings.github_repo,
            )
        with reporter.stage("PROPOSING", 75):
            proposal = agents.proposer.propose(investigation)
            comment_body = agents.proposer.format_as_comment(proposal, investigation)
            comment = adapter.post_comment(issue_number, comment_body)
            adapter.add_label(issue_number, AWAITING_APPROVAL_LABEL)
            logger.info(
                "Posted proposal on issue #%s (comment id=%s, approaches=%d)",
                issue_number,
                comment["id"],
                len(proposal.approaches),
            )

    reporter.record_outcome_proposal_posted(approaches=len(proposal.approaches))
    return 0


def _handle_issue_comment(
    adapter: GitHubAdapter,
    settings: Settings,
    issue_number: int,
    model: str,
    comment_body: str,
    comment_author: str,
    repos_json: Path,
    reporter: RunReporter,
) -> int:
    """Parse a human reply and run the full pipeline when approved."""
    if "agentic-flow:auto" in comment_body:
        logger.info("Ignoring bot's own comment, exiting.")
        reporter.record_outcome_noop("Ignoring bot's own comment")
        return 0

    if not adapter.has_label(issue_number, AWAITING_APPROVAL_LABEL):
        reason = (
            f"Issue #{issue_number} does not have label {AWAITING_APPROVAL_LABEL!r}; "
            "nothing to do"
        )
        logger.info(reason)
        reporter.record_outcome_noop(reason)
        return 0

    issue = adapter.get_issue(issue_number)
    comments = adapter.list_comments(issue_number)
    proposal_comment = _find_latest_proposal_comment(comments)
    if proposal_comment is None:
        raise AgentError(
            f"Issue #{issue_number} has {AWAITING_APPROVAL_LABEL!r} but no "
            f"proposal comment containing {PROPOSAL_COMMENT_HEADER!r} was found."
        )

    logger.info(
        "Processing issue comment on #%s from %r",
        issue_number,
        comment_author,
    )

    with tempfile.TemporaryDirectory(prefix="agentic-flow-comment-") as tmp:
        agents = _build_pipeline_agents(
            settings=settings,
            adapter=adapter,
            model=model,
            tmp_dir=tmp,
            repos_json=repos_json,
        )
        with reporter.stage("PARSING", 15):
            parser = ResponseParserAgent(agents.client, model, settings)
            parsed = parser.parse(
                issue["title"],
                issue["body"],
                str(proposal_comment["body"]),
                comment_body,
            )
            logger.info(
                "Parsed human reply on issue #%s as intent=%r",
                issue_number,
                parsed.intent,
            )

        if parsed.intent == "unrelated":
            reason = f"Human reply on issue #{issue_number} classified as unrelated"
            logger.info("%s; no action taken", reason)
            reporter.record_outcome_noop(reason)
            return 0

        if parsed.intent == "revise":
            with reporter.stage("PROPOSING", 50):
                adapter.post_comment(
                    issue_number,
                    (
                        "Thanks — we received your feedback and will use it when "
                        "preparing a revised proposal.\n\n"
                        f"> {parsed.feedback}"
                    ),
                )
                logger.info(
                    "Recorded revision feedback on issue #%s; label unchanged",
                    issue_number,
                )
            reporter.record_outcome_noop(
                f"Revision feedback recorded on issue #{issue_number}"
            )
            return 0

        return _handle_approval(
            adapter=adapter,
            settings=settings,
            issue=issue,
            issue_number=issue_number,
            parsed=parsed,
            comment_body=comment_body,
            agents=agents,
            reporter=reporter,
        )


def _handle_approval(
    *,
    adapter: GitHubAdapter,
    settings: Settings,
    issue: dict[str, Any],
    issue_number: int,
    parsed: ParsedIntent,
    comment_body: str,
    agents: _PipelineAgents,
    reporter: RunReporter,
) -> int:
    """Run implement → review → PR after a human approves an approach."""
    adapter.remove_label(issue_number, AWAITING_APPROVAL_LABEL)
    adapter.add_label(issue_number, IN_PROGRESS_LABEL)

    selected_name = parsed.selected_approach or "the selected approach"
    adapter.post_comment(
        issue_number,
        (
            f"Understood — proceeding with **{selected_name}**.\n\n"
            "Investigating, implementing, reviewing, and opening a pull request."
        ),
    )

    with reporter.stage("INVESTIGATING", 25):
        investigation = agents.investigator.investigate(
            issue["title"],
            issue["body"],
            settings.github_repo,
        )
    with reporter.stage("PROPOSING", 40):
        proposal = agents.proposer.propose(investigation)
        approach = resolve_approach(proposal.approaches, parsed.selected_approach)
        logger.info(
            "Running orchestrator for issue #%s with approach %r",
            issue_number,
            approach.name,
        )

    result = agents.orchestrator.run(
        issue,
        investigation,
        approach,
        settings.github_repo,
        issue_number,
        human_approval_text=comment_body,
        reporter=reporter,
    )

    if result.passed and result.pr_url:
        if adapter.has_label(issue_number, IN_PROGRESS_LABEL):
            adapter.remove_label(issue_number, IN_PROGRESS_LABEL)
        adapter.add_label(issue_number, DONE_LABEL)
        adapter.post_comment(
            issue_number,
            f"Pull request opened for the approved fix:\n\n{result.pr_url}",
        )
        files = (
            list(result.implementation_result.files_changed)
            if result.implementation_result
            else []
        )
        reporter.record_outcome_pr_opened(
            result.pr_url,
            round_count=len(result.round_history),
            files=files,
        )
        return 0

    files = (
        list(result.implementation_result.files_changed)
        if result.implementation_result
        else []
    )
    reporter.record_outcome_needs_human(
        round_count=len(result.round_history),
        files=files,
    )
    logger.warning(
        "Pipeline stalled for issue #%s; diagnostic already posted by orchestrator",
        issue_number,
    )
    return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the agentic-flow pipeline for issue-opened or issue-comment events."
        )
    )
    parser.add_argument(
        "--event",
        choices=("issue_opened", "issue_comment"),
        required=True,
        help="GitHub event driving this run",
    )
    parser.add_argument(
        "--issue-number",
        type=int,
        required=True,
        help="GitHub issue number to process",
    )
    parser.add_argument(
        "--comment-body",
        default=None,
        help="Body of the triggering issue comment (required for issue_comment)",
    )
    parser.add_argument(
        "--comment-author",
        default=None,
        help="Login of the comment author (required for issue_comment)",
    )
    args = parser.parse_args(argv)

    if args.event == "issue_comment":
        if args.comment_author is None or not str(args.comment_author).strip():
            parser.error("--comment-author is required when --event is issue_comment")
        if args.comment_body is None:
            parser.error("--comment-body is required when --event is issue_comment")

    return args


def main(argv: list[str] | None = None) -> int:
    """Run the full agentic-flow pipeline for the requested GitHub event."""
    args = _parse_args(argv)
    issue_number = args.issue_number
    reporter = RunReporter(issue_number=issue_number, event=args.event)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    reporter.print_startup_banner()

    adapter: GitHubAdapter | None = None
    exit_code = 1

    try:
        settings = Settings.from_env()
        reporter.repo = settings.github_repo
        adapter = GitHubAdapter(settings)
        model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        repos_json = Path(__file__).resolve().parent / "repos.json"

        if args.event == "issue_opened":
            exit_code = _handle_issue_opened(
                adapter,
                settings,
                issue_number,
                model,
                repos_json,
                reporter,
            )
        else:
            exit_code = _handle_issue_comment(
                adapter,
                settings,
                issue_number,
                model,
                str(args.comment_body),
                str(args.comment_author).strip(),
                repos_json,
                reporter,
            )

    except ConfigurationError as exc:
        error_text = f"Configuration error: {exc}"
        logger.error(error_text)
        reporter.record_outcome_error(error_text)
        exit_code = 1
    except (AdapterError, AgentError) as exc:
        _post_agent_error(adapter, issue_number, exc)
        reporter.record_outcome_error(_error_summary(exc))
        exit_code = 1
    except Exception as exc:
        _post_agent_error(adapter, issue_number, exc)
        reporter.record_outcome_error(_error_summary(exc))
        exit_code = 1
    finally:
        reporter.write_step_summary(exit_code)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
