#!/usr/bin/env python3
"""Single entrypoint for the agentic-flow GitHub Action and local CLI runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from anthropic import Anthropic
from github import GithubException

from adapters.github_adapter import GitHubAdapter
from agents.implementer import ImplementerAgent
from agents.investigator import InvestigatorAgent
from agents.proposer import PROPOSAL_SECTION_HEADER, ProposerAgent
from agents.response_parser import ResponseParserAgent
from agents.reviewer import ReviewerAgent
from agents.task_decomposer import TaskDecomposerAgent
from config import ConfigurationError, Settings
from core.exceptions import AgentError
from core.models import Approach, Investigation, ParsedIntent, Proposal, SubtaskPlan, MAX_SUBTASKS
from core.pipeline_state import (
    RESTART_INVESTIGATION_MODE,
    STATE_MARKER,
    find_latest_state_comment,
    format_state_comment,
    parse_state_comment,
)
from core.orchestrator import (
    DEFAULT_MAX_ROUNDS_PER_SUBTASK,
    DONE_LABEL,
    IN_PROGRESS_LABEL,
    NEEDS_HUMAN_LABEL,
    ImplementationOrchestrator,
    OrchestratorResult,
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


def _default_fix_branch(issue_number: int) -> str:
    return f"agent/fix-issue-{issue_number}"


def _assert_repo_pipeline_available(
    adapter: GitHubAdapter,
    issue_number: int,
) -> None:
    """Refuse to start when another issue in this repo already has in-progress work."""
    active = adapter.list_open_issue_numbers_with_label(IN_PROGRESS_LABEL)
    others = [number for number in active if number != issue_number]
    if not others:
        return
    preview = ", ".join(f"#{number}" for number in others[:5])
    if len(others) > 5:
        preview += f" (+{len(others) - 5} more)"
    raise AgentError(
        f"Cannot start issue #{issue_number}: the agent is already running on "
        f"{preview}. Wait until that run completes, then try again."
    )


def _post_proposal_state(
    adapter: GitHubAdapter,
    issue_number: int,
    investigation: Investigation,
    proposal: Proposal,
) -> None:
    """Post hidden investigation + full proposal before waiting for human approval."""
    branch = _default_fix_branch(issue_number)
    adapter.post_comment(
        issue_number,
        format_state_comment(
            branch,
            investigation=investigation,
            proposal=proposal,
        ),
    )
    logger.info(
        "Posted proposal state for issue #%s (branch=%r, approaches=%d)",
        issue_number,
        branch,
        len(proposal.approaches),
    )


def _post_approval_state(
    adapter: GitHubAdapter,
    issue_number: int,
    investigation: Investigation,
    approach: Approach,
    proposal: Proposal | None = None,
    *,
    subtask_plan: SubtaskPlan | None = None,
    subtask_index: int | None = None,
    checkpoint_completed: int | None = None,
) -> None:
    """Post hidden state with the human-selected approach for resume after stalls."""
    branch = _default_fix_branch(issue_number)
    adapter.post_comment(
        issue_number,
        format_state_comment(
            branch,
            investigation=investigation,
            proposal=proposal,
            approach=approach,
            subtask_plan=subtask_plan,
            subtask_index=subtask_index,
            checkpoint_completed=checkpoint_completed,
        ),
    )
    logger.info(
        "Posted approval state for issue #%s (branch=%r, approach=%r)",
        issue_number,
        branch,
        approach.name,
    )


def _ensure_state_comment_for_needs_human(
    adapter: GitHubAdapter,
    issue_number: int,
) -> None:
    """Ensure a resume marker exists before labeling ``agent:needs-human``."""
    comments = adapter.list_comments(issue_number)
    if find_latest_state_comment(comments) is not None:
        return
    branch = _default_fix_branch(issue_number)
    adapter.post_comment(
        issue_number,
        format_state_comment(branch, resume_mode=RESTART_INVESTIGATION_MODE),
    )
    logger.info(
        "Posted restart state marker for issue #%s (branch=%r)",
        issue_number,
        branch,
    )


@dataclass
class _PipelineAgents:
    """Agents and tools shared across the post-approval pipeline."""

    client: Anthropic
    investigator: InvestigatorAgent
    proposer: ProposerAgent
    implementer: ImplementerAgent
    reviewer: ReviewerAgent
    orchestrator: ImplementationOrchestrator


def _plain_language_error(exc: BaseException) -> str:
    """Map an exception to a short plain-language explanation for issue comments."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GithubException):
            return "GitHub API had a temporary problem."
        if isinstance(current, (json.JSONDecodeError, ValidationError)):
            return "The AI's response was malformed."
        if isinstance(current, subprocess.CalledProcessError):
            command = current.cmd
            command_text = (
                " ".join(str(part) for part in command)
                if isinstance(command, (list, tuple))
                else str(command)
            )
            if "git" in command_text.lower():
                return "A git operation failed."
        current = current.__cause__ or current.__context__
    return "An unexpected internal error occurred."


def _error_summary(exc: BaseException) -> str:
    """Return a short, user-facing error summary without a stack trace."""
    message = str(exc).strip().replace("\r", " ")
    first_line = message.split("\n", 1)[0].strip()
    return first_line[:500] if first_line else type(exc).__name__


def _post_unhandled_pipeline_failure(
    adapter: GitHubAdapter | None,
    issue_number: int,
    exc: BaseException,
    reporter: RunReporter,
) -> None:
    """Log technical details, post a plain-language issue comment, and label needs-human."""
    print("::group::Technical details", flush=True)
    logger.exception("Unhandled pipeline failure for issue #%s", issue_number)
    print(traceback.format_exc(), flush=True)
    print("::endgroup::", flush=True)

    plain = _plain_language_error(exc)
    short = _error_summary(exc)
    comment = (
        "## Agent error\n\n"
        f"{plain}\n\n"
        f"**Details:** {short}\n\n"
        "The issue has been labeled **`agent:needs-human`**. "
        "Comment **continue** or **retry** on the issue to resume saved progress, "
        "or check the workflow logs for full details."
    )

    if adapter is None:
        reporter.record_outcome_needs_human(round_count=0)
        return

    try:
        _ensure_state_comment_for_needs_human(adapter, issue_number)
        adapter.post_comment(issue_number, comment)
        if adapter.has_label(issue_number, IN_PROGRESS_LABEL):
            adapter.remove_label(issue_number, IN_PROGRESS_LABEL)
        if adapter.has_label(issue_number, AWAITING_APPROVAL_LABEL):
            adapter.remove_label(issue_number, AWAITING_APPROVAL_LABEL)
        adapter.add_label(issue_number, NEEDS_HUMAN_LABEL)
    except Exception as post_exc:
        logger.exception(
            "Could not post failure comment or label on issue #%s: %s",
            issue_number,
            post_exc,
        )

    reporter.record_outcome_needs_human(round_count=0)


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
    decomposer = TaskDecomposerAgent(client, model, settings)
    orchestrator = ImplementationOrchestrator(
        adapter,
        implementer,
        reviewer,
        decomposer,
        review_strategy=settings.review_strategy,
    )

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
            _post_proposal_state(adapter, issue_number, investigation, proposal)
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

    if adapter.has_label(issue_number, NEEDS_HUMAN_LABEL):
        logger.info(
            "Issue #%s has %r; resuming from saved state (comment=%r)",
            issue_number,
            NEEDS_HUMAN_LABEL,
            comment_body.strip()[:120],
        )
        return _handle_resume(
            adapter=adapter,
            settings=settings,
            issue_number=issue_number,
            model=model,
            comment_body=comment_body,
            repos_json=repos_json,
            reporter=reporter,
        )

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
            with reporter.stage("INVESTIGATING", 30):
                investigation = agents.investigator.investigate(
                    issue["title"],
                    issue["body"],
                    settings.github_repo,
                )
            with reporter.stage("PROPOSING", 60):
                proposal = agents.proposer.propose(
                    investigation,
                    revision_feedback=parsed.feedback,
                )
                revised_comment_body = agents.proposer.format_as_comment(
                    proposal,
                    investigation,
                )
                comment = adapter.post_comment(issue_number, revised_comment_body)
                _post_proposal_state(adapter, issue_number, investigation, proposal)
                logger.info(
                    "Posted revised proposal on issue #%s (comment id=%s, "
                    "approaches=%d); %r label kept",
                    issue_number,
                    comment["id"],
                    len(proposal.approaches),
                    AWAITING_APPROVAL_LABEL,
                )
            reporter.record_outcome_proposal_posted(
                approaches=len(proposal.approaches)
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


def _finish_orchestrator_run(
    *,
    adapter: GitHubAdapter,
    issue_number: int,
    result: OrchestratorResult,
    reporter: RunReporter,
) -> int:
    """Apply labels/comments after orchestrator.run completes."""
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


def _handle_resume(
    *,
    adapter: GitHubAdapter,
    settings: Settings,
    issue_number: int,
    model: str,
    comment_body: str,
    repos_json: Path,
    reporter: RunReporter,
) -> int:
    """Resume implement/review from saved state on an existing fix branch."""
    issue = adapter.get_issue(issue_number)
    comments = adapter.list_comments(issue_number)
    state_comment = find_latest_state_comment(comments)
    if state_comment is None:
        raise AgentError(
            f"Issue #{issue_number} has {NEEDS_HUMAN_LABEL!r} but no "
            f"{STATE_MARKER!r} state comment was found to resume from."
        )

    try:
        state = parse_state_comment(str(state_comment["body"]))
    except AgentError as exc:
        raise AgentError(
            f"Issue #{issue_number} saved progress could not be loaded: {exc} "
            f"(max {MAX_SUBTASKS} subtasks in stored plans)."
        ) from exc

    if state.resume_mode == RESTART_INVESTIGATION_MODE:
        logger.info(
            "Issue #%s resume marker requests restart; re-running investigation",
            issue_number,
        )
        adapter.remove_label(issue_number, NEEDS_HUMAN_LABEL)
        adapter.post_comment(
            issue_number,
            "Previous run failed before saving full progress. "
            "Re-running investigation and posting a fresh proposal.",
        )
        return _handle_issue_opened(
            adapter,
            settings,
            issue_number,
            model,
            repos_json,
            reporter,
        )

    if state.investigation is None or state.approach is None:
        raise AgentError(
            f"Issue #{issue_number} state comment is missing investigation or "
            "selected approach required to resume implementation."
        )

    investigation = state.investigation
    approach = state.approach
    branch = state.branch
    _assert_repo_pipeline_available(adapter, issue_number)
    logger.info(
        "Resuming issue #%s from saved state on branch %r (approach=%r)",
        issue_number,
        branch,
        approach.name,
    )

    adapter.remove_label(issue_number, NEEDS_HUMAN_LABEL)
    adapter.add_label(issue_number, IN_PROGRESS_LABEL)
    resume_note = (
        f":repeat: Resuming work on issue #{issue_number}, continuing from the "
        f"existing branch `{branch}`."
    )
    if state.subtask_plan is not None and state.subtask_index < len(
        state.subtask_plan.subtasks
    ):
        next_subtask = state.subtask_plan.subtasks[state.subtask_index]
        resume_note += (
            f"\n\nContinuing subtask **{state.subtask_index + 1}/"
            f"{len(state.subtask_plan.subtasks)}**: {next_subtask.name}."
        )
        if state.checkpoint_completed is not None:
            total_checkpoints = (
                len(state.subtask_plan.subtasks) * DEFAULT_MAX_ROUNDS_PER_SUBTASK * 2
            )
            resume_note += (
                f"\n\nProgress checkpoint **{state.checkpoint_completed}/"
                f"{total_checkpoints}** from the previous run."
            )
        if state.stall_findings:
            resume_note += (
                f"\n\nCarrying forward **{len(state.stall_findings)}** reviewer "
                "finding(s) from the last attempt — the agent will continue fixing "
                "those before moving on."
            )
    adapter.post_comment(issue_number, resume_note)

    with tempfile.TemporaryDirectory(prefix="agentic-flow-resume-") as tmp:
        agents = _build_pipeline_agents(
            settings=settings,
            adapter=adapter,
            model=model,
            tmp_dir=tmp,
            repos_json=repos_json,
        )
        _post_approval_state(
            adapter,
            issue_number,
            investigation,
            approach,
            state.proposal,
            subtask_plan=state.subtask_plan,
            subtask_index=state.subtask_index,
            checkpoint_completed=state.checkpoint_completed,
        )
        with reporter.stage("RESUMING", 35):
            result = agents.orchestrator.run(
                issue,
                investigation,
                approach,
                settings.github_repo,
                issue_number,
                human_approval_text=comment_body,
                existing_branch=branch,
                proposal=state.proposal,
                subtask_plan=state.subtask_plan,
                start_subtask_index=state.subtask_index,
                checkpoint_completed=state.checkpoint_completed,
                resume_stall_findings=state.stall_findings,
                resume_stall_summary=state.stall_summary,
                resume_stall_files=state.stall_files,
                reporter=reporter,
            )

    return _finish_orchestrator_run(
        adapter=adapter,
        issue_number=issue_number,
        result=result,
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
    _assert_repo_pipeline_available(adapter, issue_number)
    adapter.remove_label(issue_number, AWAITING_APPROVAL_LABEL)
    adapter.add_label(issue_number, IN_PROGRESS_LABEL)

    comments = adapter.list_comments(issue_number)
    state_comment = find_latest_state_comment(comments)
    if state_comment is None:
        raise AgentError(
            f"Issue #{issue_number} has no {STATE_MARKER!r} comment; "
            "cannot load the investigation the human approved against."
        )
    state = parse_state_comment(str(state_comment["body"]))
    if state.investigation is None or state.proposal is None:
        raise AgentError(
            f"Issue #{issue_number} state comment is missing investigation or proposal."
        )

    investigation = state.investigation
    approach = resolve_approach(state.proposal.approaches, parsed.selected_approach)
    selected_name = approach.name

    adapter.post_comment(
        issue_number,
        (
            f"Understood — proceeding with **{selected_name}**.\n\n"
            "Breaking the work into small subtasks, then implementing and "
            "reviewing **one subtask at a time** before opening a pull request."
        ),
    )

    logger.info(
        "Running orchestrator for issue #%s with approved approach %r",
        issue_number,
        approach.name,
    )

    _post_approval_state(
        adapter,
        issue_number,
        investigation,
        approach,
        state.proposal,
    )

    result = agents.orchestrator.run(
        issue,
        investigation,
        approach,
        settings.github_repo,
        issue_number,
        human_approval_text=comment_body,
        proposal=state.proposal,
        reporter=reporter,
    )

    return _finish_orchestrator_run(
        adapter=adapter,
        issue_number=issue_number,
        result=result,
        reporter=reporter,
    )


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
    except Exception as exc:
        _post_unhandled_pipeline_failure(adapter, issue_number, exc, reporter)
        exit_code = 1
    finally:
        reporter.write_step_summary(exit_code)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
