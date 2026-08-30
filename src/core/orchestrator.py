"""Orchestrate decompose → implement → review loops and pull-request creation."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from adapters.base import AdapterError, IssueProviderAdapter
from agents.implementer import ImplementerAgent
from agents.reviewer import ReviewerAgent
from agents.task_decomposer import TaskDecomposerAgent
from core.exceptions import AgentError
from core.models import (
    Approach,
    ImplementationResult,
    Investigation,
    Proposal,
    ReviewResult,
    Subtask,
    SubtaskPlan,
)
from core.pipeline_state import format_state_comment, format_subtask_plan_comment
from utils.logger import RunReporter, get_logger

IN_PROGRESS_LABEL = "agent:in-progress"
NEEDS_HUMAN_LABEL = "agent:needs-human"
DONE_LABEL = "agent:done"
DEFAULT_MAX_ROUNDS = 6
DEFAULT_MAX_ROUNDS_PER_SUBTASK = 3


@dataclass
class OrchestratorResult:
    """Outcome of :meth:`ImplementationOrchestrator.run`."""

    passed: bool
    review_result: ReviewResult | None = None
    implementation_result: ImplementationResult | None = None
    pr_url: str | None = None
    diagnostic_comment: str | None = None
    round_history: list[dict[str, Any]] = field(default_factory=list)
    subtask_plan: SubtaskPlan | None = None


@dataclass
class _SubtaskOutcome:
    passed: bool
    implementation_result: ImplementationResult | None = None
    review_result: ReviewResult | None = None
    failure_reason: str | None = None


class ImplementationOrchestrator:
    """Decompose approved work into subtasks, then implement/review sequentially."""

    def __init__(
        self,
        adapter: IssueProviderAdapter,
        implementer: ImplementerAgent,
        reviewer: ReviewerAgent,
        decomposer: TaskDecomposerAgent,
        *,
        max_rounds_per_subtask: int = DEFAULT_MAX_ROUNDS_PER_SUBTASK,
        logger: logging.Logger | None = None,
    ) -> None:
        """Bind orchestration to adapter and agent instances."""
        self._adapter = adapter
        self._implementer = implementer
        self._reviewer = reviewer
        self._decomposer = decomposer
        self._max_rounds_per_subtask = max(1, max_rounds_per_subtask)
        self._logger = logger or get_logger(__name__)

    def run(
        self,
        issue: dict[str, Any],
        investigation: Investigation,
        approach: Approach,
        primary_repo: str,
        issue_number: int,
        *,
        human_approval_text: str | None = None,
        existing_branch: str | None = None,
        proposal: Proposal | None = None,
        subtask_plan: SubtaskPlan | None = None,
        start_subtask_index: int = 0,
        reporter: RunReporter | None = None,
    ) -> OrchestratorResult:
        """Decompose, implement each subtask, review, and open a PR when complete."""
        history: list[dict[str, Any]] = []
        branch_name = existing_branch or f"agent/fix-issue-{issue_number}"

        decompose_ctx = (
            reporter.stage("PLANNING", 50) if reporter is not None else nullcontext()
        )
        with decompose_ctx:
            if subtask_plan is None:
                self._logger.info(
                    "Decomposing approved approach into subtasks for issue #%s",
                    issue_number,
                )
                subtask_plan = self._decomposer.decompose(
                    str(issue.get("title") or ""),
                    str(issue.get("body") or ""),
                    investigation,
                    approach,
                    human_approval_text=human_approval_text,
                )
                self._adapter.post_comment(
                    issue_number,
                    format_subtask_plan_comment(subtask_plan, approach.name),
                )
                self._persist_state(
                    issue_number=issue_number,
                    branch=branch_name,
                    investigation=investigation,
                    approach=approach,
                    proposal=proposal,
                    subtask_plan=subtask_plan,
                    subtask_index=0,
                )

        total_subtasks = len(subtask_plan.subtasks)
        combined_files: list[str] = []
        combined_summaries: list[str] = []
        last_review: ReviewResult | None = None
        last_implementation: ImplementationResult | None = None

        for subtask_index in range(max(0, start_subtask_index), total_subtasks):
            subtask = subtask_plan.subtasks[subtask_index]
            is_final = subtask_index == total_subtasks - 1
            self._logger.info(
                "Orchestrator subtask %d/%d for issue #%s: %r",
                subtask_index + 1,
                total_subtasks,
                issue_number,
                subtask.name,
            )
            subtask_pct = 55 + int((subtask_index / max(total_subtasks, 1)) * 35)
            subtask_ctx = (
                reporter.stage("SUBTASK", subtask_pct)
                if reporter is not None
                else nullcontext()
            )

            with subtask_ctx:
                outcome = self._run_subtask(
                    issue=issue,
                    investigation=investigation,
                    approach=approach,
                    primary_repo=primary_repo,
                    issue_number=issue_number,
                    subtask=subtask,
                    subtask_index=subtask_index + 1,
                    subtask_total=total_subtasks,
                    is_final_subtask=is_final,
                    human_approval_text=human_approval_text,
                    existing_branch=branch_name,
                    history=history,
                    reporter=reporter,
                )

            last_review = outcome.review_result
            last_implementation = outcome.implementation_result

            if not outcome.passed:
                diagnostic = self._build_diagnostic_comment(
                    issue=issue,
                    approach=approach,
                    history=history,
                    reason=outcome.failure_reason or "Subtask implementation stalled.",
                    last_review=last_review,
                    subtask_plan=subtask_plan,
                    subtask_index=subtask_index,
                )
                self._mark_needs_human(issue_number, diagnostic)
                self._persist_state(
                    issue_number=issue_number,
                    branch=branch_name,
                    investigation=investigation,
                    approach=approach,
                    proposal=proposal,
                    subtask_plan=subtask_plan,
                    subtask_index=subtask_index,
                )
                return OrchestratorResult(
                    passed=False,
                    review_result=last_review,
                    implementation_result=last_implementation,
                    diagnostic_comment=diagnostic,
                    round_history=history,
                    subtask_plan=subtask_plan,
                )

            if outcome.implementation_result is not None:
                for path in outcome.implementation_result.files_changed:
                    if path not in combined_files:
                        combined_files.append(path)
                combined_summaries.append(
                    f"**{subtask.name}:** {outcome.implementation_result.summary}"
                )

            self._persist_state(
                issue_number=issue_number,
                branch=branch_name,
                investigation=investigation,
                approach=approach,
                proposal=proposal,
                subtask_plan=subtask_plan,
                subtask_index=subtask_index + 1,
            )

        if last_implementation is None or last_review is None:
            raise AgentError("Orchestrator completed subtasks without implementation output.")

        combined_implementation = last_implementation.model_copy(
            update={
                "branch_name": branch_name,
                "files_changed": combined_files,
                "summary": "\n\n".join(combined_summaries),
            }
        )
        pr_payload = self.format_pr(
            issue,
            investigation,
            approach,
            combined_implementation,
            last_review,
            subtask_plan=subtask_plan,
        )
        try:
            pr = self._adapter.open_pr(
                pr_payload["title"],
                pr_payload["body"],
                branch_name,
            )
        except AdapterError as exc:
            diagnostic = self._build_diagnostic_comment(
                issue=issue,
                approach=approach,
                history=history,
                reason=f"All subtasks passed review but opening the PR failed: {exc}",
                last_review=last_review,
                subtask_plan=subtask_plan,
            )
            self._mark_needs_human(issue_number, diagnostic)
            return OrchestratorResult(
                passed=False,
                review_result=last_review,
                implementation_result=combined_implementation,
                diagnostic_comment=diagnostic,
                round_history=history,
                subtask_plan=subtask_plan,
            )

        pr_url = str(pr.get("url", ""))
        self._logger.info("Opened PR for issue #%s: %s", issue_number, pr_url)
        return OrchestratorResult(
            passed=True,
            review_result=last_review,
            implementation_result=combined_implementation,
            pr_url=pr_url or None,
            round_history=history,
            subtask_plan=subtask_plan,
        )

    def _run_subtask(
        self,
        *,
        issue: dict[str, Any],
        investigation: Investigation,
        approach: Approach,
        primary_repo: str,
        issue_number: int,
        subtask: Subtask,
        subtask_index: int,
        subtask_total: int,
        is_final_subtask: bool,
        human_approval_text: str | None,
        existing_branch: str,
        history: list[dict[str, Any]],
        reporter: RunReporter | None,
    ) -> _SubtaskOutcome:
        """Run implement/review rounds for a single subtask."""
        last_review: ReviewResult | None = None
        last_implementation: ImplementationResult | None = None
        last_round_failure_note: str | None = None

        for round_index in range(1, self._max_rounds_per_subtask + 1):
            self._logger.info(
                "Subtask %d/%d round %d/%d for issue #%s",
                subtask_index,
                subtask_total,
                round_index,
                self._max_rounds_per_subtask,
                issue_number,
            )
            implement_ctx = (
                reporter.stage("IMPLEMENTING", 60 + round_index * 3)
                if reporter is not None
                else nullcontext()
            )
            review_ctx = (
                reporter.stage("REVIEWING", 72 + round_index * 3)
                if reporter is not None
                else nullcontext()
            )

            review_findings: list[str] | None = None
            if last_review is not None and last_review.findings:
                review_findings = list(last_review.findings)

            round_failure_note = last_round_failure_note
            last_round_failure_note = None

            try:
                with implement_ctx:
                    implementation = self._implementer.implement(
                        issue,
                        investigation,
                        approach,
                        primary_repo,
                        human_approval_text=human_approval_text,
                        review_findings=review_findings,
                        attempt_failure_note=round_failure_note,
                        existing_branch=existing_branch,
                        subtask=subtask,
                        subtask_index=subtask_index,
                        subtask_total=subtask_total,
                    )
            except Exception as exc:
                short_error = _short_error(exc)
                self._logger.warning(
                    "Implementer failed on subtask %d round %d for issue #%s: %s",
                    subtask_index,
                    round_index,
                    issue_number,
                    short_error,
                )
                last_round_failure_note = (
                    f"Your previous attempt failed with: {short_error}. Try again."
                )
                history.append(
                    {
                        "subtask": subtask.name,
                        "subtask_index": subtask_index,
                        "round": round_index,
                        "stage": "implement_error",
                        "error": short_error,
                    }
                )
                continue

            try:
                with review_ctx:
                    review = self._reviewer.review(
                        issue,
                        investigation,
                        implementation,
                        primary_repo,
                        human_approval_text=human_approval_text,
                        subtask=subtask,
                        subtask_index=subtask_index,
                        subtask_total=subtask_total,
                        is_final_subtask=is_final_subtask,
                    )
            except Exception as exc:
                short_error = _short_error(exc)
                self._logger.warning(
                    "Reviewer failed on subtask %d round %d for issue #%s: %s",
                    subtask_index,
                    round_index,
                    issue_number,
                    short_error,
                )
                last_round_failure_note = (
                    f"Your previous attempt failed with: {short_error}. Try again."
                )
                history.append(
                    {
                        "subtask": subtask.name,
                        "subtask_index": subtask_index,
                        "round": round_index,
                        "stage": "review_error",
                        "error": short_error,
                        "implementation_summary": implementation.summary,
                    }
                )
                last_implementation = implementation
                continue

            last_implementation = implementation
            last_review = review
            history.append(
                {
                    "subtask": subtask.name,
                    "subtask_index": subtask_index,
                    "round": round_index,
                    "approved": review.approved,
                    "making_progress": review.making_progress,
                    "implementation_summary": implementation.summary,
                    "review_summary": review.summary,
                    "findings": list(review.findings),
                    "test_output_summary": review.test_output_summary,
                }
            )

            if review.approved:
                return _SubtaskOutcome(
                    passed=True,
                    implementation_result=implementation,
                    review_result=review,
                )

            if not review.making_progress:
                return _SubtaskOutcome(
                    passed=False,
                    implementation_result=implementation,
                    review_result=review,
                    failure_reason=(
                        f"Subtask {subtask_index}/{subtask_total} ({subtask.name!r}) "
                        "stalled: reviewer set making_progress=false."
                    ),
                )

        return _SubtaskOutcome(
            passed=False,
            implementation_result=last_implementation,
            review_result=last_review,
            failure_reason=(
                f"Subtask {subtask_index}/{subtask_total} ({subtask.name!r}) did not "
                f"pass review after {self._max_rounds_per_subtask} round(s)."
            ),
        )

    def _persist_state(
        self,
        *,
        issue_number: int,
        branch: str,
        investigation: Investigation,
        approach: Approach,
        proposal: Proposal | None,
        subtask_plan: SubtaskPlan,
        subtask_index: int,
    ) -> None:
        """Post hidden state so resume can continue from the next subtask."""
        self._adapter.post_comment(
            issue_number,
            format_state_comment(
                branch,
                investigation=investigation,
                approach=approach,
                proposal=proposal,
                subtask_plan=subtask_plan,
                subtask_index=subtask_index,
            ),
        )

    def format_pr(
        self,
        issue: dict[str, Any],
        investigation: Investigation,
        approach: Approach,
        implementation_result: ImplementationResult,
        review_result: ReviewResult,
        *,
        subtask_plan: SubtaskPlan | None = None,
    ) -> dict[str, str]:
        """Build PR title and Markdown body for an approved change."""
        issue_title = str(issue.get("title") or "Untitled issue")
        issue_body = str(issue.get("body") or "").strip() or "(empty)"
        issue_number = issue.get("number", "?")

        files_block = "\n".join(f"- `{path}`" for path in implementation_result.files_changed)
        if not files_block:
            files_block = "- (none listed)"

        layers_checked = _layers_checked_summary(review_result)
        test_output_summary = _test_output_summary(review_result)
        ui_section = _verification_section(
            "UI verification",
            review_result.ui_verification,
            executed_label="Playwright / live UI flow",
        )
        db_section = _verification_section(
            "Database verification",
            review_result.db_verification,
            executed_label="Independent DB row check",
        )

        open_questions = investigation.open_questions
        if open_questions:
            questions_block = "\n".join(f"- {item}" for item in open_questions)
        else:
            questions_block = "- None recorded."

        subtasks_block = ""
        if subtask_plan is not None:
            lines = [
                f"{index}. **{item.name}** — {item.scope}"
                for index, item in enumerate(subtask_plan.subtasks, start=1)
            ]
            subtasks_block = "## Subtasks completed\n\n" + "\n".join(lines) + "\n\n"

        body = (
            "## Original issue\n\n"
            f"Fixes #{issue_number}: **{issue_title}**\n\n"
            f"{issue_body}\n\n"
            "## Root cause\n\n"
            f"{investigation.root_cause}\n\n"
            "## Chosen approach\n\n"
            f"**{approach.name}** ({approach.nature})\n\n"
            f"{approach.description}\n\n"
            "**Why this approach:** "
            f"{approach.why_it_works}\n\n"
            f"{subtasks_block}"
            "## What changed\n\n"
            f"{implementation_result.summary}\n\n"
            "**Files touched:**\n"
            f"{files_block}\n\n"
            "## Testing\n\n"
            f"**Layers checked:** {layers_checked}\n\n"
            f"**Automated checks:** {test_output_summary}\n\n"
            f"**Review summary:** {review_result.summary}\n\n"
            f"{ui_section}\n\n"
            f"{db_section}\n\n"
            "## Open questions for humans\n\n"
            f"{questions_block}\n"
        )
        return {
            "title": f"Fix: {issue_title}",
            "body": body.strip(),
        }

    def _mark_needs_human(self, issue_number: int, diagnostic: str) -> None:
        """Move the issue to a stalled state and post the diagnostic comment."""
        if self._adapter.has_label(issue_number, IN_PROGRESS_LABEL):
            self._adapter.remove_label(issue_number, IN_PROGRESS_LABEL)
        self._adapter.add_label(issue_number, NEEDS_HUMAN_LABEL)
        self._adapter.post_comment(issue_number, diagnostic)

    def _build_diagnostic_comment(
        self,
        *,
        issue: dict[str, Any],
        approach: Approach,
        history: list[dict[str, Any]],
        reason: str,
        last_review: ReviewResult | None = None,
        subtask_plan: SubtaskPlan | None = None,
        subtask_index: int | None = None,
    ) -> str:
        """Format a GitHub comment when orchestration stalls."""
        lines = [
            "## Agent stalled — needs human help",
            "",
            f"Issue: **{issue.get('title', 'Untitled issue')}**",
            f"Approved approach: **{approach.name}**",
            "",
            f"**Why it stopped:** {reason}",
        ]
        if subtask_plan is not None and subtask_index is not None:
            total = len(subtask_plan.subtasks)
            current = subtask_plan.subtasks[subtask_index]
            lines.extend(
                [
                    "",
                    f"**Subtask progress:** {subtask_index}/{total} completed before stop — "
                    f"blocked on **{current.name}**",
                ]
            )
        if last_review is not None and last_review.test_output_summary.strip():
            lines.extend(
                [
                    "",
                    f"**Automated checks:** {last_review.test_output_summary}",
                ]
            )
        lines.extend(["", "### Attempts"])
        if not history:
            lines.append("- No rounds completed.")
        else:
            for entry in history:
                stage = entry.get("stage")
                prefix = ""
                if entry.get("subtask"):
                    prefix = f"[{entry['subtask']}] "
                if stage in {"implement_error", "review_error", "agent_error"}:
                    lines.append(
                        f"- {prefix}Round {entry['round']}: {stage.replace('_', ' ')} — "
                        f"{entry.get('error')}"
                    )
                    continue
                status = "approved" if entry.get("approved") else "not approved"
                lines.append(
                    f"- {prefix}Round {entry['round']}: {status}; "
                    f"implementation={entry.get('implementation_summary')!r}; "
                    f"review={entry.get('review_summary')!r}"
                )
                findings = entry.get("findings") or []
                for finding in findings:
                    lines.append(f"  - {finding}")
        lines.extend(
            [
                "",
                "The issue has been labeled **`agent:needs-human`**. "
                "Comment **continue** to resume from the saved subtask progress, "
                "or inspect the branch and take over manually.",
            ]
        )
        return "\n".join(lines)


def resolve_approach(proposal_approaches: list[Approach], selected: str | None) -> Approach:
    """Match a human-selected approach name/number to a :class:`Approach`."""
    if not proposal_approaches:
        raise AgentError("Cannot resolve approach from an empty proposal.")
    if not selected or not selected.strip():
        return proposal_approaches[0]

    needle = selected.strip().lower()
    for approach in proposal_approaches:
        if approach.name.lower() == needle:
            return approach
        if needle in approach.name.lower() or approach.name.lower() in needle:
            return approach

    for index, approach in enumerate(proposal_approaches, start=1):
        if needle in {str(index), f"approach {index}", f"#{index}"}:
            return approach

    return proposal_approaches[0]


def _short_error(exc: BaseException) -> str:
    """Return a single-line error summary safe to pass to the next agent turn."""
    message = str(exc).strip().replace("\r", " ")
    first_line = message.split("\n", 1)[0].strip()
    if first_line:
        return first_line[:300]
    return type(exc).__name__


def _layers_checked_summary(review_result: ReviewResult) -> str:
    layers = review_result.layers_checked or review_result.layers_detected
    if not layers:
        return "No layer flags recorded."
    run_layers = [name for name, value in sorted(layers.items()) if value]
    if not run_layers:
        return "none (no automated checks ran for this change)"
    return ", ".join(run_layers)


def _test_output_summary(review_result: ReviewResult) -> str:
    if review_result.test_output_summary.strip():
        return review_result.test_output_summary.strip()
    parts = [review_result.summary]
    if review_result.findings:
        parts.append("Findings: " + "; ".join(review_result.findings))
    return " ".join(parts)


def _verification_section(
    heading: str,
    payload: dict[str, Any] | None,
    *,
    executed_label: str,
) -> str:
    if payload is None:
        return f"### {heading}\n\nNot run for this change (logical/code review only)."
    if payload.get("skipped"):
        return (
            f"### {heading}\n\n"
            f"**Status:** skipped\n"
            f"**Details:** {payload.get('details', '(no details)')}"
        )
    passed = payload.get("ui_passed", payload.get("db_passed", payload.get("passed")))
    details = payload.get("details", "(no details provided)")
    marker = payload.get("test_marker")
    status = "passed" if passed else "failed"
    lines = [
        f"### {heading}",
        "",
        f"**Status:** {status}",
        f"**What ran:** {executed_label}",
        f"**Details:** {details}",
    ]
    if marker:
        lines.append(f"**Test marker:** `{marker}`")
    return "\n".join(lines)
