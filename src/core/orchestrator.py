"""Orchestrate implement → review loops and pull-request creation."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from adapters.base import AdapterError, IssueProviderAdapter
from agents.implementer import ImplementerAgent
from agents.reviewer import ReviewerAgent
from core.exceptions import AgentError
from core.models import (
    Approach,
    ImplementationResult,
    Investigation,
    ReviewResult,
)
from utils.logger import RunReporter, get_logger

IN_PROGRESS_LABEL = "agent:in-progress"
NEEDS_HUMAN_LABEL = "agent:needs-human"
DONE_LABEL = "agent:done"
DEFAULT_MAX_ROUNDS = 6


@dataclass
class OrchestratorResult:
    """Outcome of :meth:`ImplementationOrchestrator.run`."""

    passed: bool
    review_result: ReviewResult | None = None
    implementation_result: ImplementationResult | None = None
    pr_url: str | None = None
    diagnostic_comment: str | None = None
    round_history: list[dict[str, Any]] = field(default_factory=list)


class ImplementationOrchestrator:
    """Run implement/review rounds, then open a PR or post a stall diagnostic."""

    def __init__(
        self,
        adapter: IssueProviderAdapter,
        implementer: ImplementerAgent,
        reviewer: ReviewerAgent,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        logger: logging.Logger | None = None,
    ) -> None:
        """Bind orchestration to adapter and agent instances."""
        self._adapter = adapter
        self._implementer = implementer
        self._reviewer = reviewer
        self._max_rounds = max(1, max_rounds)
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
        reporter: RunReporter | None = None,
    ) -> OrchestratorResult:
        """Implement, review, and open a PR when review approves the change."""
        history: list[dict[str, Any]] = []
        last_review: ReviewResult | None = None
        last_implementation: ImplementationResult | None = None
        last_round_failure_note: str | None = None

        for round_index in range(1, self._max_rounds + 1):
            # Each implement()/review() call starts a fresh BaseAgent.run()
            # conversation (messages=[]). Cross-round context is limited to the
            # approved approach plus the prior round's findings/failure note.
            self._logger.info(
                "Orchestrator round %d/%d for issue #%s",
                round_index,
                self._max_rounds,
                issue_number,
            )
            implement_pct = min(60 + (round_index - 1) * 5, 78)
            review_pct = min(78 + (round_index - 1) * 5, 92)
            implement_ctx = (
                reporter.stage("IMPLEMENTING", implement_pct)
                if reporter is not None
                else nullcontext()
            )
            review_ctx = (
                reporter.stage("REVIEWING", review_pct)
                if reporter is not None
                else nullcontext()
            )

            review_findings: list[str] | None = None
            if last_review is not None and last_review.findings:
                review_findings = list(last_review.findings)

            round_failure_note = last_round_failure_note
            last_round_failure_note = None

            implementation: ImplementationResult | None = None
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
                    )
            except Exception as exc:
                short_error = _short_error(exc)
                self._logger.warning(
                    "Implementer failed in round %d/%d for issue #%s: %s",
                    round_index,
                    self._max_rounds,
                    issue_number,
                    short_error,
                )
                last_round_failure_note = (
                    f"Your previous attempt failed with: {short_error}. "
                    "Try again, fixing that."
                )
                history.append(
                    {
                        "round": round_index,
                        "stage": "implement_error",
                        "error": short_error,
                    }
                )
                continue

            review: ReviewResult | None = None
            try:
                with review_ctx:
                    review = self._reviewer.review(
                        issue,
                        investigation,
                        implementation,
                        primary_repo,
                        human_approval_text=human_approval_text,
                    )
            except Exception as exc:
                short_error = _short_error(exc)
                self._logger.warning(
                    "Reviewer failed in round %d/%d for issue #%s: %s",
                    round_index,
                    self._max_rounds,
                    issue_number,
                    short_error,
                )
                last_round_failure_note = (
                    f"Your previous attempt failed with: {short_error}. "
                    "Try again, fixing that."
                )
                history.append(
                    {
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
                    "round": round_index,
                    "approved": review.approved,
                    "making_progress": review.making_progress,
                    "implementation_summary": implementation.summary,
                    "review_summary": review.summary,
                    "findings": list(review.findings),
                }
            )

            if review.approved:
                pr_payload = self.format_pr(
                    issue,
                    investigation,
                    approach,
                    implementation,
                    review,
                )
                try:
                    pr = self._adapter.open_pr(
                        pr_payload["title"],
                        pr_payload["body"],
                        implementation.branch_name,
                    )
                except AdapterError as exc:
                    diagnostic = self._build_diagnostic_comment(
                        issue=issue,
                        approach=approach,
                        history=history,
                        reason=f"Review passed but opening the PR failed: {exc}",
                    )
                    self._mark_needs_human(issue_number, diagnostic)
                    return OrchestratorResult(
                        passed=False,
                        review_result=review,
                        implementation_result=implementation,
                        diagnostic_comment=diagnostic,
                        round_history=history,
                    )

                pr_url = str(pr.get("url", ""))
                self._logger.info("Opened PR for issue #%s: %s", issue_number, pr_url)
                return OrchestratorResult(
                    passed=True,
                    review_result=review,
                    implementation_result=implementation,
                    pr_url=pr_url or None,
                    round_history=history,
                )

            if not review.making_progress:
                diagnostic = self._build_diagnostic_comment(
                    issue=issue,
                    approach=approach,
                    history=history,
                    reason=(
                        "The reviewer indicated that further automated rounds "
                        "are unlikely to help (making_progress=false)."
                    ),
                )
                self._mark_needs_human(issue_number, diagnostic)
                return OrchestratorResult(
                    passed=False,
                    review_result=review,
                    implementation_result=implementation,
                    diagnostic_comment=diagnostic,
                    round_history=history,
                )

        diagnostic = self._build_diagnostic_comment(
            issue=issue,
            approach=approach,
            history=history,
            reason=f"Review did not approve after {self._max_rounds} round(s).",
        )
        self._mark_needs_human(issue_number, diagnostic)
        return OrchestratorResult(
            passed=False,
            review_result=last_review,
            implementation_result=last_implementation,
            diagnostic_comment=diagnostic,
            round_history=history,
        )

    def format_pr(
        self,
        issue: dict[str, Any],
        investigation: Investigation,
        approach: Approach,
        implementation_result: ImplementationResult,
        review_result: ReviewResult,
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
            "## What changed\n\n"
            f"{implementation_result.summary}\n\n"
            "**Files touched:**\n"
            f"{files_block}\n\n"
            "## Testing\n\n"
            f"**Layers checked:** {layers_checked}\n\n"
            f"**Review summary:** {review_result.summary}\n\n"
            f"**Test output summary:** {test_output_summary}\n\n"
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
    ) -> str:
        """Format a GitHub comment when orchestration stalls."""
        lines = [
            "## Agent stalled — needs human help",
            "",
            f"Issue: **{issue.get('title', 'Untitled issue')}**",
            f"Approved approach: **{approach.name}**",
            "",
            f"**Why it stopped:** {reason}",
            "",
            "### Attempts",
        ]
        if not history:
            lines.append("- No rounds completed.")
        else:
            for entry in history:
                stage = entry.get("stage")
                if stage in {"implement_error", "review_error", "agent_error"}:
                    lines.append(
                        f"- Round {entry['round']}: {stage.replace('_', ' ')} — "
                        f"{entry.get('error')}"
                    )
                    continue
                status = "approved" if entry.get("approved") else "not approved"
                lines.append(
                    f"- Round {entry['round']}: {status}; "
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
                "Please inspect the branch/commits, adjust the plan, or take over manually.",
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
    layers = getattr(review_result, "layers_checked", None) or review_result.layers_detected
    if not layers:
        return "No layer flags recorded."
    return ", ".join(
        f"{name}={'yes' if bool(value) else 'no'}" for name, value in sorted(layers.items())
    )


def _test_output_summary(review_result: ReviewResult) -> str:
    explicit = getattr(review_result, "test_output_summary", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
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
