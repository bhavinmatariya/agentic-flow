"""Hidden GitHub issue comment state for resume and subtask progress."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from core.exceptions import AgentError
from core.models import Approach, Investigation, Proposal, SubtaskPlan, MAX_SUBTASKS

STATE_MARKER = "agentic-flow:state"
RESTART_INVESTIGATION_MODE = "restart_investigation"
_STATE_COMMENT_PATTERN = re.compile(
    r"<!--\s*agentic-flow:state\s*\n(.*?)\n-->",
    re.DOTALL,
)


@dataclass
class PipelineState:
    """Parsed pipeline payload from a state comment."""

    branch: str
    investigation: Investigation | None = None
    proposal: Proposal | None = None
    approach: Approach | None = None
    subtask_plan: SubtaskPlan | None = None
    subtask_index: int = 0
    checkpoint_completed: int | None = None
    resume_mode: str | None = None


def format_state_comment(
    branch: str,
    *,
    investigation: Investigation | None = None,
    proposal: Proposal | None = None,
    approach: Approach | None = None,
    subtask_plan: SubtaskPlan | None = None,
    subtask_index: int | None = None,
    checkpoint_completed: int | None = None,
    resume_mode: str | None = None,
) -> str:
    """Build a hidden state payload comment."""
    payload: dict[str, Any] = {"branch": branch}
    if resume_mode is not None:
        payload["resume_mode"] = resume_mode
    if investigation is not None:
        payload["investigation"] = investigation.model_dump()
    if proposal is not None:
        payload["proposal"] = proposal.model_dump()
    if approach is not None:
        payload["approach"] = approach.model_dump()
    if subtask_plan is not None:
        payload["subtask_plan"] = subtask_plan.model_dump()
    if subtask_index is not None:
        payload["subtask_index"] = subtask_index
    if checkpoint_completed is not None:
        payload["checkpoint_completed"] = max(0, int(checkpoint_completed))
    return f"<!-- {STATE_MARKER}\n{json.dumps(payload, ensure_ascii=False)}\n-->"


def parse_state_comment(body: str) -> PipelineState:
    """Parse pipeline state from a state comment body."""
    match = _STATE_COMMENT_PATTERN.search(body)
    if match is None:
        raise AgentError(
            f"State comment on issue is missing a valid {STATE_MARKER!r} payload."
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise AgentError(f"State comment JSON is invalid: {exc}") from exc

    if not isinstance(payload, dict):
        raise AgentError("State comment payload must be a JSON object.")

    branch = str(payload.get("branch") or "").strip()
    if not branch:
        raise AgentError("State comment payload is missing a non-empty branch name.")

    resume_mode = payload.get("resume_mode")
    if resume_mode == RESTART_INVESTIGATION_MODE:
        return PipelineState(branch=branch, resume_mode=RESTART_INVESTIGATION_MODE)

    investigation: Investigation | None = None
    proposal: Proposal | None = None
    approach: Approach | None = None
    subtask_plan: SubtaskPlan | None = None
    subtask_index = int(payload.get("subtask_index") or 0)
    checkpoint_completed: int | None = None
    if "checkpoint_completed" in payload:
        checkpoint_completed = max(0, int(payload.get("checkpoint_completed") or 0))
    try:
        if payload.get("investigation") is not None:
            investigation = Investigation.model_validate(payload.get("investigation"))
        if payload.get("proposal") is not None:
            proposal = Proposal.model_validate(payload.get("proposal"))
        if payload.get("approach") is not None:
            approach = Approach.model_validate(payload.get("approach"))
        if payload.get("subtask_plan") is not None:
            subtask_plan = SubtaskPlan.model_validate(payload.get("subtask_plan"))
    except ValidationError as exc:
        raise AgentError(
            f"State comment payload failed validation: {exc} "
            f"(stored subtask plans support up to {MAX_SUBTASKS} items)."
        ) from exc

    return PipelineState(
        branch=branch,
        investigation=investigation,
        proposal=proposal,
        approach=approach,
        subtask_plan=subtask_plan,
        subtask_index=max(0, subtask_index),
        checkpoint_completed=checkpoint_completed,
        resume_mode=str(resume_mode) if resume_mode else None,
    )


def find_latest_state_comment(
    comments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the most recent comment containing an agentic-flow state marker."""
    matches = [
        comment
        for comment in comments
        if isinstance(comment.get("body"), str) and STATE_MARKER in comment["body"]
    ]
    if not matches:
        return None
    return max(matches, key=lambda comment: str(comment.get("created_at", "")))


def format_subtask_plan_comment(plan: SubtaskPlan, approach_name: str) -> str:
    """Render a human-visible subtask breakdown comment."""
    lines = [
        "## Implementation plan — subtasks",
        "",
        f"Approved approach: **{approach_name}**",
        "",
        "Work will proceed **one subtask at a time** (fresh implement/review "
        "per step) to stay within token limits.",
        "",
    ]
    for index, subtask in enumerate(plan.subtasks, start=1):
        lines.extend(
            [
                f"### {index}. {subtask.name}",
                "",
                subtask.description,
                "",
                f"**Scope:** {subtask.scope}",
                "",
            ]
        )
    return "\n".join(lines).strip()
