"""Task decomposer agent: split an approved approach into small ordered subtasks."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.models import Approach, Investigation, SubtaskPlan

TASK_DECOMPOSER_SYSTEM_PROMPT: Final[str] = (
    "You are a senior engineer breaking an already-approved fix approach into "
    "small, ordered implementation subtasks.\n\n"
    "Rules:\n"
    "- Each subtask must be completable in ONE focused implementer session: "
    "roughly 1–3 files or one vertical slice (e.g. 'backend register route' "
    "then 'frontend login form'), NOT the entire feature at once.\n"
    "- Order subtasks so later steps can build on earlier ones (migrations/schema "
    "before API before UI, shared types before consumers, etc.).\n"
    "- Together, all subtasks must fully cover the approved approach — no gaps, "
    "no duplicate work.\n"
    "- Use 1 subtask when the approved approach is already small (~one file or "
    "a trivial change). Use 2–5 subtasks for larger features. Never exceed 5.\n"
    "- Keep each description concrete and scoped. scope: a short phrase "
    "(e.g. '~2 files, backend routes only').\n\n"
    "Respond with ONLY JSON: "
    '{"subtasks": [{"name": str, "description": str, "scope": str, "order": int}]}'
)


class TaskDecomposerAgent(BaseAgent):
    """Decompose an approved approach into ordered :class:`Subtask` steps."""

    def __init__(self, client: Anthropic, model: str, settings: Settings) -> None:
        """Create a task decomposer bound to an Anthropic client and model id."""
        super().__init__(client, model, settings, "task_decomposer")
        self.system_prompt = TASK_DECOMPOSER_SYSTEM_PROMPT
        self.tool_definitions = []

    def decompose(
        self,
        issue_title: str,
        issue_body: str,
        investigation: Investigation,
        approach: Approach,
        *,
        human_approval_text: str | None = None,
    ) -> SubtaskPlan:
        """Return an ordered plan of subtasks for ``approach``."""
        user_message = self._build_user_message(
            issue_title=issue_title,
            issue_body=issue_body,
            investigation=investigation,
            approach=approach,
            human_approval_text=human_approval_text,
        )
        return self.run(user_message, SubtaskPlan)

    def _build_user_message(
        self,
        *,
        issue_title: str,
        issue_body: str,
        investigation: Investigation,
        approach: Approach,
        human_approval_text: str | None,
    ) -> str:
        body = issue_body.strip() if issue_body else "(empty)"
        approval = (
            human_approval_text.strip()
            if human_approval_text and human_approval_text.strip()
            else "(none)"
        )
        return (
            "Break the approved approach into ordered implementation subtasks.\n\n"
            f"Issue: {issue_title}\n\n"
            f"Issue body:\n{body}\n\n"
            f"Human approval / extra requirements:\n{approval}\n\n"
            f"Issue nature:\n{investigation.issue_nature}\n\n"
            f"Root cause:\n{investigation.root_cause}\n\n"
            "Approved approach:\n"
            f"- Name: {approach.name}\n"
            f"- Nature: {approach.nature}\n"
            f"- Description: {approach.description}\n"
            f"- Why it works: {approach.why_it_works}\n"
            f"- Estimated scope: {approach.estimated_scope}\n\n"
            "Respond with only the JSON object specified in your instructions."
        )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Refuse tool calls; this agent does not expose any tools."""
        _ = tool_input
        return json.dumps(
            {"error": f"TaskDecomposerAgent has no tools; got {tool_name!r}"},
            ensure_ascii=False,
        )
