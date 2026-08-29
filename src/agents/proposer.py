"""Proposer agent: turn an investigation into variable-length fix approaches."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from core.models import Approach, Investigation, Proposal

PROPOSER_SYSTEM_PROMPT: Final[str] = (
    "You are a senior engineer proposing solutions, based on the "
    "investigation you're given.\n\n"
    "Rules:\n"
    "- Decide how many genuinely distinct, viable approaches exist for "
    "THIS issue. Do not force a fixed number. If there is truly one "
    "reasonable way to fix it, output exactly one. Never invent a fake "
    "alternative just to look thorough.\n"
    "- If a fast/temporary mitigation exists alongside a permanent "
    "structural fix, include both, clearly labeled in your own words, "
    "with tradeoffs: speed vs durability vs risk vs scope.\n"
    "- For each approach, explain what it does, why it addresses the "
    "root cause (or why it's only a mitigation), its risk level, and "
    "rough scope.\n"
    "- If the investigation raised open questions that materially change "
    "which approach is best, surface that ambiguity honestly in the "
    "approach description rather than silently picking one assumption.\n\n"
    "Respond with ONLY a JSON object matching this shape:\n"
    '{"approaches": [{"name": str, "nature": str, "description": str, '
    '"why_it_works": str, "risk": str, "tradeoffs": str, '
    '"estimated_scope": str}]}'
)


class ProposerAgent(BaseAgent):
    """Propose fix approaches from a completed :class:`Investigation`.

    This agent does not use tools; it reasons only from investigation
    findings already gathered by :class:`InvestigatorAgent`.
    """

    def __init__(self, client: Anthropic, model: str) -> None:
        """Create a proposer bound to an Anthropic client and model id.

        Args:
            client: Authenticated Anthropic SDK client.
            model: Model identifier passed to ``messages.create``.
        """
        super().__init__(client, model)
        self.system_prompt = PROPOSER_SYSTEM_PROMPT
        self.tool_definitions = []

    def propose(self, investigation: Investigation) -> Proposal:
        """Build approaches from ``investigation`` and return a :class:`Proposal`.

        Args:
            investigation: Validated findings from the investigator agent.

        Returns:
            A validated proposal with one or more approaches.

        Raises:
            AgentError: If Claude fails or the structured output is invalid.
        """
        user_message = self._build_user_message(investigation)
        return self.run(user_message, Proposal)

    def format_as_comment(self, proposal: Proposal) -> str:
        """Render ``proposal`` as Markdown suitable for a GitHub issue comment.

        Args:
            proposal: Validated proposal to format for human review.

        Returns:
            Markdown text with a numbered list of approaches and a closing
            prompt for the human to pick an option or approve a single one.
        """
        lines: list[str] = ["## Proposed approaches", ""]

        for index, approach in enumerate(proposal.approaches, start=1):
            lines.extend(_format_approach(index, approach))
            lines.append("")

        if len(proposal.approaches) == 1:
            closing = (
                "Reply with **approved** (or describe any changes you want) "
                "and we will proceed with this approach."
            )
        else:
            closing = (
                "Reply with the **number or name** of the approach you want "
                "us to proceed with, or describe a variation you prefer."
            )

        lines.extend(["---", closing])
        return "\n".join(lines).strip()

    def _build_user_message(self, investigation: Investigation) -> str:
        """Serialize investigation fields into a readable user turn."""
        if investigation.evidence:
            evidence_block = "\n".join(f"- {item}" for item in investigation.evidence)
        else:
            evidence_block = "(none)"

        if investigation.relevant_files:
            files_block = "\n".join(
                f"- {item.repo}:{item.path}\n  {item.reason}"
                for item in investigation.relevant_files
            )
        else:
            files_block = "(none)"

        if investigation.open_questions:
            questions_block = "\n".join(
                f"- {question}" for question in investigation.open_questions
            )
        else:
            questions_block = "(none)"

        return (
            "Propose fix/build approaches based on this investigation.\n\n"
            f"Issue nature:\n{investigation.issue_nature}\n\n"
            f"Root cause:\n{investigation.root_cause}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Relevant files:\n{files_block}\n\n"
            f"Confidence: {investigation.confidence}\n\n"
            f"Open questions:\n{questions_block}\n\n"
            "Respond with only the JSON object specified in your instructions."
        )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Refuse tool calls; this agent does not expose any tools."""
        _ = tool_input
        return json.dumps(
            {"error": f"ProposerAgent has no tools; got {tool_name!r}"},
            ensure_ascii=False,
        )


def _format_approach(index: int, approach: Approach) -> list[str]:
    """Return Markdown lines for one numbered approach."""
    return [
        f"### {index}. {approach.name}",
        "",
        f"**Nature:** {approach.nature}",
        "",
        f"**Description:** {approach.description}",
        "",
        f"**Why it works:** {approach.why_it_works}",
        "",
        f"**Risk:** {approach.risk}",
        "",
        f"**Tradeoffs:** {approach.tradeoffs}",
        "",
        f"**Estimated scope:** {approach.estimated_scope}",
    ]
