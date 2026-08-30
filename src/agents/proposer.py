"""Proposer agent: turn an investigation into variable-length fix approaches."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.models import Approach, Investigation, Proposal

PROPOSER_SYSTEM_PROMPT: Final[str] = (
    "You are a senior engineer proposing solutions, based on the "
    "investigation you're given.\n\n"
    "Rules:\n"
    "- Only propose multiple approaches when there is a genuine tradeoff "
    "between them — e.g., a fast temporary mitigation vs. a slower "
    "permanent fix, where both are legitimate given real constraints like "
    "urgency. Do NOT propose an approach that violates the codebase's own "
    "established conventions (e.g., a native browser alert() when the "
    "codebase already has a custom dialog system, or a raw SQL query "
    "pattern when the codebase consistently uses an ORM) merely to create "
    "a second option — an approach that breaks existing conventions is not "
    "a legitimate alternative, it is worse engineering, and should not be "
    "presented as a peer choice to a human. If only one approach is "
    "actually good and consistent with the codebase, present just that "
    "one. A second option must earn its place by being a real, defensible "
    "tradeoff — not exist just so there's something to choose between.\n"
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
    "approach description rather than silently picking one assumption.\n"
    "- Keep every field concise and scannable — this will be read as a "
    "GitHub comment, not a design document. description: max 2 sentences. "
    "why_it_works: max 2 sentences. tradeoffs: max 2 sentences, "
    "bullet-style if there are multiple distinct tradeoffs. "
    "estimated_scope: a short phrase, not a sentence (e.g. '~15 lines, "
    "one file'). Do not repeat information across fields — if something "
    "is already said in description, don't restate it in why_it_works.\n"
    "- For the risk field, use ONLY the word low, medium, or high (no emoji, "
    "no colon shortcodes like :large_green_circle:, no extra prose).\n\n"
    "Respond with ONLY a JSON object matching this shape:\n"
    '{"approaches": [{"name": str, "nature": str, "description": str, '
    '"why_it_works": str, "risk": str, "tradeoffs": str, '
    '"estimated_scope": str}]}'
)

PROPOSAL_SECTION_HEADER: Final[str] = "## Proposed approaches"


class ProposerAgent(BaseAgent):
    """Propose fix approaches from a completed :class:`Investigation`.

    This agent does not use tools; it reasons only from investigation
    findings already gathered by :class:`InvestigatorAgent`.
    """

    def __init__(self, client: Anthropic, model: str, settings: Settings) -> None:
        """Create a proposer bound to an Anthropic client and model id.

        Args:
            client: Authenticated Anthropic SDK client.
            model: Model identifier passed to the centralized Claude client.
            settings: Application settings with per-agent Claude defaults.
        """
        super().__init__(client, model, settings, "proposer")
        self.system_prompt = PROPOSER_SYSTEM_PROMPT
        self.tool_definitions = []

    def propose(
        self,
        investigation: Investigation,
        *,
        revision_feedback: str | None = None,
    ) -> Proposal:
        """Build approaches from ``investigation`` and return a :class:`Proposal`.

        Args:
            investigation: Validated findings from the investigator agent.
            revision_feedback: Optional human feedback when revising a prior
                proposal; included in the user message so new options address
                what they rejected or asked to change.

        Returns:
            A validated proposal with one or more approaches.

        Raises:
            AgentError: If Claude fails or the structured output is invalid.
        """
        user_message = self._build_user_message(
            investigation,
            revision_feedback=revision_feedback,
        )
        return self.run(user_message, Proposal)

    def format_as_comment(
        self,
        proposal: Proposal,
        investigation: Investigation | None = None,
    ) -> str:
        """Render ``proposal`` as Markdown suitable for a GitHub issue comment.

        Args:
            proposal: Validated proposal to format for human review.
            investigation: Optional investigation used for a one-line summary.

        Returns:
            Markdown text with collapsible approach details and a closing
            prompt for the human to pick an option or approve a single one.
        """
        lines: list[str] = []

        if investigation is not None:
            lines.extend(
                [
                    "## Investigation summary",
                    investigation.root_cause.strip(),
                    "",
                ]
            )

        lines.extend([PROPOSAL_SECTION_HEADER, ""])

        single_approach = len(proposal.approaches) == 1
        for index, approach in enumerate(proposal.approaches, start=1):
            if index > 1:
                lines.extend(["---", ""])
            lines.extend(
                _format_approach_for_comment(
                    approach,
                    index=index,
                    single_approach=single_approach,
                )
            )
            lines.append("")

        if single_approach:
            closing = (
                "**Reply 'approved' (or with any changes you'd like) to proceed.**"
            )
        else:
            closing = (
                "**Reply with the option number (or a variation you'd "
                "prefer) to proceed.**"
            )

        lines.append(closing)
        return "\n".join(lines).strip()

    def _build_user_message(
        self,
        investigation: Investigation,
        *,
        revision_feedback: str | None = None,
    ) -> str:
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

        message = (
            "Propose fix/build approaches based on this investigation.\n\n"
            f"Issue nature:\n{investigation.issue_nature}\n\n"
            f"Root cause:\n{investigation.root_cause}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Relevant files:\n{files_block}\n\n"
            f"Confidence: {investigation.confidence}\n\n"
            f"Open questions:\n{questions_block}\n\n"
        )
        if revision_feedback and revision_feedback.strip():
            message += (
                "Human revision feedback (they rejected the previous proposal "
                "or asked for different options — address this directly in "
                "your new approaches):\n"
                f"{revision_feedback.strip()}\n\n"
            )
        message += (
            "Respond with only the JSON object specified in your instructions."
        )
        return message

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Refuse tool calls; this agent does not expose any tools."""
        _ = tool_input
        return json.dumps(
            {"error": f"ProposerAgent has no tools; got {tool_name!r}"},
            ensure_ascii=False,
        )


def _normalize_risk_level(risk: str) -> str:
    """Return ``low``, ``medium``, or ``high`` from free-text or noisy model output."""
    cleaned = risk.strip()
    for token in (
        ":large_green_circle:",
        ":green_circle:",
        ":large_yellow_circle:",
        ":yellow_circle:",
        ":red_circle:",
        "large_green_circle",
        "large_yellow_circle",
        "red_circle",
        "green_circle",
        "yellow_circle",
    ):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip(" -—:").lower()
    if "high" in cleaned:
        return "high"
    if "medium" in cleaned or cleaned == "med":
        return "medium"
    if "low" in cleaned:
        return "low"
    return "medium"


def _risk_display(risk: str) -> tuple[str, str]:
    """Return a visible emoji plus a short risk label for GitHub comments."""
    level = _normalize_risk_level(risk)
    emoji_by_level = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
    }
    return emoji_by_level[level], level


def _format_approach_for_comment(
    approach: Approach,
    *,
    index: int,
    single_approach: bool,
) -> list[str]:
    """Return Markdown lines for one approach in the GitHub comment layout."""
    lines: list[str] = []

    if single_approach:
        lines.append(f"**{approach.nature}** — {approach.description}")
    else:
        emoji, risk_level = _risk_display(approach.risk)
        lines.append(
            f"### Option {index}: {approach.name} — {emoji} {risk_level}"
        )
        lines.append("")
        lines.append(f"**{approach.nature}** — {approach.description}")

    lines.extend(
        [
            "",
            "<details>",
            "<summary>Why this works, tradeoffs, and scope</summary>",
            "",
            f"- **Why it works:** {approach.why_it_works}",
            f"- **Tradeoffs:** {approach.tradeoffs}",
            f"- **Scope:** {approach.estimated_scope}",
            "",
            "</details>",
        ]
    )
    return lines
