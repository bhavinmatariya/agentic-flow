"""Response parser agent: interpret human replies to fix proposals."""

from __future__ import annotations

import json
from typing import Any, Final

from anthropic import Anthropic

from agents.base_agent import BaseAgent
from config import Settings
from core.models import ParsedIntent

RESPONSE_PARSER_SYSTEM_PROMPT: Final[str] = (
    "You are reading a human's reply on a GitHub issue where they were "
    "shown one or more proposed fix approaches. Determine their intent — "
    "approve (and which approach, matched by name or clear reference, "
    "even if worded loosely), revise (they reject the proposal(s) and "
    "want different options before proceeding), or unrelated (general "
    "chat, not a decision).\n\n"
    "IMPORTANT: If the human accepts or approves the proposed approach in "
    "principle but also adds extra requirements, tweaks, or conditions "
    "(phrases like \"approved with changes:\", \"go with option X but "
    "also...\", \"yes, and also make it...\", \"approved — please also "
    "...\"), classify as intent=\"approve\" with selected_approach set and "
    "the extra requirements captured in feedback. That is NOT revise.\n\n"
    "Reserve intent=\"revise\" ONLY when the human rejects the proposed "
    "approach(es) entirely or asks for different/additional options before "
    "proceeding — e.g. \"I don't like either of these, what else could "
    "work?\", \"none of these are right\", or \"can you propose other "
    "alternatives?\". Do not use revise when they picked an option and "
    "added implementation details.\n\n"
    "Respond with ONLY JSON: {\"intent\": ..., \"selected_approach\": ..., "
    "\"feedback\": ...}"
)


class ResponseParserAgent(BaseAgent):
    """Parse a human issue comment into a structured :class:`ParsedIntent`.

    This agent does not use tools; it reasons from issue context, the
    agent's proposal comment, and the human's reply text.
    """

    def __init__(self, client: Anthropic, model: str, settings: Settings) -> None:
        """Create a response parser bound to an Anthropic client and model id.

        Args:
            client: Authenticated Anthropic SDK client.
            model: Model identifier passed to the centralized Claude client.
            settings: Application settings with per-agent Claude defaults.
        """
        super().__init__(client, model, settings, "response_parser")
        self.system_prompt = RESPONSE_PARSER_SYSTEM_PROMPT
        self.tool_definitions = []

    def parse(
        self,
        issue_title: str,
        issue_body: str,
        proposal_comment: str,
        human_comment: str,
    ) -> ParsedIntent:
        """Classify ``human_comment`` against the posted proposal.

        Args:
            issue_title: GitHub issue title for context.
            issue_body: GitHub issue body for context (may be empty).
            proposal_comment: Markdown comment the agent posted with approaches.
            human_comment: The human reply to interpret.

        Returns:
            Validated parsed intent.

        Raises:
            AgentError: If Claude fails or the structured output is invalid.
        """
        user_message = self._build_user_message(
            issue_title=issue_title,
            issue_body=issue_body,
            proposal_comment=proposal_comment,
            human_comment=human_comment,
        )
        return self.run(user_message, ParsedIntent)

    def _build_user_message(
        self,
        *,
        issue_title: str,
        issue_body: str,
        proposal_comment: str,
        human_comment: str,
    ) -> str:
        """Assemble the user turn from issue and comment text."""
        body = issue_body.strip() if issue_body else "(empty)"
        return (
            "Parse the human's reply to the agent's fix proposal.\n\n"
            f"Issue title: {issue_title}\n\n"
            f"Issue body:\n{body}\n\n"
            f"Agent proposal comment:\n{proposal_comment}\n\n"
            f"Human reply:\n{human_comment}\n\n"
            "Respond with only the JSON object specified in your instructions. "
            "Set selected_approach when intent is approve (null otherwise). "
            "Set feedback when intent is revise (required) or approve with "
            "extra requirements or tweaks (optional). Set feedback to null "
            "for unrelated or plain approve with no extra requirements."
        )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Refuse tool calls; this agent does not expose any tools."""
        _ = tool_input
        return json.dumps(
            {"error": f"ResponseParserAgent has no tools; got {tool_name!r}"},
            ensure_ascii=False,
        )
