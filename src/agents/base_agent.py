"""Abstract Claude agent with a tool-use loop and validated structured output."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from config import Settings
from core.claude_client import call_claude
from core.exceptions import AgentError
from utils.logger import get_logger

TModel = TypeVar("TModel", bound=BaseModel)

_MAX_TURNS: int = 25


class BaseAgent(ABC):
    """Claude agent that calls tools until the model returns a final JSON answer.

    Subclasses set ``system_prompt`` and ``tool_definitions`` and implement
    :meth:`_execute_tool`. :meth:`run` owns the Anthropic tool-use loop and
    validates the final text against a caller-supplied pydantic model.
    """

    system_prompt: str
    tool_definitions: list[dict[str, Any]]

    def __init__(
        self,
        client: Anthropic,
        model: str,
        settings: Settings,
        agent_type: str,
    ) -> None:
        """Bind this agent to an Anthropic client, model, and Claude settings.

        Args:
            client: Authenticated Anthropic SDK client.
            model: Model identifier passed to the centralized Claude client.
            settings: Application settings containing per-agent Claude defaults.
            agent_type: Logical agent name used to select effort/temperature/
                max_tokens (for example ``investigator``).
        """
        self._client = client
        self._model = model
        self._settings = settings
        self._agent_type = agent_type
        config = settings.agent_config(agent_type)
        self._effort = config.effort
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._logger = get_logger(__name__)
        self.system_prompt = ""
        self.tool_definitions = []

    @abstractmethod
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Run one tool and return its result as text."""

    def run(self, user_message: str, output_model: type[TModel]) -> TModel:
        """Send ``user_message`` through the tool-use loop and validate the answer."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]

        for turn in range(1, _MAX_TURNS + 1):
            response = self._create_message(messages)
            stop_reason = response.stop_reason

            if stop_reason == "tool_use":
                self._logger.info(
                    "Agent %s starting tool-use turn %d/%d",
                    self._agent_type,
                    turn,
                    _MAX_TURNS,
                )
                messages.append(
                    {"role": "assistant", "content": self._assistant_content(response)}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": self._run_tool_calls(response),
                    }
                )
                self._logger.debug(
                    "Tool-use turn %d/%d complete; continuing",
                    turn,
                    _MAX_TURNS,
                )
                continue

            final_text = self._collect_text(response)
            if not final_text.strip():
                raise AgentError(
                    f"Claude returned no text (stop_reason={stop_reason!r}). "
                    "The model must emit a JSON object matching the output schema."
                )
            return self._parse_output(final_text, output_model)

        raise AgentError(
            f"Agent {self._agent_type!r} exceeded the maximum of {_MAX_TURNS} "
            "tool-use turns without returning final JSON. The model may be "
            "stuck in a tool loop; inspect recent tool errors and retry."
        )

    def _create_message(self, messages: list[dict[str, Any]]) -> Any:
        """Call Claude through the centralized client."""
        return call_claude(
            self._client,
            self._model,
            self.system_prompt,
            messages,
            self.tool_definitions,
            self._effort,
            self._temperature,
            self._max_tokens,
            agent_name=self._agent_type,
            logger=self._logger,
        )

    def _run_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Execute every ``tool_use`` block and build ``tool_result`` payloads."""
        results: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = str(block.name)
            raw_input = block.input if isinstance(block.input, dict) else {}
            tool_input: dict[str, Any] = dict(raw_input)
            self._logger.info("Tool call: %s input=%s", name, tool_input)
            try:
                content = self._execute_tool(name, tool_input)
            except Exception as exc:
                self._logger.error(
                    "Tool %s raised %s: %s", name, type(exc).__name__, exc
                )
                content = json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                }
            )
        if not results:
            raise AgentError(
                "stop_reason was tool_use but the response contained no tool_use blocks."
            )
        return results

    def _assistant_content(self, response: Any) -> list[dict[str, Any]]:
        """Serialize assistant content blocks for the next Messages API turn."""
        serialized: list[dict[str, Any]] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                serialized.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                serialized.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input if isinstance(block.input, dict) else {},
                    }
                )
            else:
                dumped = block.model_dump() if hasattr(block, "model_dump") else {"type": block_type}
                serialized.append(dumped)
        return serialized

    def _collect_text(self, response: Any) -> str:
        """Concatenate all text blocks from a Claude response."""
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)

    def _parse_output(self, text: str, output_model: type[TModel]) -> TModel:
        """Parse Claude's final text as JSON and validate it as ``output_model``."""
        try:
            payload = _extract_json_object(text)
        except json.JSONDecodeError as exc:
            preview = text.strip().replace("\n", " ")[:500]
            raise AgentError(
                f"Agent output was not valid JSON: {exc}. "
                f"Preview: {preview!r}"
            ) from exc

        try:
            return output_model.model_validate(payload)
        except ValidationError as exc:
            raise AgentError(
                f"Agent output failed {output_model.__name__} validation: {exc}"
            ) from exc


def _extract_json_object(text: str) -> Any:
    """Load a JSON object from model text, including fenced markdown blocks."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])
