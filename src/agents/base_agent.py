"""Abstract Claude agent with a tool-use loop and validated structured output."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Final, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from config import Settings
from core.claude_client import call_claude
from core.exceptions import AgentError
from utils.logger import get_logger

TModel = TypeVar("TModel", bound=BaseModel)

_MAX_TURNS: int = 25
_MAX_JSON_RETRIES: int = 2
_TOKEN_BUDGET_THRESHOLD: int = 400_000
_CHARS_PER_TOKEN_ESTIMATE: int = 4
_TOOL_RESULT_OMITTED_PLACEHOLDER: Final[str] = (
    "file already read, contents omitted"
)
_JSON_RETRY_PROMPT: Final[str] = (
    "Your last response was not valid JSON. Respond with ONLY a single valid "
    "JSON object matching the required schema — no prose before or after."
)
_FORCE_JSON_PROMPT: Final[str] = (
    "You have used all allowed tool steps. Do NOT call any more tools. "
    "Respond with ONLY the final JSON object required by your instructions — "
    "no prose, no markdown fences."
)
_TURN_LIMIT_WARNING_AT: int = 20


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
        self._run_session_label: str | None = None
        self._max_tool_turns = _MAX_TURNS

    @abstractmethod
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Run one tool and return its result as text."""

    def run(self, user_message: str, output_model: type[TModel]) -> TModel:
        """Send ``user_message`` through the tool-use loop and validate the answer.

        Each call starts a brand-new conversation (``messages`` is empty except
        for the provided user turn). Prior tool-use transcripts are never reused.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message},
        ]

        for turn in range(1, self._max_tool_turns + 1):
            if turn == _TURN_LIMIT_WARNING_AT:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You are on tool step {turn}/{self._max_tool_turns}. "
                            "Finish soon: complete remaining edit_file calls, then "
                            "return ONLY the final JSON object."
                        ),
                    }
                )
            response = self._create_message(messages)
            stop_reason = response.stop_reason

            if stop_reason == "tool_use":
                session = self._run_session_label or self._agent_type
                self._logger.info(
                    "Agent %s [%s] tool step %d/%d (resets each implement/review call)",
                    self._agent_type,
                    session,
                    turn,
                    self._max_tool_turns,
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
                    self._max_tool_turns,
                )
                continue

            final_text = self._collect_text(response)
            if not final_text.strip():
                if stop_reason == "max_tokens":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": self._assistant_content(response),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response hit the output token limit "
                                "before finishing. Do NOT call more tools. Respond "
                                "with ONLY the final JSON object required by your "
                                "instructions — no prose, no markdown fences."
                            ),
                        }
                    )
                    self._logger.warning(
                        "Agent %s hit max_tokens on turn %d; requesting final JSON only",
                        self._agent_type,
                        turn,
                    )
                    continue
                raise AgentError(
                    f"Claude returned no text (stop_reason={stop_reason!r}). "
                    "The model must emit a JSON object matching the output schema."
                )
            return self._finalize_json_response(response, messages, output_model)

        self._logger.warning(
            "Agent %s hit tool turn limit (%d); requesting final JSON only",
            self._agent_type,
            self._max_tool_turns,
        )
        messages.append({"role": "user", "content": _FORCE_JSON_PROMPT})
        final_response = self._create_message(messages)
        if final_response.stop_reason == "tool_use":
            raise AgentError(
                f"Agent {self._agent_type!r} exceeded the maximum of "
                f"{self._max_tool_turns} tool-use turns without returning final "
                "JSON. The model may be stuck in a tool loop; inspect recent "
                "tool errors and retry."
            )
        final_text = self._collect_text(final_response)
        if not final_text.strip():
            raise AgentError(
                f"Agent {self._agent_type!r} exceeded the maximum of "
                f"{self._max_tool_turns} tool-use turns without returning final "
                "JSON. The model may be stuck in a tool loop; inspect recent "
                "tool errors and retry."
            )
        return self._finalize_json_response(final_response, messages, output_model)

    def _create_message(self, messages: list[dict[str, Any]]) -> Any:
        """Call Claude through the centralized client."""
        self._apply_token_budget(messages)
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
            fallback_model=self._settings.anthropic_fallback_model,
        )

    def _apply_token_budget(self, messages: list[dict[str, Any]]) -> None:
        """Trim oldest tool results when the session exceeds the token budget."""
        while _estimate_input_tokens(messages, self.system_prompt) > _TOKEN_BUDGET_THRESHOLD:
            if not _trim_oldest_tool_result(messages):
                break
            self._logger.warning(
                "Agent %s trimmed oldest tool_result to stay under token budget "
                "(estimated input tokens now ~%d)",
                self._agent_type,
                _estimate_input_tokens(messages, self.system_prompt),
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

    def _finalize_json_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        output_model: type[TModel],
    ) -> TModel:
        """Parse the model's final answer, retrying invalid JSON in-conversation."""
        current_response = response
        last_error: Exception | None = None

        for attempt_index in range(_MAX_JSON_RETRIES + 1):
            final_text = self._collect_text(current_response)
            if not final_text.strip():
                raise AgentError(
                    "Claude returned no text when a JSON object was required."
                )

            try:
                return self._parse_output(final_text, output_model)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt_index >= _MAX_JSON_RETRIES:
                    break

                self._logger.warning(
                    "Agent %s returned invalid JSON, retrying (attempt %d)",
                    self._agent_type,
                    attempt_index + 1,
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._assistant_content(current_response),
                    }
                )
                messages.append({"role": "user", "content": _JSON_RETRY_PROMPT})
                current_response = self._create_message(messages)
                if current_response.stop_reason == "tool_use":
                    raise AgentError(
                        f"Agent {self._agent_type!r} returned tool_use when asked "
                        "to fix invalid JSON output."
                    ) from exc

        assert last_error is not None
        if isinstance(last_error, json.JSONDecodeError):
            preview = self._collect_text(current_response).strip().replace("\n", " ")[:500]
            raise AgentError(
                f"Agent output was not valid JSON after {_MAX_JSON_RETRIES} "
                f"retries: {last_error}. Preview: {preview!r}"
            ) from last_error
        raise AgentError(
            f"Agent output failed {output_model.__name__} validation after "
            f"{_MAX_JSON_RETRIES} retries: {last_error}"
        ) from last_error

    def _parse_output(self, text: str, output_model: type[TModel]) -> TModel:
        """Parse Claude's final text as JSON and validate it as ``output_model``."""
        payload = _extract_json_object(text)
        return output_model.model_validate(payload)


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


def _estimate_input_tokens(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> int:
    """Return a conservative estimate of input tokens for ``messages``."""
    payload = json.dumps(
        {"system": system_prompt, "messages": messages},
        ensure_ascii=False,
        default=str,
    )
    return max(1, len(payload) // _CHARS_PER_TOKEN_ESTIMATE)


def _trim_oldest_tool_result(messages: list[dict[str, Any]]) -> bool:
    """Replace the oldest large tool_result body with a short placeholder."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if not isinstance(body, str):
                continue
            if body == _TOOL_RESULT_OMITTED_PLACEHOLDER:
                continue
            block["content"] = _TOOL_RESULT_OMITTED_PLACEHOLDER
            return True
    return False
