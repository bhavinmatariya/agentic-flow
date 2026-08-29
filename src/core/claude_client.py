"""Centralized Anthropic Messages API access for all agents."""

from __future__ import annotations

import logging
import time
from typing import Any

from anthropic import Anthropic, AnthropicError, APIStatusError, RateLimitError

from core.exceptions import AgentError
from utils.logger import get_logger

_MAX_ATTEMPTS = 3
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 1.0

# USD per 1M tokens (input, output) for rough logging estimates.
_MODEL_COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
}


def call_claude(
    client: Anthropic,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    effort: str,
    temperature: float,
    max_tokens: int,
    *,
    agent_name: str,
    logger: logging.Logger | None = None,
) -> Any:
    """Invoke Claude Messages API with retries, effort config, and structured logging.

    This is the only function in agentic-flow that calls ``client.messages.create``.

    Args:
        client: Authenticated Anthropic SDK client.
        model: Model identifier.
        system_prompt: System prompt for the call.
        messages: Conversation messages for the API.
        tools: Tool definitions; may be empty.
        effort: Effort level passed via ``output_config``.
        temperature: Configured sampling temperature (logged only; omitted from
            the API request because effort-based Claude models reject it).
        max_tokens: Maximum output tokens.
        agent_name: Logical agent name used in logs (for example ``investigator``).
        logger: Optional logger; defaults to the shared module logger.

    Returns:
        Raw Anthropic message response object.

    Raises:
        AgentError: When the API fails after retries or a non-retryable error occurs.
    """
    log = logger or get_logger(__name__)
    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
                "output_config": {"effort": effort},
            }
            if tools:
                request_kwargs["tools"] = tools
            # Claude 4.6+ / Sonnet 5 / Opus 5 reject temperature, top_p, and top_k
            # when using output_config.effort. Keep temperature in logs/config only.

            response = client.messages.create(**request_kwargs)
            _log_claude_call(
                log,
                agent_name=agent_name,
                model=model,
                effort=effort,
                temperature=temperature,
                max_tokens=max_tokens,
                response=response,
                attempt=attempt,
            )
            return response
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt >= _MAX_ATTEMPTS:
                break
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            log.warning(
                "Claude call retry for agent=%s model=%s effort=%s "
                "(attempt %d/%d, sleeping %.1fs): %s",
                agent_name,
                model,
                effort,
                attempt,
                _MAX_ATTEMPTS,
                delay,
                exc,
            )
            time.sleep(delay)

    assert last_error is not None
    raise AgentError(
        "Claude API call failed for "
        f"agent={agent_name!r}, model={model!r}, effort={effort!r}, "
        f"temperature={temperature}, max_tokens={max_tokens} "
        f"after {_MAX_ATTEMPTS} attempt(s): {last_error}"
    ) from last_error


def _is_retryable(exc: Exception) -> bool:
    """Return True for rate limits and transient server errors."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, AnthropicError):
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
            return True
    return False


def _log_claude_call(
    logger: logging.Logger,
    *,
    agent_name: str,
    model: str,
    effort: str,
    temperature: float,
    max_tokens: int,
    response: Any,
    attempt: int,
) -> None:
    """Log token usage and estimated cost for a successful Claude call."""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cost_usd = _estimate_cost_usd(model, input_tokens, output_tokens)
    cost_text = f"${cost_usd:.6f}" if cost_usd is not None else "n/a"

    logger.info(
        "Claude call agent=%s model=%s effort=%s configured_temperature=%s "
        "(omitted from API) max_tokens=%d attempt=%d input_tokens=%s "
        "output_tokens=%s estimated_cost=%s",
        agent_name,
        model,
        effort,
        temperature,
        max_tokens,
        attempt,
        input_tokens,
        output_tokens,
        cost_text,
    )


def _estimate_cost_usd(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Estimate USD cost when model pricing is known."""
    if input_tokens is None or output_tokens is None:
        return None

    pricing = _MODEL_COST_PER_MILLION.get(model)
    if pricing is None:
        for known_model, known_pricing in _MODEL_COST_PER_MILLION.items():
            if model.startswith(known_model):
                pricing = known_pricing
                break
    if pricing is None:
        return None

    input_rate, output_rate = pricing
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
