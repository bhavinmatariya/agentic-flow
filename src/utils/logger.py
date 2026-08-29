"""Shared logging helpers for agentic-flow.

Library code should obtain loggers through :func:`get_logger` rather than
calling ``print()`` or configuring the root logger. Entry points (``main.py``,
smoke tests) remain responsible for ``logging.basicConfig``.
"""

from __future__ import annotations

import logging

DEFAULT_LOGGER_NAME = "agentic_flow"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a hierarchical logger under the ``agentic_flow`` namespace.

    Args:
        name: Optional dotted suffix (e.g. ``tools.code_search``). When omitted,
            the package root logger is returned. A name that already starts with
            ``agentic_flow`` is used as-is so callers can pass ``__name__``.

    Returns:
        A ``logging.Logger`` instance. Handlers are not attached here; the
        application controls formatting and destination.
    """
    if name is None:
        return logging.getLogger(DEFAULT_LOGGER_NAME)
    if name == DEFAULT_LOGGER_NAME or name.startswith(f"{DEFAULT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")
