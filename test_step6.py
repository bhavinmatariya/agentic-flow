#!/usr/bin/env python3
"""Manual smoke test for Step 6: response parser agent.

Usage:
    python test_step6.py

Requires a populated .env file with ANTHROPIC_API_KEY. No GitHub calls.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from anthropic import Anthropic

from agents.response_parser import ResponseParserAgent
from config import ConfigurationError, Settings
from core.exceptions import AgentError
from core.models import ParsedIntent

DEFAULT_MODEL = "claude-sonnet-5"

FAKE_ISSUE_TITLE = "Login button returns 500 on submit"
FAKE_ISSUE_BODY = (
    "When I click Login with valid credentials, the API returns HTTP 500. "
    "Happens in Chrome on staging."
)

FAKE_PROPOSAL_COMMENT = """## Proposed approaches

### 1. Fix null guard in auth handler
**Nature:** permanent fix
**Description:** Add a null check before accessing session.user in the login route.
**Why it works:** Stack traces show AttributeError when session is unset.
**Risk:** low
**Tradeoffs:** Small code change; does not address upstream session middleware misconfiguration.
**Estimated scope:** 1 file, ~10 lines

### 2. Disable broken OAuth provider temporarily
**Nature:** temporary mitigation
**Description:** Feature-flag the OAuth path so users can sign in via email/password only.
**Why it works:** Bypasses the failing code path until a proper fix ships.
**Risk:** medium — some users lose OAuth sign-in
**Tradeoffs:** Fast rollback lever vs incomplete auth coverage.
**Estimated scope:** config change + 1 small code path

---
Reply with the **number or name** of the approach you want us to proceed with, or describe a variation you prefer."""

SAMPLE_REPLIES: list[tuple[str, str]] = [
    ("go with 2", "Approve approach by number"),
    ("can you also check X", "Revise — new information or change request"),
    ("thanks!", "Unrelated general chat"),
]


def _print_parsed_intent(label: str, result: ParsedIntent) -> None:
    """Pretty-print a ParsedIntent to stdout."""
    print(f"--- {label} ---")
    print(f"intent: {result.intent}")
    print(f"selected_approach: {result.selected_approach!r}")
    print(f"feedback: {result.feedback!r}")
    print()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    client = Anthropic(api_key=settings.anthropic_api_key)
    parser = ResponseParserAgent(client, model, settings)

    print("ResponseParserAgent smoke test")
    print(f"Model: {model}")
    print()

    for human_comment, label in SAMPLE_REPLIES:
        print(f"Human reply: {human_comment!r}")
        try:
            parsed = parser.parse(
                FAKE_ISSUE_TITLE,
                FAKE_ISSUE_BODY,
                FAKE_PROPOSAL_COMMENT,
                human_comment,
            )
        except AgentError as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            return 1
        _print_parsed_intent(label, parsed)

    print("Step 6 smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
