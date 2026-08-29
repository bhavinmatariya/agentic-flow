#!/usr/bin/env python3
"""Entry point for the agentic-flow GitHub Action and local CLI runs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adapters.base import AdapterError
from adapters.github_adapter import GitHubAdapter
from config import ConfigurationError, Settings

logger = logging.getLogger(__name__)

AGENT_ACK_MESSAGE = "🤖 Agent received this issue."


def main() -> int:
    """Load config, connect to GitHub, and acknowledge the target issue."""
    parser = argparse.ArgumentParser(
        description="Post an agent acknowledgment comment on a GitHub issue."
    )
    parser.add_argument(
        "--issue-number",
        type=int,
        required=True,
        help="GitHub issue number to acknowledge",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env()
        adapter = GitHubAdapter(settings)
        comment = adapter.post_comment(args.issue_number, AGENT_ACK_MESSAGE)
        logger.info(
            "Posted acknowledgment on issue #%s (comment id=%s)",
            args.issue_number,
            comment["id"],
        )
        return 0
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    except AdapterError as exc:
        logger.error("GitHub adapter error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
