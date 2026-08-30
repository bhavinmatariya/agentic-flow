"""Shared logging helpers for agentic-flow.

Library code should obtain loggers through :func:`get_logger`. Entry points
use :class:`RunReporter` for retro terminal banners, GitHub Actions groups,
annotations, and step summaries.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import pyfiglet

DEFAULT_LOGGER_NAME = "agentic_flow"

# Force ANSI green — GitHub Actions is not a TTY but renders these codes.
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
_STARTUP_FONT = "slant"
_STAGE_FONT = "small"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a hierarchical logger under the ``agentic_flow`` namespace."""
    if name is None:
        return logging.getLogger(DEFAULT_LOGGER_NAME)
    if name == DEFAULT_LOGGER_NAME or name.startswith(f"{DEFAULT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")


def ascii_banner(text: str, *, compact: bool = False) -> str:
    """Return ``text`` as green ASCII art suitable for GitHub Actions logs."""
    font = _STAGE_FONT if compact else _STARTUP_FONT
    try:
        art = pyfiglet.figlet_format(text, font=font)
    except pyfiglet.FontNotFound:
        art = pyfiglet.figlet_format(text, font="standard")
    return f"{ANSI_GREEN}{ANSI_BOLD}{art.rstrip()}{ANSI_RESET}"


def progress_bar(percent: int, width: int = 26) -> str:
    """Return a retro runner progress line like ``[🏃........] 15%``."""
    clamped = max(0, min(100, int(percent)))
    filled = int(round(width * clamped / 100))
    if filled >= width:
        filled = width - 1
    cells = ["·"] * width
    cells[filled] = "🏃"
    return f"[{''.join(cells)}] {clamped}%"


def _gh_escape(text: str) -> str:
    return (
        text.replace("%", "%25")
        .replace("\r", "")
        .replace("\n", "%0A")
    )


def emit_github_notice(title: str, message: str) -> None:
    """Emit a ``::notice`` workflow command."""
    print(f"::notice title={_gh_escape(title)}::{_gh_escape(message)}", flush=True)


def emit_github_warning(title: str, message: str) -> None:
    """Emit a ``::warning`` workflow command."""
    print(f"::warning title={_gh_escape(title)}::{_gh_escape(message)}", flush=True)


def emit_github_error(title: str, message: str) -> None:
    """Emit an ``::error`` workflow command."""
    print(f"::error title={_gh_escape(title)}::{_gh_escape(message)}", flush=True)


@dataclass
class StageRecord:
    """One pipeline stage recorded for the step summary."""

    name: str
    ok: bool
    note: str = ""
    checkpoint: str = ""


@dataclass
class RunReporter:
    """Retro terminal UI + GitHub Actions reporting for one pipeline run."""

    issue_number: int
    event: str
    repo: str = ""
    stages: list[StageRecord] = field(default_factory=list)
    round_count: int = 0
    files_changed: list[str] = field(default_factory=list)
    pr_url: str | None = None
    outcome: str = "running"
    outcome_detail: str = ""
    _startup_printed: bool = field(default=False, repr=False)
    _checkpoint_index: int = field(default=0, repr=False)
    _checkpoint_total: int = field(default=0, repr=False)

    def plan_checkpoints(self, total: int, *, start_at: int = 0) -> None:
        """Reserve checkpoint budget; ``start_at`` restores progress on resume."""
        self._checkpoint_total = max(1, int(total))
        self._checkpoint_index = max(0, min(int(start_at), self._checkpoint_total - 1))

    @property
    def checkpoint_index(self) -> int:
        """Return how many implement/review checkpoints have completed."""
        return self._checkpoint_index

    @property
    def checkpoint_total(self) -> int:
        """Return the planned implement/review checkpoint budget."""
        return self._checkpoint_total

    def _next_checkpoint(self, detail: str) -> tuple[str, int]:
        """Advance the checkpoint counter and return label + progress percent."""
        if self._checkpoint_total <= 0:
            return detail, 50
        self._checkpoint_index += 1
        label = f"Checkpoint {self._checkpoint_index}/{self._checkpoint_total} · {detail}"
        percent = int(round(self._checkpoint_index / self._checkpoint_total * 95))
        return label, max(1, min(95, percent))

    def print_startup_banner(self) -> None:
        """Print the main AGENTIC FLOW banner once per run."""
        if self._startup_printed:
            return
        print(ascii_banner("AGENTIC FLOW"), flush=True)
        print(
            f"{ANSI_GREEN}issue #{self.issue_number} · event={self.event}{ANSI_RESET}",
            flush=True,
        )
        self._startup_printed = True

    @contextmanager
    def stage(self, name: str, percent: int, *, detail: str = "") -> Iterator[None]:
        """Print a stage banner, progress bar, and collapse detailed logs."""
        self.print_startup_banner()
        checkpoint = ""
        if detail:
            checkpoint, percent = self._next_checkpoint(detail)
        print(ascii_banner(name, compact=True), flush=True)
        if checkpoint:
            print(f"{ANSI_GREEN}{checkpoint}{ANSI_RESET}", flush=True)
        print(f"{ANSI_GREEN}{progress_bar(percent)}{ANSI_RESET}", flush=True)
        print("::group::Details", flush=True)
        stage_ok = True
        stage_note = ""
        try:
            yield
        except BaseException as exc:
            stage_ok = False
            stage_note = str(exc).split("\n", 1)[0][:200]
            raise
        finally:
            print("::endgroup::", flush=True)
            self.stages.append(
                StageRecord(
                    name=name,
                    ok=stage_ok,
                    note=stage_note,
                    checkpoint=checkpoint,
                )
            )

    def record_outcome_proposal_posted(self, *, approaches: int) -> None:
        """Record a successful issue_opened proposal run."""
        self.outcome = "proposal_posted"
        self.outcome_detail = f"Posted {approaches} approach(es) for approval"
        with self.stage("DONE", 100):
            get_logger(__name__).info(
                "Proposal posted on issue #%s (%d approaches)",
                self.issue_number,
                approaches,
            )
        emit_github_notice(
            "Agentic Flow",
            f"Proposal posted on issue #{self.issue_number} ({approaches} approaches)",
        )

    def record_outcome_pr_opened(self, pr_url: str, *, round_count: int, files: list[str]) -> None:
        """Record a successful implement/review/PR run."""
        self.outcome = "pr_opened"
        self.pr_url = pr_url
        self.round_count = round_count
        self.files_changed = list(files)
        self.outcome_detail = f"PR opened: {pr_url}"
        with self.stage("DONE", 100):
            get_logger(__name__).info(
                "Pull request opened for issue #%s: %s (rounds=%d)",
                self.issue_number,
                pr_url,
                round_count,
            )
        emit_github_notice("Agentic Flow", f"PR opened: {pr_url}")

    def record_outcome_needs_human(self, *, round_count: int = 0, files: list[str] | None = None) -> None:
        """Record a stalled orchestrator run."""
        self.outcome = "needs_human"
        self.round_count = round_count
        if files:
            self.files_changed = list(files)
        self.outcome_detail = f"Needs human input on issue #{self.issue_number}"
        with self.stage("FAILED", 100):
            get_logger(__name__).warning(
                "Pipeline stalled for issue #%s after %d round(s)",
                self.issue_number,
                round_count,
            )
        emit_github_warning(
            "Agentic Flow",
            f"Needs human input on issue #{self.issue_number}",
        )

    def record_outcome_error(self, error: str) -> None:
        """Record an unhandled pipeline failure."""
        self.outcome = "error"
        self.outcome_detail = error
        with self.stage("FAILED", 100):
            get_logger(__name__).error("Pipeline failed: %s", error)
        emit_github_error("Agentic Flow", error)

    def record_outcome_noop(self, reason: str) -> None:
        """Record an intentional no-op exit (unrelated comment, missing label, etc.)."""
        self.outcome = "noop"
        self.outcome_detail = reason
        get_logger(__name__).info("No action taken: %s", reason)

    def write_step_summary(self, exit_code: int) -> None:
        """Append a Markdown report to ``GITHUB_STEP_SUMMARY`` when set."""
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
        if not summary_path:
            return

        lines = [
            "## Agentic Flow run summary",
            "",
            f"- **Repository:** `{self.repo or '(unknown)'}`",
            f"- **Issue:** #{self.issue_number}",
            f"- **Event:** `{self.event}`",
            f"- **Exit code:** `{exit_code}`",
            f"- **Outcome:** `{self.outcome}`",
        ]
        if self.outcome_detail:
            lines.append(f"- **Detail:** {self.outcome_detail}")
        if self.round_count:
            lines.append(f"- **Orchestrator rounds:** {self.round_count}")
        if self.files_changed:
            files_preview = ", ".join(f"`{path}`" for path in self.files_changed[:12])
            if len(self.files_changed) > 12:
                files_preview += f" … (+{len(self.files_changed) - 12} more)"
            lines.append(f"- **Files changed:** {files_preview}")
        if self.pr_url:
            lines.append(f"- **Pull request:** {self.pr_url}")

        lines.extend(["", "### Stages", ""])
        if self.stages:
            for stage in self.stages:
                icon = ":white_check_mark:" if stage.ok else ":x:"
                note = f" — {stage.note}" if stage.note else ""
                checkpoint = f"{stage.checkpoint} · " if stage.checkpoint else ""
                lines.append(f"- {icon} {checkpoint}**{stage.name}**{note}")
        else:
            lines.append("- _No staged work recorded_")

        content = "\n".join(lines) + "\n"
        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            print(
                f"::warning title=Agentic Flow::{_gh_escape(f'Could not write step summary: {exc}')}",
                file=sys.stderr,
                flush=True,
            )
