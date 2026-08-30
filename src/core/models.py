"""Pydantic models for structured data passed between tools and agents.

Every structured payload returned to a downstream agent should be an instance
of one of these models so values are validated and typed rather than raw dicts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_github_slug(value: str, *, field_name: str) -> str:
    """Validate a GitHub ``owner/repository`` slug.

    Args:
        value: Candidate slug, already stripped of surrounding whitespace.
        field_name: Field being validated, used in the error message.

    Returns:
        The accepted slug.

    Raises:
        ValueError: If the value is not exactly ``owner/repository``.
    """
    if value.count("/") != 1 or not all(part.strip() for part in value.split("/")):
        raise ValueError(
            f"{field_name} must be in 'owner/repository' format, got: {value!r}"
        )
    return value


class CodeMatch(BaseModel):
    """A single line-level hit produced by a code search.

    Attributes:
        file: Repository-relative path using POSIX separators.
        line_number: 1-based line number of the match inside ``file``.
        line: The matching line text, without a trailing newline.
    """

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., min_length=1)
    line_number: int = Field(..., ge=1)
    line: str

    @field_validator("file")
    @classmethod
    def _normalize_file_path(cls, value: str) -> str:
        """Store search hits with stable POSIX-style relative paths."""
        normalized = value.replace("\\", "/").strip()
        if not normalized:
            raise ValueError("file path must not be empty")
        return normalized


class LinkedRepo(BaseModel):
    """A named secondary repository listed in ``repos.json``.

    Attributes:
        name: Short alias agents use to refer to this repo (e.g. ``backend``).
        repo: GitHub slug in ``owner/repository`` form.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)

    @field_validator("repo")
    @classmethod
    def _validate_repo_slug(cls, value: str) -> str:
        """Reject slugs that are not ``owner/repository``."""
        return _require_github_slug(value, field_name="repo")


class RepoConfig(BaseModel):
    """Optional extra repositories to search besides the primary.

    Loaded from ``repos.json``. The primary repo is not stored here; it comes
    from ``Settings.github_repo``. Unknown keys are rejected so typos surface
    as validation errors instead of being silently ignored.

    Attributes:
        linked: Additional repositories the agent may search for context.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    linked: list[LinkedRepo] = Field(default_factory=list)


class RelevantFile(BaseModel):
    """A file the investigator found relevant to the issue.

    Attributes:
        repo: GitHub slug of the repository that contains the file.
        path: Repository-relative path to the file.
        reason: Why this file matters for the investigation.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    repo: str = Field(..., min_length=1)
    path: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)


class Investigation(BaseModel):
    """Structured findings from the investigator agent.

    ``issue_nature`` is free text written by the model. It is not an enum and
    must not be forced into categories such as bug, feature, or spike.

    Attributes:
        issue_nature: The model's own description of what the issue is.
        root_cause: Evidence-backed hypothesis of why the issue exists.
        evidence: Concrete observations from the code that support the
            hypothesis.
        relevant_files: Files that matter, each tied to a repo and a reason.
        confidence: How strongly the evidence supports the hypothesis.
        open_questions: Unresolved questions the investigator could not
            answer from the code.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issue_nature: str = Field(..., min_length=1)
    root_cause: str = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)
    relevant_files: list[RelevantFile] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    open_questions: list[str] = Field(default_factory=list)


class Approach(BaseModel):
    """One viable fix or mitigation path proposed for an issue.

    ``nature`` is free text (for example ``temporary mitigation`` or
    ``permanent fix``). It is not an enum; the model chooses wording per issue.

    Attributes:
        name: Short title for this approach.
        nature: The model's own label for what kind of solution this is.
        description: What the approach does in plain language.
        why_it_works: How it addresses the root cause or mitigates symptoms.
        risk: Risk level and what could go wrong.
        tradeoffs: Speed, durability, scope, and other trade-offs.
        estimated_scope: Rough effort or touch-area estimate in free text.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    nature: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    why_it_works: str = Field(..., min_length=1)
    risk: str = Field(..., min_length=1)
    tradeoffs: str = Field(..., min_length=1)
    estimated_scope: str = Field(..., min_length=1)


class Proposal(BaseModel):
    """Variable-length set of approaches produced by the proposer agent.

    The proposer decides how many distinct, viable approaches exist for the
    issue. There is no fixed count enforced beyond requiring at least one.

    Attributes:
        approaches: One or more proposed ways to address the investigation.
    """

    model_config = ConfigDict(extra="forbid")

    approaches: list[Approach] = Field(..., min_length=1)


class Subtask(BaseModel):
    """One ordered slice of an approved approach for sequential implementation.

    Attributes:
        name: Short title for this step.
        description: What to implement in this step only.
        scope: Rough touch-area estimate (files/layers), kept small.
        order: 1-based execution order.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    scope: str = Field(..., min_length=1)
    order: int = Field(..., ge=1)


class SubtaskPlan(BaseModel):
    """Ordered subtasks covering an approved approach end-to-end."""

    model_config = ConfigDict(extra="forbid")

    subtasks: list[Subtask] = Field(..., min_length=1, max_length=5)

    @model_validator(mode="after")
    def _sort_subtasks(self) -> SubtaskPlan:
        """Store subtasks in ascending ``order``."""
        ordered = sorted(self.subtasks, key=lambda item: item.order)
        if ordered != self.subtasks:
            object.__setattr__(self, "subtasks", ordered)
        return self


class ParsedIntent(BaseModel):
    """Structured interpretation of a human reply to a fix proposal.

    Attributes:
        intent: Whether the human approved, asked for revision, or said
            something unrelated to choosing an approach.
        selected_approach: Name or reference to the chosen approach when
            ``intent`` is ``approve``; otherwise ``None``.
        feedback: Revision notes when ``intent`` is ``revise`` (required), or
            extra requirements when ``intent`` is ``approve`` (optional).
            Must be ``None`` for ``unrelated`` and plain approve with no extras.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Literal["approve", "revise", "unrelated"]
    selected_approach: str | None = None
    feedback: str | None = None

    @model_validator(mode="after")
    def _validate_feedback_for_intent(self) -> ParsedIntent:
        """Ensure ``feedback`` matches the parsed intent."""
        if self.intent == "revise":
            if self.feedback is None or not self.feedback.strip():
                raise ValueError("feedback must be set when intent is 'revise'")
            return self
        if self.intent == "unrelated" and self.feedback is not None:
            raise ValueError("feedback must be null when intent is 'unrelated'")
        return self


class ImplementationResult(BaseModel):
    """Outcome of the implementer agent's code changes.

    Attributes:
        branch_name: Branch where the fix was committed.
        files_changed: Repository-relative paths that were edited.
        summary: Plain-language summary of what was implemented.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    branch_name: str = Field(..., min_length=1)
    files_changed: list[str] = Field(default_factory=list)
    summary: str = Field(..., min_length=1)


class ReviewResult(BaseModel):
    """Self-review outcome from the reviewer agent.

    Attributes:
        approved: Whether the change is ready to open as a pull request.
        summary: Overall review verdict in plain language.
        findings: Specific issues, test failures, or verification notes.
        making_progress: Whether another implement/review round is likely to help.
            Set to ``false`` only when the reviewer sees no viable path forward.
        layers_detected: Which change layers were detected in the diff
            (for example frontend, database, backend flags).
        layers_checked: Which layers had automated checks actually executed.
        test_output_summary: Human-readable summary of tested vs skipped checks.
        ui_verification: Playwright live UI check result when the full-stack
            tier ran; ``None`` when live UI verification did not apply.
        db_verification: Independent database row check when the full-stack
            tier ran; ``None`` when live DB verification did not apply.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approved: bool
    summary: str = Field(..., min_length=1)
    findings: list[str] = Field(default_factory=list)
    making_progress: bool = True
    layers_detected: dict[str, bool] = Field(default_factory=dict)
    layers_checked: dict[str, bool] = Field(default_factory=dict)
    test_output_summary: str = ""
    ui_verification: dict[str, Any] | None = None
    db_verification: dict[str, Any] | None = None
