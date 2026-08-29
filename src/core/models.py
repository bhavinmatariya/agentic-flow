"""Pydantic models for structured data passed between tools and agents.

Every structured payload returned to a downstream agent should be an instance
of one of these models so values are validated and typed rather than raw dicts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
