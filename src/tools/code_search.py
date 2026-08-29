"""Clone, list, read, and search local checkouts of GitHub repositories.

``CodeSearchTool`` is intentionally generic. It does not assume a language,
framework, or directory layout; callers supply the working directory, repo
slug, optional extension filter, and search query. Downstream agents receive
validated pydantic models rather than raw dicts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from core.exceptions import ToolError
from core.models import CodeMatch, RepoConfig
from utils.logger import get_logger

_GREP_HIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<file>.+):(?P<line_number>\d+):(?P<line>.*)$"
)
_CLONE_TIMEOUT_SECONDS: Final[int] = 180
_SEARCH_TIMEOUT_SECONDS: Final[int] = 60
_GIT_DIR_NAME: Final[str] = ".git"


class CodeSearchTool:
    """Clone repositories into a working directory and inspect their contents.

    External processes (``git``, ``rg``, ``grep``) are invoked through
    subprocess and translated into :class:`ToolError` so callers never deal
    with raw ``CalledProcessError`` or OS errors.
    """

    def __init__(
        self,
        working_dir: str,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a tool bound to a local working directory for checkouts.

        Args:
            working_dir: Directory where cloned repositories are stored. Created
                if it does not already exist.
            logger: Optional logger. When omitted, the shared agentic-flow
                logger for this module is used.
        """
        self._working_dir = Path(working_dir).expanduser()
        self._logger = logger or get_logger(__name__)
        try:
            self._working_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(
                f"Could not create working directory {self._working_dir}: {exc}"
            ) from exc
        self._logger.debug("CodeSearchTool working directory: %s", self._working_dir)

    def clone_repo(self, repo_full_name: str, github_token: str) -> str:
        """Shallow-clone ``owner/repo`` into the working directory.

        If a checkout of that slug is already present, the existing path is
        returned without fetching again. The token is used only for HTTPS
        authentication and is never written into the recorded ``origin`` URL.

        Args:
            repo_full_name: GitHub slug in ``owner/repository`` form.
            github_token: Token with read access to the target repository.

        Returns:
            Absolute path to the local checkout.

        Raises:
            ToolError: If the slug is invalid, git is missing, or the clone
                fails for any other reason.
        """
        owner, name = self._split_slug(repo_full_name)
        dest = self._working_dir / owner / name
        token = github_token.strip()
        if not token:
            raise ToolError("github_token must not be empty")

        if dest.exists():
            if self._is_git_checkout(dest):
                self._logger.info(
                    "Skipping clone of %s; already present at %s",
                    repo_full_name,
                    dest,
                )
                return str(dest.resolve())
            raise ToolError(
                f"Refusing to clone {repo_full_name!r}: {dest} exists and is "
                "not a git checkout. Remove it or choose a different working_dir."
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        authenticated_url = (
            f"https://x-access-token:{token}@github.com/{owner}/{name}.git"
        )
        public_url = f"https://github.com/{owner}/{name}.git"
        env = self._git_env()

        self._logger.info("Shallow-cloning %s into %s", repo_full_name, dest)
        try:
            self._run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    authenticated_url,
                    str(dest),
                ],
                env=env,
                timeout=_CLONE_TIMEOUT_SECONDS,
                secret=token,
                action=f"git clone {repo_full_name}",
            )
        except ToolError:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            raise

        try:
            self._run(
                ["git", "remote", "set-url", "origin", public_url],
                cwd=str(dest),
                env=env,
                timeout=30,
                secret=token,
                action=f"git remote set-url origin for {repo_full_name}",
            )
        except ToolError as exc:
            self._logger.warning(
                "Clone of %s succeeded but origin URL could not be scrubbed: %s",
                repo_full_name,
                exc,
            )

        return str(dest.resolve())

    def load_linked_repos(self, config_path: str) -> RepoConfig:
        """Parse optional linked repositories from a ``repos.json`` file.

        Cross-repo context is optional. A missing file is not an error: the
        returned config has an empty ``linked`` list. The primary repository
        is never read from this file; callers take it from
        ``Settings.github_repo``.

        Args:
            config_path: Path to a JSON file with a ``linked`` array, matching
                ``repos.json.example``.

        Returns:
            A validated ``RepoConfig`` instance. ``linked`` is empty when the
            file does not exist.

        Raises:
            ToolError: If the file exists but cannot be read, is not valid
                JSON, or fails pydantic validation (wrong types, unknown keys
                such as a leftover ``primary`` field, or bad slugs).
        """
        path = Path(config_path)
        self._logger.debug("Loading linked-repo config from %s", path)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._logger.info(
                "No repo config at %s; continuing with no linked repositories",
                path,
            )
            return RepoConfig(linked=[])
        except OSError as exc:
            raise ToolError(f"Could not read repo config {path}: {exc}") from exc

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"Repo config {path} is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})"
            ) from exc

        try:
            config = RepoConfig.model_validate(payload)
        except ValidationError as exc:
            raise ToolError(
                f"Malformed repo config at {path}: {exc}"
            ) from exc

        self._logger.info(
            "Loaded repo config: linked=%d",
            len(config.linked),
        )
        return config

    def list_files(
        self,
        local_repo_path: str,
        extensions: list[str] | None = None,
    ) -> list[str]:
        """List files under a local checkout, optionally filtered by extension.

        The ``.git`` directory created by clone is skipped because it is VCS
        metadata, not project source. No other directories or file types are
        excluded unless ``extensions`` is provided.

        Args:
            local_repo_path: Absolute or relative path to a local checkout.
            extensions: Optional suffixes to keep (with or without a leading
                dot, e.g. ``[".py", "ts"]``). ``None`` means every file.
                Comparison is case-insensitive.

        Returns:
            Sorted repository-relative paths using POSIX separators.

        Raises:
            ToolError: If ``local_repo_path`` is not a readable directory.
        """
        root = self._require_repo_dir(local_repo_path)
        allowed = self._normalize_extensions(extensions)
        results: list[str] = []

        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [name for name in dirnames if name != _GIT_DIR_NAME]
                for filename in filenames:
                    full = Path(dirpath) / filename
                    if allowed is not None and full.suffix.casefold() not in allowed:
                        continue
                    relative = full.relative_to(root).as_posix()
                    results.append(relative)
        except OSError as exc:
            raise ToolError(
                f"Failed to list files under {root}: {exc}"
            ) from exc

        results.sort()
        self._logger.debug("Listed %d file(s) under %s", len(results), root)
        return results

    def read_file(self, local_repo_path: str, relative_path: str) -> str:
        """Read a UTF-8 text file from a local checkout.

        Args:
            local_repo_path: Absolute or relative path to a local checkout.
            relative_path: Path inside the checkout. Must not escape the repo
                root (path traversal is rejected).

        Returns:
            The file contents as a string.

        Raises:
            ToolError: If the path is outside the repo, missing, unreadable,
                or not valid UTF-8.
        """
        root = self._require_repo_dir(local_repo_path)
        if not relative_path or not relative_path.strip():
            raise ToolError("relative_path must not be empty")

        target = (root / relative_path).resolve()
        if not self._is_inside(target, root):
            raise ToolError(
                f"Refusing to read {relative_path!r}: path escapes repository "
                f"root {root}"
            )
        if not target.is_file():
            raise ToolError(f"File not found: {relative_path} (under {root})")

        try:
            # utf-8-sig strips a leading BOM so downstream agents see clean text.
            content = target.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolError(
                f"File {relative_path!r} is not valid UTF-8: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolError(f"Could not read {relative_path!r}: {exc}") from exc

        self._logger.debug("Read %s (%d bytes)", relative_path, len(content))
        return content

    def search_code(self, local_repo_path: str, query: str) -> list[CodeMatch]:
        """Search a local checkout for ``query`` using ripgrep or grep.

        Prefers ``rg`` (ripgrep), then ``grep``, then ``git grep`` so the same
        tool works on Windows (where GNU grep is often absent but git is not).
        Hits are parsed into :class:`CodeMatch` instances; invalid rows are
        skipped with a warning rather than returned as raw dicts.

        Args:
            local_repo_path: Absolute or relative path to a local checkout.
            query: Literal search string (not a regular expression). Special
                characters are matched as written.

        Returns:
            Validated match objects, in the order produced by the search tool.

        Raises:
            ToolError: If the query is empty, no search backend is available,
                or the subprocess fails.
        """
        if not query or not query.strip():
            raise ToolError("search query must not be empty")
        query = query.strip()
        root = self._require_repo_dir(local_repo_path)

        backend, command = self._resolve_search_command(root, query)
        self._logger.info(
            "Searching %s for %r using %s", root, query, backend
        )
        result = self._run(
            command,
            cwd=str(root),
            timeout=_SEARCH_TIMEOUT_SECONDS,
            allowed_returncodes=frozenset({0, 1}),
            action=f"{backend} search {query!r}",
        )
        if result.returncode == 1 or not result.stdout.strip():
            self._logger.info("No matches for %r in %s", query, root)
            return []

        if backend == "rg":
            matches = self._parse_rg_json(result.stdout, root)
        else:
            matches = self._parse_grep_output(result.stdout, root)

        self._logger.info("Found %d match(es) for %r in %s", len(matches), query, root)
        return matches

    def _split_slug(self, repo_full_name: str) -> tuple[str, str]:
        """Split ``owner/repository`` into its two segments.

        Raises:
            ToolError: If the slug is not exactly two non-empty parts.
        """
        slug = repo_full_name.strip()
        parts = slug.split("/")
        if len(parts) != 2 or not all(parts):
            raise ToolError(
                "repo_full_name must be in 'owner/repository' format, "
                f"got: {repo_full_name!r}"
            )
        return parts[0], parts[1]

    def _require_repo_dir(self, local_repo_path: str) -> Path:
        """Resolve ``local_repo_path`` and require it to be an existing directory."""
        try:
            root = Path(local_repo_path).expanduser().resolve()
        except OSError as exc:
            raise ToolError(
                f"Could not resolve local_repo_path {local_repo_path!r}: {exc}"
            ) from exc
        if not root.is_dir():
            raise ToolError(
                f"local_repo_path is not a directory: {local_repo_path}"
            )
        return root

    def _is_git_checkout(self, path: Path) -> bool:
        """Return True if ``path`` looks like a git working tree."""
        return (path / _GIT_DIR_NAME).exists()

    def _is_inside(self, candidate: Path, root: Path) -> bool:
        """Return True if ``candidate`` is ``root`` or a descendant of it."""
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False

    def _normalize_extensions(self, extensions: list[str] | None) -> set[str] | None:
        """Turn caller-supplied suffixes into a case-folded set including the dot.

        Returns:
            ``None`` when every file should be listed; otherwise the allowed
            suffixes (e.g. ``{".py", ".ts"}``).
        """
        if extensions is None:
            return None
        allowed: set[str] = set()
        for raw in extensions:
            item = raw.strip()
            if not item:
                continue
            if not item.startswith("."):
                item = f".{item}"
            allowed.add(item.casefold())
        return allowed

    def _git_env(self) -> dict[str, str]:
        """Environment for git subprocesses that must not prompt for credentials."""
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GCM_INTERACTIVE"] = "Never"
        return env

    def _resolve_search_command(
        self,
        root: Path,
        query: str,
    ) -> tuple[str, list[str]]:
        """Pick ripgrep, GNU grep, or git-grep and build its argv.

        Returns:
            A ``(backend_name, argv)`` pair. ``argv`` is ready to run with
            ``cwd`` set to ``root``.

        Raises:
            ToolError: If none of the supported search executables are found.
        """
        rg = shutil.which("rg")
        if rg:
            return (
                "rg",
                [
                    rg,
                    "--json",
                    "--fixed-strings",
                    "--no-ignore",
                    "--hidden",
                    "--glob",
                    "!.git",
                    "--glob",
                    "!.git/**",
                    "-e",
                    query,
                    ".",
                ],
            )

        grep = self._find_grep()
        if grep:
            return (
                "grep",
                [
                    grep,
                    "-R",
                    "-n",
                    "-I",
                    "-H",
                    "-F",
                    "--exclude-dir=.git",
                    "-e",
                    query,
                    ".",
                ],
            )

        git = shutil.which("git")
        if git and self._is_git_checkout(root):
            return (
                "git-grep",
                [
                    git,
                    "grep",
                    "-n",
                    "-I",
                    "-F",
                    "--untracked",
                    "-e",
                    query,
                    "--",
                ],
            )

        raise ToolError(
            "No code-search backend found. Install ripgrep (`rg`) or GNU grep, "
            "or run search_code on a git checkout so `git grep` can be used."
        )

    def _find_grep(self) -> str | None:
        """Locate a GNU-compatible grep executable, including Git for Windows."""
        found = shutil.which("grep")
        if found:
            return found
        candidates = (
            Path(r"C:\Program Files\Git\usr\bin\grep.exe"),
            Path(r"C:\Program Files (x86)\Git\usr\bin\grep.exe"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    def _parse_rg_json(self, stdout: str, root: Path) -> list[CodeMatch]:
        """Turn ripgrep ``--json`` lines into validated :class:`CodeMatch` rows."""
        matches: list[CodeMatch] = []
        for raw_line in stdout.splitlines():
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                self._logger.warning("Skipping unparseable ripgrep JSON line")
                continue
            if payload.get("type") != "match":
                continue
            data = payload.get("data") or {}
            path_info = data.get("path") or {}
            lines_info = data.get("lines") or {}
            file_path = path_info.get("text")
            line_number = data.get("line_number")
            line_text = lines_info.get("text")
            if not isinstance(file_path, str) or not isinstance(line_number, int):
                self._logger.warning("Skipping incomplete ripgrep match: %s", payload)
                continue
            if not isinstance(line_text, str):
                line_text = ""
            match = self._to_code_match(file_path, line_number, line_text, root)
            if match is not None:
                matches.append(match)
        return matches

    def _parse_grep_output(self, stdout: str, root: Path) -> list[CodeMatch]:
        """Turn ``file:line_number:line`` grep/git-grep output into CodeMatch rows."""
        matches: list[CodeMatch] = []
        for raw_line in stdout.splitlines():
            parsed = _GREP_HIT_PATTERN.match(raw_line)
            if parsed is None:
                self._logger.warning("Skipping unparseable grep line: %r", raw_line)
                continue
            match = self._to_code_match(
                parsed.group("file"),
                int(parsed.group("line_number")),
                parsed.group("line"),
                root,
            )
            if match is not None:
                matches.append(match)
        return matches

    def _to_code_match(
        self,
        file_path: str,
        line_number: int,
        line: str,
        root: Path,
    ) -> CodeMatch | None:
        """Validate one hit, returning None (and logging) if it is malformed."""
        relative = file_path.replace("\\", "/")
        if relative.startswith("./"):
            relative = relative[2:]
        try:
            return CodeMatch(
                file=relative,
                line_number=line_number,
                line=line.rstrip("\n\r"),
            )
        except ValidationError as exc:
            self._logger.warning(
                "Skipping invalid search hit under %s (%s): %s",
                root,
                file_path,
                exc,
            )
            return None

    def _run(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int,
        allowed_returncodes: frozenset[int] = frozenset({0}),
        secret: str | None = None,
        action: str,
    ) -> subprocess.CompletedProcess[str]:
        """Run an external process and map failures onto :class:`ToolError`.

        Args:
            args: Argument vector (never passed through a shell).
            cwd: Optional working directory for the process.
            env: Optional environment mapping.
            timeout: Seconds before the process is killed.
            allowed_returncodes: Exit codes treated as success (grep uses 1
                for "no matches").
            secret: Optional string (e.g. a token) stripped from error text.
            action: Human-readable description included in ``ToolError``.

        Returns:
            The completed process, with stdout/stderr captured as text.

        Raises:
            ToolError: On missing executable, timeout, OS error, or a
                disallowed exit code. The original exception is chained.
        """
        executable = args[0]
        try:
            completed = subprocess.run(
                args,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolError(
                f"{action} failed: executable {executable!r} was not found on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"{action} timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise ToolError(f"{action} failed to start: {exc}") from exc

        if completed.returncode not in allowed_returncodes:
            detail = (completed.stderr or completed.stdout or "").strip()
            if secret:
                detail = detail.replace(secret, "***")
            if detail:
                message = f"{action} failed (exit {completed.returncode}): {detail}"
            else:
                message = f"{action} failed (exit {completed.returncode})"
            raise ToolError(message)

        return completed
