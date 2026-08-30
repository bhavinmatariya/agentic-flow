"""Automated backend/frontend checks for the reviewer pipeline.

Runs pytest and npm/tsc tooling only for layers touched by ``files_changed``,
and returns human-readable findings instead of raising on test/build failures.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from utils.logger import get_logger

_FRONTEND_MARKERS: Final[tuple[str, ...]] = (
    "frontend/",
    "client/",
    "web/",
    "ui/",
    "src/components/",
    "pages/",
    "app/",
)
_FRONTEND_EXTENSIONS: Final[tuple[str, ...]] = (
    ".tsx",
    ".jsx",
    ".vue",
    ".html",
    ".css",
    ".scss",
)
_DB_MARKERS: Final[tuple[str, ...]] = (
    "migration",
    "migrations/",
    "alembic/",
    "schema.sql",
    "models.py",
    "model.py",
    "database/",
    "db/",
    "prisma/",
    "sequelize/",
    ".sql",
)
_OUTPUT_TAIL_CHARS: Final[int] = 4000
_KEY_ERROR_LINES: Final[int] = 12

logger = get_logger(__name__)


@dataclass
class AutomatedCheckResult:
    """Outcome of :func:`run_automated_checks`."""

    findings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    layers_relevant: dict[str, bool] = field(default_factory=dict)
    layers_run: dict[str, bool] = field(default_factory=dict)


def detect_change_layers(files_changed: list[str]) -> dict[str, bool]:
    """Detect whether ``files_changed`` touches frontend, database, or backend layers."""
    frontend = False
    database = False
    backend = False
    for raw_path in files_changed:
        path = raw_path.replace("\\", "/").lower()
        if any(marker in path for marker in _FRONTEND_MARKERS) or path.endswith(
            _FRONTEND_EXTENSIONS
        ):
            frontend = True
        if any(marker in path for marker in _DB_MARKERS):
            database = True
        if path.endswith((".py", ".go", ".rs", ".java")):
            backend = True
        if any(marker in path for marker in ("api/", "server/", "backend/", "routes/")):
            backend = True
    return {"frontend": frontend, "database": database, "backend": backend}


def run_automated_checks(
    local_repo_path: str,
    files_changed: list[str],
) -> AutomatedCheckResult:
    """Run automated checks scoped to layers touched by ``files_changed``.

    Never raises on test/build failures; returns findings instead.
    """
    root = Path(local_repo_path).expanduser().resolve()
    layers_relevant = detect_change_layers(files_changed)
    layers_run = {"backend": False, "frontend": False}
    findings: list[str] = []
    skipped: list[str] = []

    if not root.is_dir():
        return AutomatedCheckResult(
            findings=[f"Automated checks skipped: checkout path missing: {root}"],
            skipped=[
                "backend: skipped: checkout path missing",
                "frontend: skipped: checkout path missing",
            ],
            layers_relevant=layers_relevant,
            layers_run=layers_run,
        )

    if not files_changed:
        skipped.append("backend: skipped: no files changed")
        skipped.append("frontend: skipped: no relevant files changed")
    elif layers_relevant["backend"]:
        if _has_pytest_setup(root):
            layers_run["backend"] = True
            pytest_finding = _run_pytest(root)
            if pytest_finding:
                findings.append(pytest_finding)
        else:
            skipped.append("backend: skipped: no tests configured")
    else:
        skipped.append("backend: skipped: no relevant files changed")

    if files_changed and layers_relevant["frontend"]:
        package_dirs = _frontend_package_dirs_for_changed_files(root, files_changed)
        if not package_dirs:
            skipped.append(
                "frontend: skipped: no frontend package directory for changed files"
            )
        else:
            layers_run["frontend"] = True
            for package_dir in package_dirs:
                layer_findings, layer_skipped = _run_frontend_checks(package_dir)
                findings.extend(layer_findings)
                skipped.extend(layer_skipped)
    elif files_changed:
        skipped.append("frontend: skipped: no relevant files changed")

    logger.info(
        "Automated checks in %s: files_changed=%d findings=%d skipped=%d "
        "layers_relevant=%s layers_run=%s",
        root,
        len(files_changed),
        len(findings),
        len(skipped),
        layers_relevant,
        layers_run,
    )
    return AutomatedCheckResult(
        findings=findings,
        skipped=skipped,
        layers_relevant=layers_relevant,
        layers_run=layers_run,
    )


def _has_pytest_setup(root: Path) -> bool:
    if (root / "pytest.ini").is_file():
        return True
    if (root / "conftest.py").is_file():
        return True
    if _tests_directory_has_files(root / "tests"):
        return True
    if _tests_directory_has_files(root / "test"):
        return True

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8", errors="replace").lower()
        if "[tool.pytest" in text or "pytest" in text:
            return True

    for name in ("requirements.txt", "requirements-dev.txt", "dev-requirements.txt"):
        req_path = root / name
        if req_path.is_file() and "pytest" in req_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).lower():
            return True

    return False


def _tests_directory_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    for candidate in path.rglob("test_*.py"):
        if candidate.is_file():
            return True
    for candidate in path.rglob("*_test.py"):
        if candidate.is_file():
            return True
    return False


def _frontend_package_dirs_for_changed_files(
    root: Path,
    files_changed: list[str],
) -> list[Path]:
    """Return package.json roots that contain at least one changed file."""
    root = root.resolve()
    seen: set[Path] = set()
    dirs: list[Path] = []
    for raw_path in files_changed:
        normalized = raw_path.replace("\\", "/").lstrip("/")
        if not normalized:
            continue
        candidate_path = root / normalized
        current = candidate_path.parent if normalized else root
        while True:
            try:
                current.relative_to(root)
            except ValueError:
                break
            package_json = current / "package.json"
            if package_json.is_file():
                resolved = current.resolve()
                if resolved not in seen:
                    dirs.append(resolved)
                    seen.add(resolved)
                break
            if current == root:
                break
            current = current.parent
    return dirs


def _run_pytest(root: Path) -> str | None:
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--maxfail=1"]
    completed = _run_subprocess(command, cwd=root, label="pytest")
    if completed is None:
        return "pytest: skipped: pytest is not available in this environment"
    if completed.returncode == 0:
        return None
    output = _combined_output(completed)
    return f"pytest failed: {_extract_key_lines(output)}"


def _run_frontend_checks(package_dir: Path) -> tuple[list[str], list[str]]:
    package_json_path = package_dir / "package.json"
    try:
        payload = json.loads(package_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ([f"frontend check failed: could not read {package_json_path}: {exc}"], [])

    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        scripts = {}

    rel = _display_rel(package_dir)
    findings: list[str] = []
    skipped: list[str] = []

    checks: list[tuple[list[str], str]] = []
    if isinstance(scripts.get("build"), str) and scripts["build"].strip():
        checks.append((["npm", "run", "build"], "Build failed"))
    if isinstance(scripts.get("test"), str) and scripts["test"].strip():
        checks.append((["npm", "test", "--if-present"], "Test failed"))
    if not checks and (package_dir / "tsconfig.json").is_file():
        npx = shutil.which("npx")
        if npx:
            checks.append(([npx, "tsc", "--noEmit"], "Typecheck failed"))

    if not checks:
        skipped.append(f"{rel}: skipped: no tests configured")
        return findings, skipped

    npm = shutil.which("npm")
    if npm is None:
        skipped.append(f"{rel}: skipped: npm is not available in this environment")
        return findings, skipped

    for argv, prefix in checks:
        if argv[0] == "npm":
            command = [npm, *argv[1:]]
        else:
            command = argv
        completed = _run_subprocess(command, cwd=package_dir, label=" ".join(command))
        if completed is None:
            skipped.append(f"{rel}: skipped: could not run {' '.join(command)!r}")
            continue
        if completed.returncode == 0:
            continue
        output = _combined_output(completed)
        findings.append(f"{prefix} ({rel}): {_extract_key_lines(output)}")

    return findings, skipped


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    label: str,
    timeout_seconds: int = 600,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("Automated check %r: executable not found", label)
        return None
    except subprocess.TimeoutExpired as exc:
        output = _combined_output(exc)
        logger.warning("Automated check %r timed out after %ss", label, timeout_seconds)
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=output,
            stderr="",
        )
    except OSError as exc:
        logger.warning("Automated check %r failed to start: %s", label, exc)
        return None


def _combined_output(result: subprocess.CompletedProcess[str] | subprocess.TimeoutExpired) -> str:
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    combined = f"{stdout}\n{stderr}".strip()
    if len(combined) > _OUTPUT_TAIL_CHARS:
        combined = combined[-_OUTPUT_TAIL_CHARS:]
    return combined


def _extract_key_lines(output: str) -> str:
    if not output.strip():
        return "(no output captured)"
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return output.strip()[:500]
    key_lines = lines[-_KEY_ERROR_LINES:]
    return " | ".join(key_lines)


def _display_rel(path: Path) -> str:
    name = path.name or str(path)
    return name if path.parent.name in {"", path.drive} else f"{path.parent.name}/{name}"
