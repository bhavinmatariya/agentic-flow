"""Automated backend/frontend checks for the reviewer pipeline.

Detects pytest and npm/tsc tooling in a checkout, runs what is configured,
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

_FRONTEND_DIR_NAMES: Final[tuple[str, ...]] = (
    "",
    "frontend",
    "client",
    "web",
    "ui",
    "app",
)
_OUTPUT_TAIL_CHARS: Final[int] = 4000
_KEY_ERROR_LINES: Final[int] = 12

logger = get_logger(__name__)


@dataclass
class AutomatedCheckResult:
    """Outcome of :func:`run_automated_checks`."""

    findings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    layers: dict[str, bool] = field(default_factory=dict)


def detect_layers(local_repo_path: str) -> dict[str, Any]:
    """Detect whether backend (pytest) and frontend (npm/tsc) checks can run."""
    root = Path(local_repo_path).expanduser().resolve()
    backend = _has_pytest_setup(root)
    frontend_dirs = _find_frontend_package_dirs(root)
    return {
        "backend": backend,
        "frontend": bool(frontend_dirs),
        "frontend_package_dirs": [str(path) for path in frontend_dirs],
    }


def run_automated_checks(local_repo_path: str) -> AutomatedCheckResult:
    """Run configured automated checks and return findings (never raises on failure)."""
    root = Path(local_repo_path).expanduser().resolve()
    if not root.is_dir():
        return AutomatedCheckResult(
            findings=[f"Automated checks skipped: checkout path missing: {root}"],
            layers={"backend": False, "frontend": False},
        )

    detection = detect_layers(str(root))
    findings: list[str] = []
    skipped: list[str] = []

    if detection["backend"]:
        pytest_finding = _run_pytest(root)
        if pytest_finding:
            findings.append(pytest_finding)
    else:
        skipped.append("backend: skipped: no tests configured")

    if detection["frontend"]:
        for package_dir in detection["frontend_package_dirs"]:
            package_path = Path(package_dir)
            layer_findings, layer_skipped = _run_frontend_checks(package_path)
            findings.extend(layer_findings)
            skipped.extend(layer_skipped)
    else:
        skipped.append("frontend: skipped: no tests configured")

    logger.info(
        "Automated checks in %s: findings=%d skipped=%d layers=%s",
        root,
        len(findings),
        len(skipped),
        detection,
    )
    return AutomatedCheckResult(
        findings=findings,
        skipped=skipped,
        layers={
            "backend": bool(detection["backend"]),
            "frontend": bool(detection["frontend"]),
        },
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


def _find_frontend_package_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for name in _FRONTEND_DIR_NAMES:
        candidate = (root / name).resolve() if name else root
        package_json = candidate / "package.json"
        if package_json.is_file() and candidate not in seen:
            dirs.append(candidate)
            seen.add(candidate)
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

    checks: list[tuple[str, str]] = []
    if isinstance(scripts.get("build"), str) and scripts["build"].strip():
        checks.append(("npm run build", "Build failed"))
    if isinstance(scripts.get("test"), str) and scripts["test"].strip():
        checks.append(("npm test --if-present", "Test failed"))
    if not checks and (package_dir / "tsconfig.json").is_file():
        checks.append(("npx tsc --noEmit", "Typecheck failed"))

    if not checks:
        skipped.append(f"{rel}: skipped: no tests configured")
        return findings, skipped

    npm = shutil.which("npm")
    if npm is None:
        skipped.append(f"{rel}: skipped: npm is not available in this environment")
        return findings, skipped

    for command, prefix in checks:
        completed = _run_subprocess(command, cwd=package_dir, label=command, shell=True)
        if completed is None:
            skipped.append(f"{rel}: skipped: could not run {command!r}")
            continue
        if completed.returncode == 0:
            continue
        output = _combined_output(completed)
        findings.append(f"{prefix} ({rel}): {_extract_key_lines(output)}")

    return findings, skipped


def _run_subprocess(
    command: str | list[str],
    *,
    cwd: Path,
    label: str,
    shell: bool = False,
    timeout_seconds: int = 600,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            shell=shell,
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
            args=command if isinstance(command, list) else [command],
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
