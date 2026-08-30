"""Disposable local test environments for full-stack live verification."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Final

from core.exceptions import EnvironmentSetupError
from utils.logger import get_logger

_DB_IMPORT_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "postgres": ("psycopg2", "asyncpg", "psycopg", "postgresql"),
    "mysql": ("mysql.connector", "pymysql", "aiomysql", "MySQLdb"),
    "mongo": ("pymongo", "motor"),
    "sqlite": ("sqlite3",),
}
_DB_DRIVER_PACKAGES: Final[dict[str, tuple[str, str] | None]] = {
    "postgres": ("psycopg2", "psycopg2-binary"),
    "postgresql": ("psycopg2", "psycopg2-binary"),
    "mysql": ("mysql.connector", "mysql-connector-python"),
    "mongo": ("pymongo", "pymongo"),
    "mongodb": ("pymongo", "pymongo"),
    "sqlite": None,
}
_BACKEND_COMMAND_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"uvicorn\s+[\w.:]+(?:\s+[\w.-]+)*", re.I), "uvicorn"),
    (re.compile(r"flask\s+run(?:\s+[\w.-]+)*", re.I), "flask run"),
    (re.compile(r"gunicorn\s+[\w.:]+(?:\s+[\w.-]+)*", re.I), "gunicorn"),
    (
        re.compile(r"python\s+manage\.py\s+runserver(?:\s+[\w.:]+)*", re.I),
        "django runserver",
    ),
    (re.compile(r"npm\s+run\s+start:server", re.I), "npm run start:server"),
)
_ENV_VAR_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"""os\.getenv\(\s*['"]([A-Z0-9_]+)['"]"""),
    re.compile(r"""os\.environ\.(?:get|\[)\(\s*['"]([A-Z0-9_]+)['"]"""),
    re.compile(r"""os\.environ\[['"]([A-Z0-9_]+)['"]\]"""),
    re.compile(r"""process\.env\.(?:\[)?['"]([A-Z0-9_]+)['"]"""),
)
_EXTERNAL_CREDENTIAL_MARKERS: Final[tuple[str, ...]] = (
    "API_KEY",
    "SECRET",
    "OAUTH",
    "TOKEN",
    "PASSWORD",
)
_DB_ENV_MARKERS: Final[tuple[str, ...]] = (
    "DATABASE",
    "DB_",
    "POSTGRES",
    "MONGO",
    "MYSQL",
    "SQLITE",
    "REDIS_URL",
)
_DEFAULT_READY_TIMEOUT: Final[int] = 90
_DOCKER_WAIT_SECONDS: Final[int] = 60
_MIN_DB_SCORE: Final[int] = 2
_MIN_CONFIDENCE_RATIO: Final[float] = 1.5
DEFAULT_TEST_USER_EMAIL: Final[str] = "agentic-flow-test@example.com"
_COMMON_SIGNUP_PATHS: Final[tuple[str, ...]] = (
    "/api/auth/register",
    "/api/register",
    "/api/signup",
    "/api/users",
    "/auth/register",
    "/register",
    "/signup",
)
_SIGNUP_ROUTE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""['"](/[^'"]*(?:register|signup)[^'"]*)['"]""",
    re.IGNORECASE,
)
_USER_TABLE_CANDIDATES: Final[tuple[str, ...]] = (
    "users",
    "user",
    "accounts",
    "auth_user",
    "app_user",
)
_EMAIL_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "email",
    "username",
    "user_email",
)
_PASSWORD_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "password",
    "password_hash",
    "hashed_password",
    "passwd",
)


def ensure_db_driver(db_type: str, logger: logging.Logger | None = None) -> None:
    """Install the database driver for ``db_type`` when it is not importable.

    Args:
        db_type: One of ``postgres``, ``mysql``, ``mongo``, or ``sqlite``.

    Raises:
        EnvironmentSetupError: If the driver cannot be imported or installed.
    """
    log = logger or get_logger(__name__)
    normalized = db_type.strip().lower()
    package_spec = _DB_DRIVER_PACKAGES.get(normalized)
    if package_spec is None:
        if normalized in {"sqlite", ""}:
            return
        raise EnvironmentSetupError(
            f"Unsupported db_type for driver install: {db_type!r}"
        )

    module_name, pip_package = package_spec
    if importlib.util.find_spec(module_name) is not None:
        return

    log.info(
        "Database driver %r not importable; installing %r",
        module_name,
        pip_package,
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_package],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.stdout.strip():
            log.debug("pip install stdout: %s", completed.stdout.strip())
    except subprocess.CalledProcessError as exc:
        raise EnvironmentSetupError(
            "Failed to install database driver "
            f"{pip_package!r} for db_type={db_type!r}: "
            f"{exc.stderr or exc.stdout or exc}"
        ) from exc
    except Exception as exc:
        raise EnvironmentSetupError(
            f"Failed to install database driver for db_type={db_type!r}: {exc}"
        ) from exc

    if importlib.util.find_spec(module_name) is None:
        raise EnvironmentSetupError(
            f"Database driver {module_name!r} still not importable after "
            f"installing {pip_package!r}"
        )


class EnvironmentManager:
    """Detect stack layout and spin up isolated backend/frontend test environments."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Create a manager with optional logging."""
        self._logger = logger or get_logger(__name__)
        self._docker_container_id: str | None = None
        self._sqlite_path: Path | None = None

    def detect_stack(self, local_repo_path: str) -> dict[str, Any]:
        """Infer database, backend, frontend, and env-var requirements from the repo.

        Args:
            local_repo_path: Absolute path to a local repository checkout.

        Returns:
            Dictionary with ``db_type``, confidence fields, ``backend_start_command``,
            ``frontend_start_command``, ``frontend_port``, and
            ``required_env_vars`` keys.

        Raises:
            EnvironmentSetupError: If DB type or backend start command cannot be
                identified with reasonable confidence.
        """
        root = Path(local_repo_path).expanduser().resolve()
        if not root.is_dir():
            raise EnvironmentSetupError(
                f"detect_stack failed: local_repo_path does not exist: {root}"
            )

        try:
            db_type, db_confidence = self._detect_db_type_with_confidence(root)
            backend_command, backend_confidence = self._detect_backend_command_with_confidence(
                root
            )
            frontend_command, frontend_port = self._detect_frontend_command(root)
            env_vars = sorted(self._detect_env_vars(root))

            if db_type is None or db_confidence == "low":
                raise EnvironmentSetupError(
                    "detect_stack: cannot identify database type with reasonable "
                    f"confidence (db_type={db_type!r}, confidence={db_confidence})"
                )
            if backend_command is None or backend_confidence == "low":
                raise EnvironmentSetupError(
                    "detect_stack: cannot identify backend start command with "
                    f"reasonable confidence (command={backend_command!r}, "
                    f"confidence={backend_confidence})"
                )

            result = {
                "db_type": db_type,
                "db_type_confidence": db_confidence,
                "backend_start_command": backend_command,
                "backend_command_confidence": backend_confidence,
                "frontend_start_command": frontend_command,
                "frontend_port": frontend_port,
                "required_env_vars": env_vars,
            }
            self._logger.info("Detected stack in %s: %s", root, result)
            return result
        except EnvironmentSetupError:
            raise
        except Exception as exc:
            raise EnvironmentSetupError(f"detect_stack failed for {root}: {exc}") from exc

    def start_test_database(self, db_type: str) -> str:
        """Start a disposable database and return its connection string.

        Args:
            db_type: One of ``postgres``, ``mysql``, ``mongo``, or ``sqlite``.

        Returns:
            A connection string pointing only at the isolated test database.

        Raises:
            EnvironmentSetupError: If Docker is unavailable or the DB fails to start.
        """
        normalized = db_type.strip().lower()
        try:
            ensure_db_driver(normalized, self._logger)
            if normalized == "sqlite":
                return self._start_sqlite_database()
            if normalized == "postgres":
                return self._start_docker_database(
                    image="postgres:16-alpine",
                    internal_port=5432,
                    env={
                        "POSTGRES_USER": "agentic_test",
                        "POSTGRES_PASSWORD": "agentic_test",
                        "POSTGRES_DB": "agentic_test",
                    },
                    connection_builder=self._postgres_connection_string,
                    readiness_check=self._wait_for_tcp,
                )
            if normalized == "mysql":
                return self._start_docker_database(
                    image="mysql:8",
                    internal_port=3306,
                    env={
                        "MYSQL_ROOT_PASSWORD": "agentic_test",
                        "MYSQL_DATABASE": "agentic_test",
                        "MYSQL_USER": "agentic_test",
                        "MYSQL_PASSWORD": "agentic_test",
                    },
                    connection_builder=self._mysql_connection_string,
                    readiness_check=self._wait_for_tcp,
                )
            if normalized in {"mongo", "mongodb"}:
                return self._start_docker_database(
                    image="mongo:7",
                    internal_port=27017,
                    env={},
                    connection_builder=self._mongo_connection_string,
                    readiness_check=self._wait_for_tcp,
                )
            raise EnvironmentSetupError(
                f"start_test_database failed: unsupported db_type {db_type!r}"
            )
        except EnvironmentSetupError:
            raise
        except Exception as exc:
            raise EnvironmentSetupError(
                f"start_test_database failed for db_type={db_type!r}: {exc}"
            ) from exc

    def run_migrations(self, local_repo_path: str, connection_string: str) -> bool:
        """Run a detected migration command, if one exists.

        Args:
            local_repo_path: Repository checkout path.
            connection_string: Disposable database connection string.

        Returns:
            ``True`` when a migration command ran successfully, ``False`` when
            none was detected (logged and skipped).

        Raises:
            EnvironmentSetupError: If a migration command was detected but failed.
        """
        root = Path(local_repo_path).expanduser().resolve()
        env = os.environ.copy()
        env.update(self.generate_dummy_env([], connection_string))
        db_type = self._infer_db_type_from_connection(connection_string)
        if db_type:
            ensure_db_driver(db_type, self._logger)

        try:
            if (root / "alembic.ini").exists():
                self._logger.info("Running Alembic migrations in %s", root)
                self._run_checked(
                    ["alembic", "upgrade", "head"],
                    cwd=str(root),
                    env=env,
                    step="run_migrations:alembic",
                )
                return True

            manage_py = root / "manage.py"
            if manage_py.exists():
                self._logger.info("Running Django migrations in %s", root)
                self._run_checked(
                    ["python", "manage.py", "migrate", "--noinput"],
                    cwd=str(root),
                    env=env,
                    step="run_migrations:django",
                )
                return True

            schema_sql = self._find_schema_sql(root)
            if schema_sql is not None:
                self._logger.info("Applying schema SQL from %s", schema_sql)
                self._apply_schema_sql(schema_sql, connection_string)
                return True

            self._logger.info(
                "run_migrations: no Alembic, manage.py migrate, or schema.sql found in %s; continuing",
                root,
            )
            return False
        except EnvironmentSetupError:
            raise
        except Exception as exc:
            raise EnvironmentSetupError(f"run_migrations failed for {root}: {exc}") from exc

    def generate_dummy_env(
        self,
        detected_env_vars: list[str],
        db_connection_string: str,
    ) -> dict[str, str]:
        """Build placeholder env vars so an app can boot without real credentials."""
        env: dict[str, str] = {
            "DATABASE_URL": db_connection_string,
            "DB_CONNECTION_STRING": db_connection_string,
        }
        for var in detected_env_vars:
            upper = var.upper()
            if any(marker in upper for marker in _DB_ENV_MARKERS):
                env[var] = db_connection_string
            elif any(marker in upper for marker in _EXTERNAL_CREDENTIAL_MARKERS):
                if "DB" not in upper and "DATABASE" not in upper:
                    env[var] = "test-placeholder-value"
            else:
                env[var] = "test-placeholder-value"
        return env

    def start_process(
        self,
        command: str,
        cwd: str,
        env: dict[str, str],
        ready_url: str,
        timeout: int = _DEFAULT_READY_TIMEOUT,
    ) -> subprocess.Popen[str]:
        """Start a subprocess and wait until ``ready_url`` responds."""
        workdir = Path(cwd).expanduser().resolve()
        if not workdir.is_dir():
            raise EnvironmentSetupError(
                f"start_process failed: cwd does not exist: {workdir}"
            )

        merged_env = os.environ.copy()
        merged_env.update(env)
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                shlex.split(command),
                cwd=str(workdir),
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._wait_for_http(ready_url, timeout=timeout, process=process)
            self._logger.info(
                "Process ready for %s (pid=%s, command=%r)",
                ready_url,
                process.pid,
                command,
            )
            return process
        except EnvironmentSetupError:
            self._terminate_process(process)
            raise
        except Exception as exc:
            self._terminate_process(process)
            raise EnvironmentSetupError(
                f"start_process failed for command={command!r} cwd={workdir}: {exc}"
            ) from exc

    def stop_process(self, handle: subprocess.Popen[Any] | None) -> None:
        """Terminate a process started by :meth:`start_process`."""
        self._terminate_process(handle)

    def stop_test_database(self) -> None:
        """Stop the disposable Docker container or delete the sqlite temp file."""
        if self._docker_container_id:
            container_id = self._docker_container_id
            self._docker_container_id = None
            try:
                subprocess.run(
                    ["docker", "stop", container_id],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self._logger.info("Stopped disposable database container %s", container_id)
            except Exception as exc:
                self._logger.warning(
                    "stop_test_database failed while stopping container %s: %s",
                    container_id,
                    exc,
                )

        if self._sqlite_path and self._sqlite_path.exists():
            try:
                self._sqlite_path.unlink(missing_ok=True)
                self._logger.info("Removed sqlite test database %s", self._sqlite_path)
            except OSError as exc:
                self._logger.warning(
                    "stop_test_database failed while deleting %s: %s",
                    self._sqlite_path,
                    exc,
                )
            finally:
                self._sqlite_path = None

    def seed_test_user(
        self,
        local_repo_path: str,
        *,
        frontend_url: str,
        backend_url: str | None,
        connection_string: str,
        db_type: str,
        email: str = DEFAULT_TEST_USER_EMAIL,
        password: str | None = None,
    ) -> dict[str, Any]:
        """Create a disposable test user for authenticated UI verification.

        Prefers the app's own signup/register HTTP endpoint when one can be
        discovered or guessed. Falls back to a direct insert into a users
        table on the disposable database only when signup is unavailable.

        Returns:
            Dict with ``seeded``, ``skipped``, credentials, URLs, ``method``,
            and ``playwright_guidance`` for script generation. When seeding
            fails, includes ``ui_verification_if_skipped`` for the reviewer.
        """
        from tools.browser_test import (
            SKIPPED_AUTH_UI_VERIFICATION,
            playwright_script_guidance,
        )

        root = Path(local_repo_path).expanduser().resolve()
        if not root.is_dir():
            raise EnvironmentSetupError(
                f"seed_test_user failed: local_repo_path does not exist: {root}"
            )

        resolved_password = password or f"agentic-flow-test-{uuid.uuid4().hex[:16]}"
        api_base = (backend_url or frontend_url).rstrip("/")
        discovered_paths = self._discover_signup_paths(root)
        signup_paths = list(dict.fromkeys([*discovered_paths, *_COMMON_SIGNUP_PATHS]))

        for path in signup_paths:
            if self._try_http_signup(api_base, path, email, resolved_password):
                self._logger.info(
                    "Seeded test user via signup endpoint %s%s",
                    api_base,
                    path,
                )
                return self._seed_result(
                    seeded=True,
                    email=email,
                    password=resolved_password,
                    frontend_url=frontend_url,
                    backend_url=backend_url,
                    method="signup_endpoint",
                    signup_path=path,
                    playwright_guidance=playwright_script_guidance(
                        base_url=frontend_url,
                        test_email=email,
                        test_password=resolved_password,
                    ),
                )

        if self._try_db_user_insert(
            connection_string,
            db_type,
            root,
            email,
            resolved_password,
        ):
            self._logger.info(
                "Seeded test user via direct DB insert for %r",
                email,
            )
            return self._seed_result(
                seeded=True,
                email=email,
                password=resolved_password,
                frontend_url=frontend_url,
                backend_url=backend_url,
                method="db_insert",
                playwright_guidance=playwright_script_guidance(
                    base_url=frontend_url,
                    test_email=email,
                    test_password=resolved_password,
                ),
            )

        self._logger.warning(
            "Could not seed test user for live UI verification in %s",
            root,
        )
        return {
            "seeded": False,
            "skipped": True,
            "skip_reason": "could not establish authenticated session",
            "email": email,
            "password": None,
            "frontend_url": frontend_url,
            "backend_url": backend_url,
            "method": None,
            "playwright_guidance": playwright_script_guidance(
                base_url=frontend_url,
                test_email=None,
                test_password=None,
            ),
            "ui_verification_if_skipped": dict(SKIPPED_AUTH_UI_VERIFICATION),
        }

    @staticmethod
    def _seed_result(
        *,
        seeded: bool,
        email: str,
        password: str,
        frontend_url: str,
        backend_url: str | None,
        method: str,
        playwright_guidance: str,
        signup_path: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seeded": seeded,
            "skipped": False,
            "skip_reason": None,
            "email": email,
            "password": password,
            "frontend_url": frontend_url,
            "backend_url": backend_url,
            "method": method,
            "playwright_guidance": playwright_guidance,
        }
        if signup_path is not None:
            payload["signup_path"] = signup_path
        return payload

    def _discover_signup_paths(self, root: Path) -> list[str]:
        """Scan the repo for register/signup route strings."""
        paths: set[str] = set()
        for path in self._iter_source_files(root):
            if path.suffix.lower() not in {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rb"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in _SIGNUP_ROUTE_PATTERN.finditer(text):
                candidate = match.group(1).split("?")[0].strip()
                if candidate.startswith("/"):
                    paths.add(candidate)
        return sorted(paths)

    def _try_http_signup(
        self,
        api_base: str,
        path: str,
        email: str,
        password: str,
    ) -> bool:
        """POST signup payloads to ``api_base + path``; return True on success."""
        url = f"{api_base.rstrip('/')}{path}"
        payloads: list[dict[str, Any]] = [
            {"email": email, "password": password},
            {"username": email, "email": email, "password": password},
            {"name": "Agentic Flow Test", "email": email, "password": password},
            {"email": email, "password": password, "password_confirmation": password},
        ]
        for payload in payloads:
            if self._post_signup_json(url, payload):
                return True
        return self._post_signup_form(
            url,
            {
                "email": email,
                "password": password,
                "password_confirmation": password,
                "username": email,
            },
        )

    @staticmethod
    def _post_signup_json(url: str, payload: dict[str, Any]) -> bool:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status in {200, 201, 204}
        except urllib.error.HTTPError as exc:
            return exc.code in {200, 201, 204}
        except Exception:
            return False

    @staticmethod
    def _post_signup_form(url: str, fields: dict[str, str]) -> bool:
        encoded = urllib.parse.urlencode(fields).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status in {200, 201, 204}
        except urllib.error.HTTPError as exc:
            return exc.code in {200, 201, 204, 302, 303}
        except Exception:
            return False

    def _try_db_user_insert(
        self,
        connection_string: str,
        db_type: str,
        root: Path,
        email: str,
        password: str,
    ) -> bool:
        """Insert a user row directly when no signup endpoint worked."""
        table = self._discover_user_table(root, connection_string, db_type)
        if table is None:
            return False
        columns = self._list_table_columns(connection_string, db_type, table)
        if not columns:
            return False

        email_column = next(
            (name for name in _EMAIL_COLUMN_CANDIDATES if name in columns),
            None,
        )
        password_column = next(
            (name for name in _PASSWORD_COLUMN_CANDIDATES if name in columns),
            None,
        )
        if email_column is None:
            return False

        values: dict[str, Any] = {email_column: email}
        if password_column is not None:
            values[password_column] = self._hash_password_for_insert(password)

        insert_columns = [name for name in values if name in columns]
        if email_column not in insert_columns:
            return False

        normalized = db_type.strip().lower()
        if normalized == "sqlite":
            placeholders = ", ".join("?" for _ in insert_columns)
        else:
            placeholders = ", ".join("%s" for _ in insert_columns)
        column_sql = ", ".join(insert_columns)
        sql = f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})"
        params = tuple(values[name] for name in insert_columns)
        try:
            self._execute_db_write(connection_string, db_type, sql, params)
            return True
        except EnvironmentSetupError as exc:
            self._logger.warning("Direct DB user insert failed: %s", exc)
            return False

    def _discover_user_table(
        self,
        root: Path,
        connection_string: str,
        db_type: str,
    ) -> str | None:
        existing = {
            name.lower()
            for name in self._list_existing_tables(connection_string, db_type)
        }
        for candidate in _USER_TABLE_CANDIDATES:
            if candidate in existing:
                return candidate
        for path in self._iter_source_files(root):
            if path.suffix.lower() != ".py":
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            match = re.search(
                r"""__tablename__\s*=\s*['"]([a-zA-Z0-9_]+)['"]""",
                text,
            )
            if match:
                table = match.group(1).lower()
                if table in existing:
                    return table
        return None

    def _list_existing_tables(
        self,
        connection_string: str,
        db_type: str,
    ) -> list[str]:
        normalized = db_type.strip().lower()
        if normalized == "sqlite" and connection_string.startswith("sqlite:///"):
            db_path = connection_string.removeprefix("sqlite:///")
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            return [str(row[0]) for row in rows]
        if normalized in {"postgres", "postgresql"} and connection_string.startswith(
            "postgresql://"
        ):
            ensure_db_driver("postgres", self._logger)
            import psycopg2

            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT tablename FROM pg_catalog.pg_tables "
                        "WHERE schemaname NOT IN ('pg_catalog', 'information_schema')"
                    )
                    rows = cursor.fetchall()
            return [str(row[0]) for row in rows]
        if normalized == "mysql" and connection_string.startswith("mysql://"):
            ensure_db_driver("mysql", self._logger)
            import mysql.connector

            with mysql.connector.connect(
                **self._parse_mysql_connection(connection_string)
            ) as conn:
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                rows = cursor.fetchall()
            return [str(row[0]) for row in rows]
        return []

    def _list_table_columns(
        self,
        connection_string: str,
        db_type: str,
        table: str,
    ) -> set[str]:
        normalized = db_type.strip().lower()
        if normalized == "sqlite" and connection_string.startswith("sqlite:///"):
            db_path = connection_string.removeprefix("sqlite:///")
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return {str(row[1]).lower() for row in rows}
        if normalized in {"postgres", "postgresql"} and connection_string.startswith(
            "postgresql://"
        ):
            ensure_db_driver("postgres", self._logger)
            import psycopg2

            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = %s",
                        (table,),
                    )
                    rows = cursor.fetchall()
            return {str(row[0]).lower() for row in rows}
        if normalized == "mysql" and connection_string.startswith("mysql://"):
            ensure_db_driver("mysql", self._logger)
            import mysql.connector

            with mysql.connector.connect(
                **self._parse_mysql_connection(connection_string)
            ) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                rows = cursor.fetchall()
            return {str(row[0]).lower() for row in rows}
        return set()

    @staticmethod
    def _parse_mysql_connection(connection_string: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(connection_string)
        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": parsed.username or "agentic_test",
            "password": parsed.password or "agentic_test",
            "database": (parsed.path or "/agentic_test").lstrip("/") or "agentic_test",
        }

    @staticmethod
    def _hash_password_for_insert(password: str) -> str:
        try:
            import bcrypt

            return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        except ImportError:
            return password

    def _execute_db_write(
        self,
        connection_string: str,
        db_type: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> None:
        normalized = db_type.strip().lower()
        try:
            if normalized == "sqlite" and connection_string.startswith("sqlite:///"):
                db_path = connection_string.removeprefix("sqlite:///")
                import sqlite3

                with sqlite3.connect(db_path) as conn:
                    conn.execute(sql, params)
                    conn.commit()
                return
            if normalized in {"postgres", "postgresql"} and connection_string.startswith(
                "postgresql://"
            ):
                ensure_db_driver("postgres", self._logger)
                import psycopg2

                with psycopg2.connect(connection_string) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(sql, params)
                    conn.commit()
                return
            if normalized == "mysql" and connection_string.startswith("mysql://"):
                ensure_db_driver("mysql", self._logger)
                import mysql.connector

                with mysql.connector.connect(
                    **self._parse_mysql_connection(connection_string)
                ) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql, params)
                    conn.commit()
                return
            raise EnvironmentSetupError(
                f"Unsupported db_type for user insert: {db_type!r}"
            )
        except EnvironmentSetupError:
            raise
        except Exception as exc:
            raise EnvironmentSetupError(f"DB write failed: {exc}") from exc

    def _detect_db_type_with_confidence(self, root: Path) -> tuple[str | None, str]:
        scores: dict[str, int] = {key: 0 for key in _DB_IMPORT_PATTERNS}
        for path in self._iter_source_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for db_name, markers in _DB_IMPORT_PATTERNS.items():
                if any(marker in text for marker in markers):
                    scores[db_name] += 1

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = ranked[0]
        if best_score == 0:
            return None, "none"

        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
        if runner_up_score > 0 and best_score < runner_up_score * _MIN_CONFIDENCE_RATIO:
            return None, "low"
        if best_score < _MIN_DB_SCORE:
            return best_name, "low"
        return best_name, "high"

    def _detect_backend_command_with_confidence(
        self,
        root: Path,
    ) -> tuple[str | None, str]:
        dockerfile = root / "Dockerfile"
        if dockerfile.exists():
            try:
                for line in dockerfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip().upper().startswith("CMD"):
                        cmd = line.split("CMD", 1)[1].strip().strip("[]").strip('"').strip("'")
                        if cmd:
                            return cmd, "high"
            except OSError:
                pass

        procfile = root / "Procfile"
        if procfile.exists():
            try:
                for line in procfile.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("web:"):
                        command = line.split(":", 1)[1].strip()
                        if command:
                            return command, "high"
            except OSError:
                pass

        package_json_paths = list(root.glob("package.json")) + list(root.glob("*/package.json"))
        for package_json in package_json_paths:
            try:
                payload = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scripts = payload.get("scripts", {})
            for key in ("dev:server", "start:server", "start", "dev"):
                command = scripts.get(key)
                if isinstance(command, str) and command.strip():
                    return command.strip(), "medium"

        for path in self._iter_source_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern, _label in _BACKEND_COMMAND_PATTERNS:
                match = pattern.search(text)
                if match:
                    return match.group(0).strip(), "medium"

        if (root / "manage.py").exists():
            return "python manage.py runserver 0.0.0.0:8000", "medium"

        if any((root / name).exists() for name in ("app.py", "main.py", "server.py")):
            return None, "low"

        return None, "none"

    def _detect_frontend_command(self, root: Path) -> tuple[str | None, int | None]:
        package_json_paths = list(root.glob("package.json")) + list(
            root.glob("*/package.json")
        )
        for package_json in package_json_paths:
            try:
                payload = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scripts = payload.get("scripts", {})
            for key in ("dev", "start", "serve"):
                command = scripts.get(key)
                if isinstance(command, str) and command.strip():
                    port = self._extract_port(command) or self._default_frontend_port(
                        package_json.parent
                    )
                    return f"npm run {key}", port
        return None, None

    def _detect_env_vars(self, root: Path) -> set[str]:
        found: set[str] = set()
        for path in self._iter_source_files(root):
            if path.suffix.lower() not in {".py", ".js", ".ts", ".tsx", ".jsx", ".env.example"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in _ENV_VAR_PATTERNS:
                found.update(pattern.findall(text))
        return found

    def _start_sqlite_database(self) -> str:
        handle = tempfile.NamedTemporaryFile(prefix="agentic-flow-sqlite-", suffix=".db", delete=False)
        handle.close()
        self._sqlite_path = Path(handle.name)
        return f"sqlite:///{self._sqlite_path.as_posix()}"

    def _start_docker_database(
        self,
        *,
        image: str,
        internal_port: int,
        env: dict[str, str],
        connection_builder: Callable[[int], str],
        readiness_check: Callable[[str, int], None],
    ) -> str:
        if shutil.which("docker") is None:
            raise EnvironmentSetupError(
                "start_test_database failed: docker executable not found on PATH"
            )

        container_name = f"agentic-flow-db-{uuid.uuid4().hex[:10]}"
        command = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"0:{internal_port}",
        ]
        for key, value in env.items():
            command.extend(["-e", f"{key}={value}"])
        command.append(image)

        container_id: str | None = None
        try:
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.CalledProcessError as exc:
                raise EnvironmentSetupError(
                    "start_test_database failed while running docker "
                    f"(image={image!r}): {exc.stderr or exc.stdout or exc}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise EnvironmentSetupError(
                    f"start_test_database timed out pulling/starting docker image={image!r}"
                ) from exc

            container_id = completed.stdout.strip()
            self._docker_container_id = container_id
            host_port = self._docker_published_port(container_id, internal_port)
            connection_string = connection_builder(host_port)
            readiness_check("127.0.0.1", host_port)
            return connection_string
        except EnvironmentSetupError:
            if container_id:
                self._force_remove_container(container_id)
            raise
        except Exception as exc:
            if container_id:
                self._force_remove_container(container_id)
            raise EnvironmentSetupError(
                f"start_test_database failed for image={image}: {exc}"
            ) from exc

    def _force_remove_container(self, container_id: str) -> None:
        try:
            subprocess.run(
                ["docker", "stop", container_id],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            self._logger.warning("Could not stop failed container %s: %s", container_id, exc)
        finally:
            if self._docker_container_id == container_id:
                self._docker_container_id = None

    def _docker_published_port(self, container_id: str, internal_port: int) -> int:
        try:
            completed = subprocess.run(
                ["docker", "port", container_id, str(internal_port)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            raise EnvironmentSetupError(
                f"Could not read published port for container {container_id}: "
                f"{exc.stderr or exc.stdout or exc}"
            ) from exc
        line = completed.stdout.strip().splitlines()[0]
        host_port = int(line.rsplit(":", 1)[-1])
        return host_port

    @staticmethod
    def _postgres_connection_string(host_port: int) -> str:
        return (
            "postgresql://agentic_test:agentic_test@127.0.0.1:"
            f"{host_port}/agentic_test"
        )

    @staticmethod
    def _mysql_connection_string(host_port: int) -> str:
        return (
            "mysql://agentic_test:agentic_test@127.0.0.1:"
            f"{host_port}/agentic_test"
        )

    @staticmethod
    def _mongo_connection_string(host_port: int) -> str:
        return f"mongodb://127.0.0.1:{host_port}/agentic_test"

    @staticmethod
    def _wait_for_tcp(host: str, port: int, timeout: int = _DOCKER_WAIT_SECONDS) -> None:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(1)
        raise EnvironmentSetupError(
            f"Database did not accept TCP connections on {host}:{port} within {timeout}s: {last_error}"
        )

    def _wait_for_http(
        self,
        ready_url: str,
        *,
        timeout: int,
        process: subprocess.Popen[str],
    ) -> None:
        deadline = time.time() + timeout
        last_error: str | None = None
        while time.time() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                stdout = process.stdout.read() if process.stdout else ""
                raise EnvironmentSetupError(
                    "start_process failed: process exited before becoming ready "
                    f"(code={process.returncode}, stdout={stdout!r}, stderr={stderr!r})"
                )
            try:
                with urllib.request.urlopen(ready_url, timeout=3) as response:
                    if 200 <= response.status < 500:
                        return
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1)
        raise EnvironmentSetupError(
            f"start_process failed: {ready_url} did not become ready within {timeout}s ({last_error})"
        )

    def _find_schema_sql(self, root: Path) -> Path | None:
        candidates = [
            root / "schema.sql",
            root / "db" / "schema.sql",
            root / "database" / "schema.sql",
            root / "migrations" / "schema.sql",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _apply_schema_sql(self, schema_path: Path, connection_string: str) -> None:
        sql = schema_path.read_text(encoding="utf-8")
        if connection_string.startswith("sqlite:///"):
            db_path = connection_string.removeprefix("sqlite:///")
            import sqlite3

            with sqlite3.connect(db_path) as conn:
                conn.executescript(sql)
            return
        if connection_string.startswith("postgresql://"):
            ensure_db_driver("postgres", self._logger)
            try:
                import psycopg2
            except ImportError as exc:
                raise EnvironmentSetupError(
                    "run_migrations failed: psycopg2 is required to apply schema.sql to postgres"
                ) from exc
            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                conn.commit()
            return
        raise EnvironmentSetupError(
            f"run_migrations failed: unsupported connection string for schema.sql: {connection_string}"
        )

    def _run_checked(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        step: str,
    ) -> None:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                self._logger.debug("%s stdout: %s", step, completed.stdout.strip())
        except subprocess.CalledProcessError as exc:
            raise EnvironmentSetupError(
                f"{step} failed (exit={exc.returncode}): {exc.stderr or exc.stdout or exc}"
            ) from exc

    @staticmethod
    def _infer_db_type_from_connection(connection_string: str) -> str | None:
        if connection_string.startswith("sqlite:///"):
            return "sqlite"
        if connection_string.startswith("postgresql://"):
            return "postgres"
        if connection_string.startswith("mysql://"):
            return "mysql"
        if connection_string.startswith("mongodb://"):
            return "mongo"
        return None

    @staticmethod
    def _iter_source_files(root: Path) -> list[Path]:
        ignored = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__"}
        files: list[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and not any(part in ignored for part in path.parts):
                files.append(path)
        return files

    @staticmethod
    def _extract_port(command: str) -> int | None:
        match = re.search(r"(?:--port|-p)\s+(\d+)", command)
        if match:
            return int(match.group(1))
        match = re.search(r":(\d{2,5})\b", command)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _default_frontend_port(package_dir: Path) -> int:
        if "vite" in package_dir.name or (package_dir / "vite.config.ts").exists():
            return 5173
        return 3000

    @staticmethod
    def _terminate_process(handle: subprocess.Popen[Any] | None) -> None:
        if handle is None or handle.poll() is not None:
            return
        handle.terminate()
        try:
            handle.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.kill()
            handle.wait(timeout=5)


def authenticated_git_remote_url(repo_full_name: str, github_token: str) -> str:
    """Build an HTTPS GitHub remote URL that includes token authentication."""
    owner, _, name = repo_full_name.strip().partition("/")
    if not owner or not name:
        raise EnvironmentSetupError(
            f"Invalid repository slug {repo_full_name!r}; expected owner/name."
        )
    token = github_token.strip()
    if not token:
        raise EnvironmentSetupError("github_token must not be empty for git operations.")
    return f"https://x-access-token:{token}@github.com/{owner}/{name}.git"


def mask_github_token(text: str, github_token: str) -> str:
    """Redact token substrings before logging git command or error output."""
    masked = text.replace(github_token, "***") if github_token else text
    return re.sub(
        r"https://x-access-token:[^@\s]+@",
        "https://x-access-token:***@",
        masked,
    )


def checkout_git_branch(
    local_repo_path: str,
    branch_name: str,
    repo_full_name: str,
    github_token: str,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Point origin at an authenticated remote, fetch, and check out a branch."""
    log = logger or get_logger(__name__)
    token = github_token.strip()
    remote_url = authenticated_git_remote_url(repo_full_name, token)
    repo_path = Path(local_repo_path)
    if not repo_path.is_dir():
        raise EnvironmentSetupError(
            f"Local repository path does not exist: {local_repo_path!r}"
        )

    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    def run_git(args: list[str], action: str) -> None:
        log.debug("%s", mask_github_token(action, token))
        try:
            completed = subprocess.run(
                args,
                cwd=str(repo_path),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise EnvironmentSetupError(
                mask_github_token(f"{action} timed out after 120s", token)
            ) from exc
        except OSError as exc:
            raise EnvironmentSetupError(
                mask_github_token(f"{action} failed to start: {exc}", token)
            ) from exc

        if completed.returncode != 0:
            detail = mask_github_token(
                (completed.stderr or completed.stdout or "").strip(),
                token,
            )
            if detail:
                message = f"{action} failed (exit {completed.returncode}): {detail}"
            else:
                message = f"{action} failed (exit {completed.returncode})"
            raise EnvironmentSetupError(mask_github_token(message, token))

    run_git(
        ["git", "remote", "set-url", "origin", remote_url],
        f"git remote set-url origin for {repo_full_name}",
    )

    ref_check = subprocess.run(
        ["git", "rev-parse", "--verify", branch_name],
        cwd=str(repo_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if ref_check.returncode != 0:
        try:
            run_git(
                [
                    "git",
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    f"{branch_name}:{branch_name}",
                ],
                f"git fetch origin {branch_name}:{branch_name}",
            )
        except EnvironmentSetupError as exc:
            raise EnvironmentSetupError(
                f"Could not fetch branch {branch_name!r} for local clone "
                f"{local_repo_path!r}: {exc}"
            ) from exc

    try:
        run_git(
            ["git", "checkout", branch_name],
            f"git checkout {branch_name}",
        )
    except EnvironmentSetupError as exc:
        raise EnvironmentSetupError(
            f"Could not check out branch {branch_name!r} in local clone "
            f"{local_repo_path!r}: {exc}"
        ) from exc

    log.info("Checked out branch %r in %s", branch_name, local_repo_path)
