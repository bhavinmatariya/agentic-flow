"""Read-only and scoped cleanup queries against disposable test databases."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any
from urllib.parse import urlparse

from core.exceptions import EnvironmentSetupError, ToolError
from tools.environment_manager import ensure_db_driver
from utils.logger import get_logger

_MAX_DELETE_ROWS: int = 5


class DBVerifierTool:
    """Verify and clean up rows written during live full-stack tests."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Create the tool with optional logging."""
        self._logger = logger or get_logger(__name__)

    def query(
        self,
        connection_string: str,
        db_type: str,
        table: str,
        marker_column: str,
        marker_value: str,
    ) -> list[dict[str, Any]]:
        """Return rows whose ``marker_column`` exactly equals ``marker_value``.

        An empty list means no matching row was found — that is a legitimate
        verification outcome, not an infrastructure failure.

        Raises:
            EnvironmentSetupError: If the database cannot be reached or queried.
            ToolError: If identifiers are invalid.
        """
        normalized_table = self._validate_identifier(table, "table")
        normalized_column = self._validate_identifier(marker_column, "marker_column")
        sql = f'SELECT * FROM {normalized_table} WHERE {normalized_column} = ?'
        rows = self._execute_query(
            connection_string=connection_string,
            db_type=db_type,
            sql=sql,
            params=(marker_value,),
            marker_column=normalized_column,
            marker_value=marker_value,
        )
        self._logger.info(
            "DB query on %s.%s found %d row(s) for marker=%r",
            normalized_table,
            normalized_column,
            len(rows),
            marker_value,
        )
        return rows

    def delete_by_marker(
        self,
        connection_string: str,
        db_type: str,
        table: str,
        marker_column: str,
        marker_value: str,
    ) -> int:
        """Delete rows matching ``marker_value`` when the match count is small."""
        normalized_table = self._validate_identifier(table, "table")
        normalized_column = self._validate_identifier(marker_column, "marker_column")
        count_sql = (
            f'SELECT COUNT(*) AS row_count FROM {normalized_table} '
            f'WHERE {normalized_column} = ?'
        )
        count_rows = self._execute_query(
            connection_string=connection_string,
            db_type=db_type,
            sql=count_sql,
            params=(marker_value,),
        )
        match_count = int(count_rows[0].get("row_count", 0)) if count_rows else 0
        if match_count > _MAX_DELETE_ROWS:
            raise ToolError(
                f"Refusing to delete {match_count} rows from {normalized_table!r}; "
                f"expected at most {_MAX_DELETE_ROWS} test rows for marker={marker_value!r}"
            )

        delete_sql = (
            f'DELETE FROM {normalized_table} WHERE {normalized_column} = ?'
        )
        deleted = self._execute_write(
            connection_string=connection_string,
            db_type=db_type,
            sql=delete_sql,
            params=(marker_value,),
            marker_column=normalized_column,
            marker_value=marker_value,
        )
        self._logger.info(
            "Deleted %d test row(s) from %s.%s for marker=%r",
            deleted,
            normalized_table,
            normalized_column,
            marker_value,
        )
        return deleted

    def _execute_query(
        self,
        *,
        connection_string: str,
        db_type: str,
        sql: str,
        params: tuple[Any, ...],
        marker_column: str = "",
        marker_value: str = "",
    ) -> list[dict[str, Any]]:
        normalized = db_type.strip().lower()
        try:
            ensure_db_driver(normalized, self._logger)
            if normalized == "sqlite" or connection_string.startswith("sqlite:///"):
                return self._sqlite_query(connection_string, sql, params)
            if normalized in {"postgres", "postgresql"} or connection_string.startswith(
                "postgresql://"
            ):
                return self._postgres_query(connection_string, sql, params)
            if normalized == "mysql" or connection_string.startswith("mysql://"):
                return self._mysql_query(connection_string, sql, params)
            if normalized in {"mongo", "mongodb"} or connection_string.startswith(
                "mongodb://"
            ):
                return self._mongo_query(
                    connection_string,
                    table_from_sql(sql),
                    marker_column,
                    marker_value,
                )
            raise ToolError(f"Unsupported db_type for query: {db_type!r}")
        except ToolError:
            raise
        except EnvironmentSetupError:
            raise
        except Exception as exc:
            raise EnvironmentSetupError(
                f"Database connection/query failed (infrastructure): {exc}"
            ) from exc

    def _execute_write(
        self,
        *,
        connection_string: str,
        db_type: str,
        sql: str,
        params: tuple[Any, ...],
        marker_column: str = "",
        marker_value: str = "",
    ) -> int:
        normalized = db_type.strip().lower()
        try:
            ensure_db_driver(normalized, self._logger)
            if normalized == "sqlite" or connection_string.startswith("sqlite:///"):
                return self._sqlite_write(connection_string, sql, params)
            if normalized in {"postgres", "postgresql"} or connection_string.startswith(
                "postgresql://"
            ):
                return self._postgres_write(connection_string, sql, params)
            if normalized == "mysql" or connection_string.startswith("mysql://"):
                return self._mysql_write(connection_string, sql, params)
            if normalized in {"mongo", "mongodb"} or connection_string.startswith(
                "mongodb://"
            ):
                return self._mongo_delete(
                    connection_string,
                    table_from_sql(sql),
                    marker_column,
                    marker_value,
                )
            raise ToolError(f"Unsupported db_type for delete: {db_type!r}")
        except ToolError:
            raise
        except EnvironmentSetupError:
            raise
        except Exception as exc:
            raise EnvironmentSetupError(
                f"Database connection/delete failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _validate_identifier(value: str, label: str) -> str:
        cleaned = value.strip()
        if not cleaned.replace("_", "").isalnum():
            raise ToolError(f"Invalid SQL identifier for {label}: {value!r}")
        return cleaned

    @staticmethod
    def _sqlite_query(
        connection_string: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        db_path = connection_string.removeprefix("sqlite:///")
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise EnvironmentSetupError(
                f"SQLite connection/query failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _sqlite_write(
        connection_string: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        db_path = connection_string.removeprefix("sqlite:///")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(sql, params)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise EnvironmentSetupError(
                f"SQLite connection/delete failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _postgres_query(
        connection_string: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as exc:
            raise EnvironmentSetupError(
                "Postgres driver unavailable (infrastructure): psycopg2 is required"
            ) from exc

        pg_sql = sql.replace("?", "%s")
        try:
            with psycopg2.connect(connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(pg_sql, params)
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            raise EnvironmentSetupError(
                f"Postgres connection/query failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _postgres_write(
        connection_string: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        try:
            import psycopg2
        except ImportError as exc:
            raise EnvironmentSetupError(
                "Postgres driver unavailable (infrastructure): psycopg2 is required"
            ) from exc

        pg_sql = sql.replace("?", "%s")
        try:
            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(pg_sql, params)
                    conn.commit()
                    return cursor.rowcount
        except Exception as exc:
            raise EnvironmentSetupError(
                f"Postgres connection/delete failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _mysql_query(
        connection_string: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> list[dict[str, Any]]:
        try:
            import mysql.connector
        except ImportError as exc:
            raise EnvironmentSetupError(
                "MySQL driver unavailable (infrastructure): mysql-connector-python is required"
            ) from exc

        parsed = urlparse(connection_string)
        mysql_sql = sql.replace("?", "%s")
        try:
            conn = mysql.connector.connect(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 3306,
                user=parsed.username or "root",
                password=parsed.password or "",
                database=parsed.path.lstrip("/") or None,
            )
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(mysql_sql, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()
        except Exception as exc:
            raise EnvironmentSetupError(
                f"MySQL connection/query failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _mysql_write(
        connection_string: str,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        try:
            import mysql.connector
        except ImportError as exc:
            raise EnvironmentSetupError(
                "MySQL driver unavailable (infrastructure): mysql-connector-python is required"
            ) from exc

        parsed = urlparse(connection_string)
        mysql_sql = sql.replace("?", "%s")
        try:
            conn = mysql.connector.connect(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 3306,
                user=parsed.username or "root",
                password=parsed.password or "",
                database=parsed.path.lstrip("/") or None,
            )
            try:
                cursor = conn.cursor()
                cursor.execute(mysql_sql, params)
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
        except Exception as exc:
            raise EnvironmentSetupError(
                f"MySQL connection/delete failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _mongo_query(
        connection_string: str,
        table: str,
        marker_column: str,
        marker_value: str,
    ) -> list[dict[str, Any]]:
        try:
            from pymongo import MongoClient
            from pymongo.errors import PyMongoError
        except ImportError as exc:
            raise EnvironmentSetupError(
                "Mongo driver unavailable (infrastructure): pymongo is required"
            ) from exc

        try:
            client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            db_name = urlparse(connection_string).path.lstrip("/") or "agentic_test"
            docs = list(client[db_name][table].find({marker_column: marker_value}))
            client.close()
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            return docs
        except PyMongoError as exc:
            raise EnvironmentSetupError(
                f"Mongo connection/query failed (infrastructure): {exc}"
            ) from exc

    @staticmethod
    def _mongo_delete(
        connection_string: str,
        table: str,
        marker_column: str,
        marker_value: str,
    ) -> int:
        try:
            from pymongo import MongoClient
            from pymongo.errors import PyMongoError
        except ImportError as exc:
            raise EnvironmentSetupError(
                "Mongo driver unavailable (infrastructure): pymongo is required"
            ) from exc

        try:
            client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            db_name = urlparse(connection_string).path.lstrip("/") or "agentic_test"
            result = client[db_name][table].delete_many({marker_column: marker_value})
            client.close()
            return int(result.deleted_count)
        except PyMongoError as exc:
            raise EnvironmentSetupError(
                f"Mongo connection/delete failed (infrastructure): {exc}"
            ) from exc


def table_from_sql(sql: str) -> str:
    """Extract a table name from a simple SQL statement."""
    tokens = sql.replace(",", " ").split()
    for keyword in ("FROM", "INTO", "UPDATE"):
        if keyword in tokens:
            return tokens[tokens.index(keyword) + 1]
    raise ToolError(f"Could not parse table name from SQL: {sql!r}")
