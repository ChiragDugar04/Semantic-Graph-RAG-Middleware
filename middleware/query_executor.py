from __future__ import annotations

import logging
import re
import time
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool

from middleware.models import DBResult, ExtractedParameters, QueryTemplate

logger = logging.getLogger(__name__)

_pool: Optional[MySQLConnectionPool] = None


def _load_db_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "db_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _get_pool() -> MySQLConnectionPool:
    global _pool
    if _pool is None:
        cfg = _load_db_config()
        db = cfg["database"]
        pool_cfg = cfg.get("connection_pool", {})

        _pool = MySQLConnectionPool(
            pool_name="rag_pool",
            pool_size=pool_cfg.get("pool_size", 5),
            host=db["host"],
            port=db["port"],
            user=db["user"],
            password=db["password"],
            database=db["database"],
            connection_timeout=pool_cfg.get("connection_timeout", 30),
            autocommit=True,
        )
        logger.info(
            "MySQL connection pool created: size=%d host=%s db=%s",
            pool_cfg.get("pool_size", 5), db["host"], db["database"]
        )
    return _pool


def _get_connection() -> mysql.connector.MySQLConnection:
    return _get_pool().get_connection()


def _load_schema() -> dict:
    schema_path = Path(__file__).parent.parent / "config" / "graph_schema.yaml"
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)


def _build_empty_result_message(question: str, description: str) -> str:
    """
    Return an entity-specific empty-result message by matching the question
    against each node's empty_result_keywords list in graph_schema.yaml.

    Schema-driven: adding a new entity with empty_result_hint and
    empty_result_keywords requires only a YAML edit — zero Python changes.
    """
    q_lower = question.lower()
    schema = _load_schema()

    for node_data in schema.get("nodes", {}).values():
        hint = node_data.get("empty_result_hint", "")
        keywords = node_data.get("empty_result_keywords", [])
        if hint and any(kw in q_lower for kw in keywords):
            return hint

    return (
        "No records were found for your query. "
        "Please check your search terms and try again."
    )


def _serialize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import decimal
    import datetime

    clean = []
    for row in rows:
        clean_row: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, decimal.Decimal):
                clean_row[key] = float(value)
            elif isinstance(value, (datetime.date, datetime.datetime)):
                clean_row[key] = str(value)
            else:
                clean_row[key] = value
        clean.append(clean_row)
    return clean


def _build_display_sql(sql: str, params: Dict[str, Any]) -> str:
    display = sql
    for key, value in params.items():
        display = display.replace(f"%({key})s", f"'{value}'")
    return display


def execute_query(
    template: QueryTemplate,
    parameters: ExtractedParameters,
    user_question: str,
) -> DBResult:
    if not template.sql_template.strip():
        logger.info("No SQL template provided — returning empty result")
        return DBResult(
            rows=[],
            row_count=0,
            error="no_matching_intent",
            self_healing_triggered=True,
            healing_reason="Question did not match any known database query pattern",
            query_executed="-- No matching query built",
            params_used={},
            execution_time_ms=0.0,
        )

    if parameters.missing_required:
        logger.info("Missing required params: %s", parameters.missing_required)
        missing_str = ", ".join(parameters.missing_required)
        return DBResult(
            rows=[],
            row_count=0,
            error="missing_parameters",
            self_healing_triggered=True,
            healing_reason=f"Missing required parameters: {missing_str}",
            query_executed=f"-- Skipped: missing {missing_str}",
            params_used=parameters.params,
            execution_time_ms=0.0,
        )

    raw_params = parameters.params
    sql = template.sql_template.strip()

    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        start_time = time.perf_counter()
        cursor.execute(sql, raw_params)
        rows: List[Dict[str, Any]] = cursor.fetchall()
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        cursor.close()

        display_sql = _build_display_sql(sql, raw_params)

        if not rows:
            empty_message = _build_empty_result_message(user_question, template.description)
            logger.info(
                "Query returned 0 rows for: '%s' (%.3fms)",
                user_question, execution_time_ms
            )
            return DBResult(
                rows=[{"healing_message": empty_message}],
                row_count=0,
                execution_time_ms=round(execution_time_ms, 3),
                query_executed=display_sql,
                params_used=raw_params,
                error="empty_result",
                self_healing_triggered=True,
                healing_reason="Query returned 0 rows",
            )

        serialized = _serialize_rows(rows)

        logger.info(
            "Query returned %d rows in %.3fms for: '%s'",
            len(serialized), execution_time_ms, user_question
        )

        return DBResult(
            rows=serialized,
            row_count=len(serialized),
            execution_time_ms=round(execution_time_ms, 3),
            query_executed=display_sql,
            params_used=raw_params,
            error=None,
            self_healing_triggered=False,
        )

    except mysql.connector.Error as db_err:
        logger.error("MySQL error for query '%s': %s", user_question, db_err)
        return DBResult(
            rows=[],
            row_count=0,
            execution_time_ms=0.0,
            query_executed=sql,
            params_used=raw_params,
            error=f"Database error: {str(db_err)}",
            self_healing_triggered=False,
        )

    except Exception as exc:
        logger.error("Unexpected error in execute_query: %s", exc, exc_info=True)
        return DBResult(
            rows=[],
            row_count=0,
            execution_time_ms=0.0,
            query_executed=sql,
            params_used=raw_params,
            error=f"Unexpected error: {str(exc)}",
            self_healing_triggered=False,
        )

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
