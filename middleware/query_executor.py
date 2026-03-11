"""
middleware/query_executor.py

Executes parameterized SQL queries against MySQL and returns DBResult.

This module is responsible for:
  1. Safe parameterized query execution (no SQL injection possible)
  2. Precise execution time measurement
  3. The Self-Healing loop — intercepts missing params and empty
     results before they become user-facing errors

The LLM never writes SQL. Every query that runs was written by a
human developer and lives in intents.yaml.
"""

from __future__ import annotations

import time
import yaml
import mysql.connector

from pathlib import Path
from typing import Any, Dict, List, Optional

from middleware.models import DBResult, ExtractedParameters, QueryTemplate


# ============================================================
# EXACT-MATCH PARAMS
# These params must NOT get % wildcard wrapping.
# They match against ENUM or fixed-value columns in MySQL.
# ============================================================

_EXACT_MATCH_PARAMS = {
    "status",       # orders.status ENUM: pending/processing/shipped/delivered/cancelled
    "category",     # products.category: exact string match needed
    "manager_name", # pre-wrapped by graph_query_builder (subquery uses LIKE already)
}


# ============================================================
# CONFIG + CONNECTION
# ============================================================

def _load_db_config() -> dict:
    """Load database connection settings from db_config.yaml.

    Returns:
        dict: The 'database' section of the config.
    """
    config_path = Path(__file__).parent.parent / "config" / "db_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["database"]


def _get_connection() -> mysql.connector.MySQLConnection:
    """Create and return a fresh MySQL connection.

    Returns:
        MySQLConnection: An open, ready-to-use connection.

    Raises:
        mysql.connector.Error: If connection fails.
    """
    cfg = _load_db_config()
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        connection_timeout=cfg.get("connection_timeout", 30),
    )


# ============================================================
# SELF-HEALING HELPER
# ============================================================

def _generate_healing_message(
    user_question: str,
    intent_description: str,
    missing_params: List[str],
    healing_reason: str,
    base_url: str,
    synthesis_model: str,
) -> str:
    """Ask the LLM to generate a polite follow-up question for the user.

    Called when: (a) required parameters are missing, or
                 (b) the query returned 0 rows.

    Args:
        user_question: The original user question.
        intent_description: Human-readable description of what was attempted.
        missing_params: Parameter names that were not found.
        healing_reason: Why healing was triggered.
        base_url: Ollama API base URL.
        synthesis_model: Model to use for generating the message.

    Returns:
        str: A natural language follow-up question for the user.
    """
    if missing_params:
        missing_str = ", ".join(f'"{p}"' for p in missing_params)
        prompt = f"""A user asked: "{user_question}"

We understood they want: {intent_description}
But we could not find the required information: {missing_str}

Write a single, polite, conversational follow-up question asking the user
to provide the missing information. Be specific about what is needed.
Keep it to 1-2 sentences maximum. Do not explain technical details."""
    else:
        prompt = f"""A user asked: "{user_question}"

We searched the database for: {intent_description}
The database returned no matching records.

Write a single, polite response telling the user no results were found,
and ask them to verify the details or try a different search.
Keep it to 1-2 sentences maximum."""

    try:
        import requests
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": synthesis_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 100},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception:
        if missing_params:
            return (
                f"I need a bit more information to answer that. "
                f"Could you provide the {missing_params[0]}?"
            )
        return "I couldn't find any matching records. Could you check the details and try again?"


# ============================================================
# MAIN EXECUTOR
# ============================================================

def execute_query(
    template: QueryTemplate,
    parameters: ExtractedParameters,
    user_question: str,
) -> DBResult:
    """Execute a parameterized SQL query and return a DBResult.

    Flow:
      1. Check for missing required parameters → trigger healing if needed
      2. Check for 'other' intent (no SQL) → trigger healing
      3. Build parameterized query safely
      4. Execute against MySQL with precise timing
      5. Check for empty results → trigger healing if needed
      6. Return populated DBResult

    Args:
        template: The QueryTemplate with SQL and param definitions.
        parameters: Extracted parameters from param_extractor.
        user_question: Original user question (needed for healing prompt).

    Returns:
        DBResult: Query results or a healing-triggered result.
    """
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        model_cfg = yaml.safe_load(f)["models"]

    base_url: str        = model_cfg["ollama_base_url"]
    synthesis_model: str = model_cfg["synthesis_model"]

    # ── Guard 1: No SQL template (intent = 'other') ──────────
    if not template.sql_template.strip():
        _generate_healing_message(
            user_question=user_question,
            intent_description="answer your question from the database",
            missing_params=[],
            healing_reason="no_matching_intent",
            base_url=base_url,
            synthesis_model=synthesis_model,
        )
        return DBResult(
            rows=[],
            row_count=0,
            error="no_matching_intent",
            self_healing_triggered=True,
            healing_reason="Question did not match any known database query",
            query_executed="-- No matching intent found",
            params_used={},
            execution_time_ms=0.0,
        )

    # ── Guard 2: Missing required parameters ─────────────────
    if parameters.missing_required:
        healing_msg = _generate_healing_message(
            user_question=user_question,
            intent_description=template.description,
            missing_params=parameters.missing_required,
            healing_reason="missing_parameters",
            base_url=base_url,
            synthesis_model=synthesis_model,
        )
        return DBResult(
            rows=[{"healing_message": healing_msg}],
            row_count=0,
            error="missing_parameters",
            self_healing_triggered=True,
            healing_reason=f"Missing required params: {parameters.missing_required}",
            query_executed=f"-- Skipped: missing {parameters.missing_required}",
            params_used=parameters.params,
            execution_time_ms=0.0,
        )

    # ── Build parameterized query ─────────────────────────────
    # IMPORTANT: Only wrap LIKE-search fields with % wildcards.
    # Exact-match fields (status, category) must NOT get wildcards
    # because they match against ENUM or fixed-value columns.
    query_params: Dict[str, Any] = {}
    for key, value in parameters.params.items():
        if isinstance(value, str) and key not in _EXACT_MATCH_PARAMS:
            # LIKE search — wrap with wildcards so "Sarah" matches "Sarah Connor"
            query_params[key] = f"%{value}%"
        else:
            # Exact match — pass value as-is
            query_params[key] = value

    sql = template.sql_template.strip()

    # ── Execute against MySQL ─────────────────────────────────
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        start_time = time.perf_counter()
        cursor.execute(sql, query_params)
        rows: List[Dict[str, Any]] = cursor.fetchall()
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        cursor.close()

        # Build human-readable SQL for the Glass Box panel
        display_sql = sql
        for key, value in query_params.items():
            display_sql = display_sql.replace(
                f"%({key})s", f"'{value}'"
            )

        # ── Guard 3: Empty result set ─────────────────────────
        if not rows:
            healing_msg = _generate_healing_message(
                user_question=user_question,
                intent_description=template.description,
                missing_params=[],
                healing_reason="empty_result",
                base_url=base_url,
                synthesis_model=synthesis_model,
            )
            return DBResult(
                rows=[{"healing_message": healing_msg}],
                row_count=0,
                execution_time_ms=round(execution_time_ms, 3),
                query_executed=display_sql,
                params_used=query_params,
                error="empty_result",
                self_healing_triggered=True,
                healing_reason="Query returned 0 rows",
            )

        # ── Success ───────────────────────────────────────────
        serialized_rows = _serialize_rows(rows)

        return DBResult(
            rows=serialized_rows,
            row_count=len(serialized_rows),
            execution_time_ms=round(execution_time_ms, 3),
            query_executed=display_sql,
            params_used=query_params,
            error=None,
            self_healing_triggered=False,
        )

    except mysql.connector.Error as db_err:
        return DBResult(
            rows=[],
            row_count=0,
            execution_time_ms=0.0,
            query_executed=sql,
            params_used=query_params,
            error=f"Database error: {str(db_err)}",
            self_healing_triggered=False,
        )
    finally:
        if conn and conn.is_connected():
            conn.close()


def _serialize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MySQL-specific types to JSON-serializable Python types.

    Args:
        rows: Raw rows from MySQL cursor (dictionary=True mode).

    Returns:
        List of rows with all values converted to standard Python types.
    """
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