"""
middleware/context_formatter.py

Converts raw MySQL result rows into a clean, structured text block
that the LLM can reason about effectively.

Why this step exists:
  Dumping raw JSON into an LLM prompt works, but LLMs perform better
  when context is formatted as natural structured text. This module
  bridges the gap between database output and LLM input.

No LLM calls here. No database calls here.
Pure Python transformation: List[Dict] → str.
"""

from __future__ import annotations

from typing import Any, Dict, List

from middleware.models import DBResult, QueryTemplate


def format_context(db_result: DBResult, template: QueryTemplate) -> str:
    """Format database rows into a clean context string for the LLM.

    Produces a structured text block that includes:
    - What was queried (from template description)
    - How many results were found
    - The data itself, formatted as readable key-value pairs
    - A clear instruction boundary so the LLM knows where data ends

    Args:
        db_result: The DBResult from query_executor.
        template: The QueryTemplate used (provides description context).

    Returns:
        str: A formatted context string ready to inject into the LLM prompt.

    Example:
        >>> context = format_context(db_result, template)
        >>> print(context)
        --- DATABASE RESULTS ---
        Query: Get salary information for a specific employee
        Records found: 1
        ...
    """
    # Handle self-healing case — no real data to format
    if db_result.self_healing_triggered:
        if db_result.rows and "healing_message" in db_result.rows[0]:
            return db_result.rows[0]["healing_message"]
        return "No data available."

    if db_result.row_count == 0:
        return f"No records found for: {template.description}"

    lines = [
        "--- DATABASE RESULTS ---",
        f"Query type: {template.description}",
        f"Records found: {db_result.row_count}",
        "",
    ]

    if db_result.row_count == 1:
        # Single result — format as clean key-value pairs
        row = db_result.rows[0]
        for key, value in row.items():
            formatted_key = key.replace("_", " ").title()
            formatted_value = _format_value(key, value)
            lines.append(f"  {formatted_key}: {formatted_value}")
    else:
        # Multiple results — format as a numbered list
        for i, row in enumerate(db_result.rows, start=1):
            lines.append(f"  Record {i}:")
            for key, value in row.items():
                formatted_key = key.replace("_", " ").title()
                formatted_value = _format_value(key, value)
                lines.append(f"    {formatted_key}: {formatted_value}")
            lines.append("")

    lines.append("--- END OF DATABASE RESULTS ---")
    return "\n".join(lines)


def _format_value(key: str, value: Any) -> str:
    """Format a single value for human-readable display.

    Applies currency formatting to salary/price/budget fields,
    and handles None values gracefully.

    Args:
        key: The column name (used to detect currency fields).
        value: The raw value from the database row.

    Returns:
        str: A human-readable string representation of the value.
    """
    if value is None:
        return "N/A"

    # Currency formatting for financial fields
    currency_fields = {"salary", "budget", "price"}
    if key in currency_fields and isinstance(value, (int, float)):
        return f"${value:,.2f}"

    # Stock quantity with unit
    if key == "stock_quantity" and isinstance(value, int):
        return f"{value} units"

    return str(value)