from __future__ import annotations

import logging
from typing import Any

from middleware.models import DBResult, QueryTemplate

logger = logging.getLogger(__name__)


def format_context(db_result: DBResult, template: QueryTemplate) -> str:
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
        row = db_result.rows[0]
        for key, value in row.items():
            formatted_key = key.replace("_", " ").title()
            formatted_value = _format_value(key, value)
            lines.append(f"  {formatted_key}: {formatted_value}")
    else:
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
    if value is None:
        return "N/A"

    currency_fields = {"salary", "budget", "price", "project_budget"}
    if key in currency_fields:
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)

    if key == "stock_quantity" and isinstance(value, int):
        return f"{value} units"

    return str(value)
