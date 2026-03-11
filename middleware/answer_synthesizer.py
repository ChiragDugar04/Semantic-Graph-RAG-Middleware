"""
middleware/answer_synthesizer.py

Generates the final natural language answer.

SPEED STRATEGY (3-tier):
  Tier 0 — Template bypass (0ms):
    Formats results directly in Python using the aliased column names
    that graph_query_builder produces (employee_name, salary, etc.)
    Handles the majority of queries with zero LLM calls.

  Tier 1 — Fast LLM (qwen2.5:1.5b, ~2-5s):
    Used only for questions where NL reasoning genuinely adds value:
    comparisons, aggregations, yes/no questions.

  Tier 2 — Fallback:
    Returns raw formatted context if LLM fails.
"""

from __future__ import annotations

import requests
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional

from middleware.models import DBResult


def _load_model_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


# ============================================================
# HELPERS — column extraction with alias fallbacks
# ============================================================

def _get(row: Dict, *keys: str, default=None):
    """Get the first matching key from a row dict."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _fmt_currency(val) -> str:
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return str(val)


def _is_comparison(q: str) -> bool:
    keywords = [
        "highest", "lowest", "most", "least", "best", "worst", "top",
        "maximum", "minimum", "richest", "who earns", "who makes",
        "most expensive", "cheapest", "highest paid", "lowest paid",
    ]
    return any(kw in q for kw in keywords)


def _is_aggregation(q: str) -> bool:
    keywords = ["how many", "count", "total", "average", "avg", "number of"]
    return any(kw in q for kw in keywords)


# ============================================================
# TIER 0 — TEMPLATE BYPASS (pure Python, 0ms)
# ============================================================

def _try_template_answer(
    user_question: str,
    db_result: DBResult,
) -> Optional[str]:
    """
    Format the answer directly from structured DB rows.
    Returns None to fall through to LLM for complex reasoning.

    Handles graph_query_builder aliased columns:
      employee_name, employee_role, salary, department_name,
      project_name, assignment_role, product_name, price, etc.
    """
    rows = db_result.rows
    if not rows:
        return None

    q = user_question.lower()

    # Skip template for queries that need ranking/comparison/aggregation
    if _is_comparison(q) or _is_aggregation(q):
        return None

    # ── Single row lookup ─────────────────────────────────────
    if len(rows) == 1:
        row = rows[0]

        # Salary lookup: "what is X's salary"
        name   = _get(row, "employee_name", "name")
        salary = _get(row, "salary")
        role   = _get(row, "employee_role", "role")
        if name and salary is not None:
            role_str = f", {role}" if role else ""
            dept = _get(row, "department_name")
            dept_str = f" ({dept})" if dept else ""
            return f"**{name}**{role_str}{dept_str} earns **{_fmt_currency(salary)}** per year."

        # Stock quantity
        pname = _get(row, "product_name", "name")
        stock = _get(row, "stock_quantity")
        if pname and stock is not None:
            return f"There are **{stock} units** of {pname} in stock."

        # Department budget
        dname  = _get(row, "department_name", "name")
        budget = _get(row, "budget")
        if dname and budget is not None and "budget" in q:
            return f"The **{dname}** department has a budget of **{_fmt_currency(budget)}**."

        # Single product price
        price = _get(row, "price")
        if pname and price is not None:
            return f"**{pname}** costs **{_fmt_currency(price)}**."

        return None  # Unknown single-row shape — use LLM

    # ── Multi-row results ─────────────────────────────────────
    rows_keys = set(rows[0].keys())

    # ── Employee list (with optional salary / dept / project) ─
    has_emp_name = "employee_name" in rows_keys or "name" in rows_keys
    if has_emp_name:
        lines = []
        for i, row in enumerate(rows, 1):
            name   = _get(row, "employee_name", "name") or "Unknown"
            role   = _get(row, "employee_role", "role")
            salary = _get(row, "salary")
            dept   = _get(row, "department_name")
            proj   = _get(row, "project_name")
            asgn   = _get(row, "assignment_role")

            parts = [f"**{name}**"]
            if role:
                parts.append(f"({role})")
            if dept and dept.lower() not in q:  # skip dept if already in question
                parts.append(f"— {dept}")
            if proj:
                parts.append(f"— {proj}")
            if asgn:
                parts.append(f"[{asgn}]")
            if salary is not None:
                parts.append(f"— {_fmt_currency(salary)}")

            lines.append(f"{i}. {' '.join(parts)}")

        count = len(rows)
        noun  = "employee" if count == 1 else "employees"
        return f"Found **{count} {noun}**:\n\n" + "\n".join(lines)

    # ── Product list ───────────────────────────────────────────
    if "product_name" in rows_keys or ("name" in rows_keys and "price" in rows_keys):
        lines = []
        for i, row in enumerate(rows, 1):
            name  = _get(row, "product_name", "name") or "Unknown"
            price = _get(row, "price")
            stock = _get(row, "stock_quantity")
            cat   = _get(row, "category")
            parts = [f"**{name}**"]
            if cat:
                parts.append(f"({cat})")
            if price is not None:
                parts.append(f"— {_fmt_currency(price)}")
            if stock is not None:
                parts.append(f"· {stock} in stock")
            lines.append(f"{i}. {' '.join(parts)}")
        return f"Found **{len(rows)} products**:\n\n" + "\n".join(lines)

    # ── Order list ─────────────────────────────────────────────
    if "status" in rows_keys and any(kw in q for kw in ["order", "orders", "purchase"]):
        lines = []
        for i, row in enumerate(rows, 1):
            status = _get(row, "status", "order_status") or "unknown"
            oid    = _get(row, "order_id", "id") or i
            total  = _get(row, "total_amount")
            date   = _get(row, "order_date")
            parts  = [f"Order #{oid}", f"**{status}**"]
            if total:
                parts.append(f"— {_fmt_currency(total)}")
            if date:
                parts.append(f"({date})")
            lines.append(f"{i}. {' '.join(parts)}")
        return f"Found **{len(rows)} orders**:\n\n" + "\n".join(lines)

    return None  # Unknown shape — fall through to LLM


# ============================================================
# TIER 1 — FAST LLM
# ============================================================

def _llm_synthesize(user_question: str, context: str, model_cfg: dict) -> str:
    synthesis_model = model_cfg["synthesis_model"]
    base_url        = model_cfg["ollama_base_url"]
    temperature     = model_cfg.get("synthesis_temperature", 0.1)
    max_tokens      = model_cfg.get("max_tokens_synthesis", 200)

    prompt = (
        "You are a data assistant. Answer concisely using ONLY the data below.\n"
        "Rules: use bullet points for lists, format currency as $X,XXX.XX, "
        "never say 'database' or 'records'.\n\n"
        f"{context}\n\n"
        f"Question: {user_question}\nAnswer:"
    )

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": synthesis_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 2048,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return answer if answer else context
    except requests.RequestException:
        return context


# ============================================================
# PUBLIC API
# ============================================================

def synthesize_answer(
    user_question: str,
    context: str,
    db_result: DBResult,
) -> str:
    if db_result.self_healing_triggered:
        return context

    if db_result.row_count == 0:
        return (
            "I couldn't find any matching data for that question. "
            "Try rephrasing or check that the name/department/project is correct."
        )

    # Tier 0: pure Python formatting — covers most queries instantly
    template_answer = _try_template_answer(user_question, db_result)
    if template_answer:
        return template_answer

    # Tier 1: LLM for comparisons, aggregations, complex reasoning
    model_cfg = _load_model_config()
    return _llm_synthesize(user_question, context, model_cfg)
