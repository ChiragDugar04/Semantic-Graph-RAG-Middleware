"""
middleware/answer_synthesizer.py

Generates the final natural language answer.

SPEED STRATEGY (3-tier):
  Tier 0 — Template bypass (0ms):
    Simple list/lookup results with ≤20 rows are formatted directly
    in Python. No LLM call at all. Handles the majority of queries.

  Tier 1 — Fast LLM (qwen2.5:1.5b, ~2-5s):
    Used for comparison, aggregation, cross-entity, and yes/no questions
    where natural language reasoning adds real value.

  Tier 2 — Fallback (same fast model, context-only):
    If LLM call fails, return the raw formatted context so the user
    always gets an answer.
"""

from __future__ import annotations

import requests
import yaml

from pathlib import Path
from typing import Any, Dict, List

from middleware.models import DBResult, QueryTemplate


def _load_model_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


# ============================================================
# TIER 0 — TEMPLATE BYPASS (pure Python, 0ms)
# ============================================================

def _try_template_answer(
    user_question: str,
    context: str,
    db_result: DBResult,
) -> str | None:
    """
    Try to answer directly from structured data without an LLM call.
    Returns a string if successful, None to fall through to LLM.

    Handles:
      - Single-row lookups (salary, role, stock, budget)
      - Multi-row list results (employees, products, orders)
    Does NOT handle: comparisons, aggregations, yes/no questions —
    those need LLM reasoning.
    """
    rows = db_result.rows
    if not rows:
        return None

    q = user_question.lower()

    # ── Single row — direct field answer ─────────────────────
    if len(rows) == 1:
        row = rows[0]
        keys = set(row.keys())

        # Salary lookup
        if "salary" in keys and "name" in keys:
            name   = row["name"]
            salary = row["salary"]
            role   = row.get("role", "")
            role_str = f", {role}" if role else ""
            return f"{name}{role_str} earns **${salary:,.2f}** per year."

        # Stock quantity
        if "stock_quantity" in keys and "name" in keys:
            name = row["name"]
            qty  = row["stock_quantity"]
            return f"There are **{qty} units** of {name} in stock."

        # Budget
        if "budget" in keys and "name" in keys:
            name   = row["name"]
            budget = row["budget"]
            return f"The {name} department has a budget of **${budget:,.2f}**."

        # Price (single product)
        if "price" in keys and "name" in keys:
            name  = row["name"]
            price = row["price"]
            return f"{name} costs **${price:,.2f}**."

        return None  # Single row but unknown shape — use LLM

    # ── Multi-row list results ────────────────────────────────
    if len(rows) > 1:
        keys = set(rows[0].keys())

        # List of employees (with optional salary/role/dept)
        if "name" in keys and not any(
            kw in q for kw in ["highest", "lowest", "most", "least",
                                "best", "worst", "top", "maximum", "minimum",
                                "who earns", "who makes", "expensive", "cheapest",
                                "how many", "count", "total", "average"]
        ):
            lines = []
            for i, row in enumerate(rows, 1):
                name = row.get("name") or row.get("employee_name", "Unknown")
                parts = [f"**{name}**"]
                if "role" in row and row["role"]:
                    parts.append(f"({row['role']})")
                if "department" in row and row["department"]:
                    parts.append(f"— {row['department']}")
                if "project_name" in row and row["project_name"]:
                    parts.append(f"— {row['project_name']}")
                if "salary" in row and row["salary"]:
                    parts.append(f"— ${row['salary']:,.2f}")
                lines.append(f"{i}. {' '.join(parts)}")

            count = len(rows)
            noun  = "employee" if count == 1 else "employees"
            header = f"Found **{count} {noun}**:\n\n"
            return header + "\n".join(lines)

        # List of products
        if "price" in keys and "name" in keys and len(rows) > 1:
            if not any(kw in q for kw in ["highest", "most expensive", "cheapest"]):
                lines = []
                for i, row in enumerate(rows, 1):
                    name  = row.get("name", "Unknown")
                    price = row.get("price", 0)
                    stock = row.get("stock_quantity", "")
                    stock_str = f" · {stock} in stock" if stock != "" else ""
                    lines.append(f"{i}. **{name}** — ${price:,.2f}{stock_str}")
                return f"Found **{len(rows)} products**:\n\n" + "\n".join(lines)

        # List of orders
        if "status" in keys and any(kw in q for kw in ["order", "orders", "purchase"]):
            lines = []
            for i, row in enumerate(rows, 1):
                status   = row.get("status", "")
                total    = row.get("total_amount", "")
                date     = row.get("order_date", "")
                total_str = f" · ${total:,.2f}" if total else ""
                date_str  = f" · {date}" if date else ""
                lines.append(f"{i}. Order #{row.get('id', i)} — **{status}**{total_str}{date_str}")
            return f"Found **{len(rows)} orders**:\n\n" + "\n".join(lines)

    return None  # Fall through to LLM


# ============================================================
# TIER 1 — FAST LLM SYNTHESIS
# ============================================================

def _llm_synthesize(
    user_question: str,
    context: str,
    model_cfg: dict,
) -> str:
    """Call the fast LLM for questions requiring reasoning."""
    synthesis_model = model_cfg["synthesis_model"]
    base_url        = model_cfg["ollama_base_url"]
    temperature     = model_cfg.get("synthesis_temperature", 0.1)
    max_tokens      = model_cfg.get("max_tokens_synthesis", 200)

    # Tight prompt — fewer tokens in = faster response
    prompt = (
        "You are a data assistant. Answer the question using ONLY the data below.\n"
        "Be concise. Use bullet points for lists. Format currency with $ and commas.\n"
        "Never mention 'database', 'query', or 'records'.\n\n"
        f"{context}\n\n"
        f"Question: {user_question}\n"
        "Answer:"
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
                    "num_ctx": 2048,   # smaller context window = faster
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
    """Generate a natural language answer from database context.

    Uses a 3-tier strategy:
      0. Template bypass (Python only, 0ms) for simple results
      1. Fast LLM (qwen2.5:1.5b) for reasoning-heavy questions
      2. Raw context fallback if LLM fails

    Args:
        user_question: The original user question.
        context: Formatted context string from context_formatter.
        db_result: The DBResult object.

    Returns:
        str: The final answer.
    """
    if db_result.self_healing_triggered:
        return context

    if db_result.row_count == 0:
        return (
            "I couldn't find any matching data for that question. "
            "Try rephrasing or check that the name/department is correct."
        )

    # ── Tier 0: try Python template answer first ─────────────
    template_answer = _try_template_answer(user_question, context, db_result)
    if template_answer:
        return template_answer

    # ── Tier 1: LLM for complex reasoning ────────────────────
    model_cfg = _load_model_config()
    return _llm_synthesize(user_question, context, model_cfg)
