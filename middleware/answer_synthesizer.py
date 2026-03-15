from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from middleware.models import DBResult

logger = logging.getLogger(__name__)


def _load_model_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


def _fmt_currency(val: Any) -> str:
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return str(val)


def _get(row: Dict, *keys: str, default=None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _needs_llm_synthesis(user_question: str, rows: List[Dict]) -> bool:
    if not rows:
        return False

    q = user_question.lower()

    comparison_keywords = [
        "highest", "lowest", "most", "least", "best", "worst", "top",
        "maximum", "minimum", "richest", "who earns", "who makes",
        "most expensive", "cheapest", "priciest", "highest paid",
        "lowest paid", "best paid", "rank", "compare",
    ]
    aggregation_keywords = [
        "how many", "count", "total", "average", "avg", "number of",
        "sum of", "breakdown",
    ]

    return any(kw in q for kw in comparison_keywords + aggregation_keywords)


def _format_single_row(row: Dict, user_question: str) -> Optional[str]:
    q = user_question.lower()

    # D6: order_id is the definitive anchor — check it before employee salary.
    # Expanded Order rows contain employee_name+salary which would otherwise
    # be misformatted as an employee record.
    order_id = _get(row, "order_id")
    if order_id is not None:
        status = _get(row, "status", "order_status") or "unknown"
        product = _get(row, "product_name", "product")
        quantity = _get(row, "quantity")
        order_date = _get(row, "order_date")
        ordered_by = _get(row, "employee_name", "ordered_by")
        parts = [f"Order **#{order_id}**"]
        if product:
            parts.append(f"for **{product}**")
        if quantity:
            parts.append(f"(qty: {quantity})")
        parts.append(f"is **{status}**")
        if order_date:
            parts.append(f"· Placed: {order_date}")
        if ordered_by:
            parts.append(f"· By: {ordered_by}")
        return " ".join(parts) + "."

    name = _get(row, "employee_name", "name")
    salary = _get(row, "salary")
    role = _get(row, "employee_role", "role")
    dept = _get(row, "department_name")
    hire_date = _get(row, "hire_date")
    email = _get(row, "email")

    if name and salary is not None:
        parts = [f"**{name}**"]
        if role:
            parts.append(f"({role})")
        if dept:
            parts.append(f"in **{dept}**")
        parts.append(f"earns **{_fmt_currency(salary)}** per year")
        if hire_date and "hire" in q:
            parts.append(f"· Hired: {hire_date}")
        if email and "email" in q:
            parts.append(f"· Email: {email}")
        return " ".join(parts) + "."

    pname = _get(row, "product_name", "name")
    stock = _get(row, "stock_quantity")
    price = _get(row, "price")
    supplier = _get(row, "supplier")
    category = _get(row, "category")

    if pname and stock is not None and "stock" in q:
        parts = [f"There are **{stock} units** of **{pname}** in stock"]
        if price is not None:
            parts.append(f"priced at **{_fmt_currency(price)}**")
        if supplier and "supplier" in q:
            parts.append(f"· Supplier: {supplier}")
        return " ".join(parts) + "."

    if pname and price is not None and stock is None:
        return f"**{pname}** ({category or 'Product'}) costs **{_fmt_currency(price)}**."

    if pname and price is not None:
        parts = [f"**{pname}**"]
        if category:
            parts.append(f"({category})")
        parts.append(f"costs **{_fmt_currency(price)}**")
        if stock is not None:
            parts.append(f"· {stock} units in stock")
        if supplier:
            parts.append(f"· Supplier: {supplier}")
        return " ".join(parts) + "."

    dname = _get(row, "department_name")
    budget = _get(row, "budget")
    location = _get(row, "location")
    headcount = _get(row, "headcount")
    manager_name = _get(row, "manager_name")

    if dname and (budget is not None or headcount is not None):
        parts = [f"The **{dname}** department"]
        if budget is not None:
            parts.append(f"has a budget of **{_fmt_currency(budget)}**")
        if headcount is not None:
            parts.append(f"with **{headcount} employees**")
        if manager_name:
            parts.append(f"· Managed by: **{manager_name}**")
        if location:
            parts.append(f"· Location: {location}")
        return " ".join(parts) + "."

    proj_name = _get(row, "project_name")
    proj_status = _get(row, "project_status")
    proj_budget = _get(row, "project_budget")
    start_date = _get(row, "start_date")
    end_date = _get(row, "end_date")

    if proj_name:
        parts = [f"**{proj_name}**"]
        if manager_name:
            parts.append(f"is managed by **{manager_name}**")
        if proj_status:
            parts.append(f"· Status: {proj_status}")
        if proj_budget is not None:
            parts.append(f"· Budget: {_fmt_currency(proj_budget)}")
        if start_date:
            date_str = f"· {start_date}"
            if end_date:
                date_str += f" → {end_date}"
            else:
                date_str += " → ongoing"
            parts.append(date_str)
        return " ".join(parts) + "."

    order_id = _get(row, "order_id", "id")
    status = _get(row, "status", "order_status")
    product = _get(row, "product")
    quantity = _get(row, "quantity")
    order_date = _get(row, "order_date")
    ordered_by = _get(row, "ordered_by")

    if order_id and status:
        parts = [f"Order **#{order_id}**"]
        if product:
            parts.append(f"for **{product}**")
        if quantity:
            parts.append(f"(qty: {quantity})")
        parts.append(f"is **{status}**")
        if order_date:
            parts.append(f"· Placed: {order_date}")
        if ordered_by:
            parts.append(f"· By: {ordered_by}")
        return " ".join(parts) + "."

    return None


def _format_employee_list(rows: List[Dict], user_question: str) -> str:
    q = user_question.lower()

    # ------------------------------------------------------------------ #
    # Deduplication: collapse multiple rows for the same employee into    #
    # one entry, collecting all project assignments into a single line.   #
    # ------------------------------------------------------------------ #
    # Key = (employee_name, role, salary, department)
    # Value = list of (project_name, assignment_role) tuples
    seen_employees: Dict[str, Dict] = {}   # name → merged row
    project_map: Dict[str, List[str]] = {} # name → [proj assignment strings]

    for row in rows:
        name = _get(row, "employee_name", "name") or "Unknown"
        proj = _get(row, "project_name")
        asgn = _get(row, "assignment_role")

        if name not in seen_employees:
            seen_employees[name] = row
            project_map[name] = []

        if proj:
            proj_str = proj
            if asgn:
                proj_str += f" [{asgn}]"
            if proj_str not in project_map[name]:
                project_map[name].append(proj_str)

    lines = []
    for i, (name, row) in enumerate(seen_employees.items(), 1):
        role = _get(row, "employee_role", "role")
        salary = _get(row, "salary")
        dept = _get(row, "department_name")
        hire_date = _get(row, "hire_date")

        parts = [f"**{name}**"]
        if role:
            parts.append(f"({role})")
        if dept and "department" not in q.split("in ")[-1][:20]:
            parts.append(f"— {dept}")

        # Render all projects on one line, comma-separated
        if project_map[name]:
            parts.append(f"→ {', '.join(project_map[name])}")

        if salary is not None:
            parts.append(f"— {_fmt_currency(salary)}")
        if hire_date and "hire" in q:
            parts.append(f"· Hired: {hire_date}")

        lines.append(f"{i}. {' '.join(parts)}")

    count = len(seen_employees)
    noun = "employee" if count == 1 else "employees"
    return f"Found **{count} {noun}**:\n\n" + "\n".join(lines)


def _format_product_list(rows: List[Dict]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        name = _get(row, "product_name", "name") or "Unknown"
        price = _get(row, "price")
        stock = _get(row, "stock_quantity")
        cat = _get(row, "category")
        supplier = _get(row, "supplier")

        parts = [f"**{name}**"]
        if cat:
            parts.append(f"({cat})")
        if price is not None:
            parts.append(f"— {_fmt_currency(price)}")
        if stock is not None:
            parts.append(f"· {stock} in stock")
        if supplier:
            parts.append(f"· {supplier}")

        lines.append(f"{i}. {' '.join(parts)}")

    return f"Found **{len(rows)} products**:\n\n" + "\n".join(lines)


def _format_department_list(rows: List[Dict]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        name = _get(row, "department_name", "name") or "Unknown"
        budget = _get(row, "budget")
        headcount = _get(row, "headcount")
        location = _get(row, "location")
        manager_name = _get(row, "manager_name")

        parts = [f"**{name}**"]
        if budget is not None:
            parts.append(f"— {_fmt_currency(budget)}")
        if headcount is not None:
            parts.append(f"· {headcount} employees")
        if manager_name:
            parts.append(f"· Mgr: {manager_name}")
        if location:
            parts.append(f"· {location}")

        lines.append(f"{i}. {' '.join(parts)}")

    return f"Found **{len(rows)} departments**:\n\n" + "\n".join(lines)


def _format_order_list(rows: List[Dict]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        oid = _get(row, "order_id", "id") or i
        # D6: expanded Order path produces 'product_name' and 'employee_name'
        # (not 'product' / 'ordered_by' which were the old intents.yaml aliases)
        product = _get(row, "product_name", "product")
        quantity = _get(row, "quantity")
        status = _get(row, "status", "order_status") or "unknown"
        date = _get(row, "order_date")
        ordered_by = _get(row, "employee_name", "ordered_by")

        parts = [f"Order **#{oid}**"]
        if product:
            parts.append(f"— {product}")
        if quantity:
            parts.append(f"(qty: {quantity})")
        parts.append(f"**{status}**")
        if date:
            parts.append(f"· {date}")
        if ordered_by:
            parts.append(f"· {ordered_by}")

        lines.append(f"{i}. {' '.join(parts)}")

    return f"Found **{len(rows)} orders**:\n\n" + "\n".join(lines)


def _format_project_list(rows: List[Dict]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        name = _get(row, "project_name") or "Unknown"
        mgr = _get(row, "manager_name")
        status = _get(row, "project_status")
        budget = _get(row, "project_budget")
        start = _get(row, "start_date")
        end = _get(row, "end_date")

        parts = [f"**{name}**"]
        if mgr:
            parts.append(f"(by {mgr})")
        if status:
            parts.append(f"· {status}")
        if budget is not None:
            parts.append(f"· {_fmt_currency(budget)}")
        if start:
            date_str = f"· {start}"
            if end:
                date_str += f" → {end}"
            else:
                date_str += " → ongoing"
            parts.append(date_str)

        lines.append(f"{i}. {' '.join(parts)}")

    count = len(rows)
    noun = "project" if count == 1 else "projects"
    return f"Found **{count} {noun}**:\n\n" + "\n".join(lines)


def _try_template_answer(
    user_question: str,
    db_result: DBResult,
) -> Optional[str]:
    rows = db_result.rows
    if not rows:
        return None

    if _needs_llm_synthesis(user_question, rows):
        return None

    if len(rows) == 1:
        formatted = _format_single_row(rows[0], user_question)
        if formatted:
            return formatted

    row_keys = set(rows[0].keys())

    # D6: order_id is a definitive discriminant — check it FIRST.
    # Expanded Order queries (path=[Order, Employee, Product]) include employee_name
    # and salary columns, which would otherwise trigger has_employee and return
    # "Found N employees" instead of "Found N orders".
    has_order = (
        ("status" in row_keys or "order_status" in row_keys)
        and "order_id" in row_keys
    )
    has_employee = "employee_name" in row_keys or (
        "name" in row_keys and "salary" in row_keys
    )
    has_product = "product_name" in row_keys or (
        "name" in row_keys and "price" in row_keys
    )
    has_department = "department_name" in row_keys or (
        "name" in row_keys and "budget" in row_keys
    )
    has_project = "project_name" in row_keys and "employee_name" not in row_keys

    if has_order:
        return _format_order_list(rows)

    if has_employee:
        return _format_employee_list(rows, user_question)

    if has_product:
        return _format_product_list(rows)

    if has_department:
        return _format_department_list(rows)

    if has_project:
        return _format_project_list(rows)

    if has_order:
        return _format_order_list(rows)

    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in row.items() if v is not None]
        lines.append(f"{i}. {' | '.join(parts)}")

    return f"Found **{len(rows)} records**:\n\n" + "\n".join(lines)


def _llm_synthesize(user_question: str, context: str, model_cfg: dict) -> str:
    prompt = (
        "You are a concise data analyst. Answer the question using ONLY the data provided below.\n"
        "Rules:\n"
        "- Use bullet points for lists\n"
        "- Format currency as $X,XXX.XX\n"
        "- Be specific and direct\n"
        "- Never say 'database' or 'records' or 'query'\n"
        "- If comparing, clearly state who/what is highest/lowest\n\n"
        f"{context}\n\n"
        f"Question: {user_question}\n"
        "Answer:"
    )

    try:
        resp = requests.post(
            f"{model_cfg['ollama_base_url']}/api/generate",
            json={
                "model": model_cfg["synthesis_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                    "num_ctx": 2048,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        logger.info("LLM synthesis completed (%d chars)", len(answer))
        return answer if answer else context

    except requests.RequestException as exc:
        logger.warning("LLM synthesis failed: %s — falling back to raw context", exc)
        return context


def synthesize_answer(
    user_question: str,
    context: str,
    db_result: DBResult,
) -> str:
    if db_result.self_healing_triggered:
        if db_result.rows and "healing_message" in db_result.rows[0]:
            return db_result.rows[0]["healing_message"]
        return (
            "I couldn't find any matching data for that question. "
            "Please check your search terms and try again."
        )

    if db_result.row_count == 0:
        return (
            "I couldn't find any matching data for that question. "
            "Try rephrasing or verify the name, department, or product is correct."
        )

    template_answer = _try_template_answer(user_question, db_result)
    if template_answer:
        return template_answer

    logger.info("Falling through to LLM synthesis for comparison/aggregation query")
    model_cfg = _load_model_config()
    return _llm_synthesize(user_question, context, model_cfg)
