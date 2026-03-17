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



def _build_context(rows: List[Dict], user_question: str) -> str:
    if not rows:
        return "(no results)"

    q = user_question.lower()
    currency_fields = {"salary", "budget", "price", "project_budget"}

    wants_salary  = any(w in q for w in (
        "salary", "pay", "earn", "earns", "paid", "compensation",
        "income", "wage", "wages", "makes", "make",
        "highest paid", "lowest paid", "richest", "most expensive", "cheapest",
        "highest", "lowest", "most", "least", "rank", "difference",
    ))
    wants_project = any(w in q for w in (
        "project", "projects", "initiative", "program",
        "managing", "managed", "assigned", "assignment",
        "working on", "involved in", "work on",
    ))
    wants_dept = any(w in q for w in (
        "department", "dept", "team", "division", "group",
        "belong", "belongs", "which team", "which dept",
    ))
    wants_order = any(w in q for w in (
        "order", "orders", "purchase", "purchases", "bought", "ordered",
        "pending", "shipped", "delivered", "processing", "cancelled",
    ))
    wants_product = any(w in q for w in (
        "product", "item", "stock", "inventory", "price", "supplier",
        "category", "electronics", "furniture",
    ))

    suppress = {"manager_id_raw"}
    if not wants_salary:
        suppress.update({"salary", "budget", "project_budget"})
    if not wants_order:
        suppress.update({"order_id", "quantity", "order_date"})
    if not wants_product:
        suppress.update({"price", "stock_quantity", "category"})

    def fmt_val(k: str, v: Any) -> str:
        if k in currency_fields:
            try:
                return f"${float(v):,.2f}"
            except (TypeError, ValueError):
                pass
        return str(v)

    has_employee_col = (
        rows
        and ("employee_name" in rows[0] or "salary" in rows[0])
        and "order_id" not in rows[0]
    )

    if has_employee_col:
        seen_emp: Dict[str, Dict] = {}
        proj_map: Dict[str, List[str]] = {}

        for row in rows:
            emp = _get(row, "employee_name", "name")
            if not emp:
                continue
            proj  = _get(row, "project_name")
            asgn  = _get(row, "assignment_role")
            mgr   = _get(row, "manager_name")
            if emp not in seen_emp:
                seen_emp[emp] = row
                proj_map[emp] = []
            if proj:
                entry = proj
                if asgn:
                    entry += f" [{asgn}]"
                if mgr and wants_project:
                    entry += f" (managed by {mgr})"
                if entry not in proj_map[emp]:
                    proj_map[emp].append(entry)

        lines = []
        for emp, row in seen_emp.items():
            role = _get(row, "employee_role", "role")
            dept = _get(row, "department_name")
            sal  = _get(row, "salary")

            
            attrs = []
            if role:
                attrs.append(f"role: {role}")
            if dept and (wants_dept or not wants_project):
                attrs.append(f"department: {dept}")
            if sal is not None and wants_salary:
                attrs.append(f"salary: {fmt_val('salary', sal)}")
            if proj_map[emp] and (wants_project or not wants_dept):
                proj_str = "; ".join(proj_map[emp])
                attrs.append(f"projects: {proj_str}")

            attr_str = " | ".join(attrs)
            lines.append(f"- {emp} — {attr_str}" if attr_str else f"- {emp}")

        return "\n".join(lines) if lines else "(no results)"

    
    if rows and "order_id" in rows[0]:
        lines = []
        for row in rows:
            oid    = _get(row, "order_id")
            prod   = _get(row, "product_name")
            qty    = _get(row, "quantity")
            status = _get(row, "status")
            date   = _get(row, "order_date")
            emp    = _get(row, "employee_name")
            parts  = [f"Order #{oid}"]
            if prod:   parts.append(f"product: {prod}")
            if qty:    parts.append(f"qty: {qty}")
            if status: parts.append(f"status: {status}")
            if date:   parts.append(f"date: {date}")
            if emp:    parts.append(f"placed by: {emp}")
            lines.append("- " + " | ".join(parts))
        return "\n".join(lines)

    
    lines = []
    for row in rows:
        parts = []
        for k, v in row.items():
            if v is None or k in suppress:
                continue
            label = k.replace("_", " ")
            parts.append(f"{label}: {fmt_val(k, v)}")
        if parts:
            lines.append("- " + " | ".join(parts))

    return "\n".join(lines) if lines else "(no results)"



def _claude_synthesize(user_question: str, context: str) -> Optional[str]:
    """
    Synthesize a natural-language answer using the Anthropic Messages API.
    This produces significantly better prose than the local 1.5b model —
    richer formatting, better handling of lists, comparisons, and edge cases.
    """
    system_prompt = (
        "You are a concise, professional data assistant answering questions about "
        "a company's employees, departments, products, orders, and projects. "
        "You receive structured database results and convert them into clear, "
        "natural answers. You never mention databases, queries, or data structures."
    )

    user_prompt = (
        f"Question: {user_question}\n\n"
        f"Data:\n{context}\n\n"
        "Instructions:\n"
        "- Answer the question directly using only the data provided.\n"
        "- Single-fact questions (who earns the most, which project, etc.) → one sentence.\n"
        "- List questions → clean numbered or bulleted list, one item per line.\n"
        "- For employees: include name, role, and only the fields the question asks about.\n"
        "  If the question asks about projects, list their projects. If salary, show salary.\n"
        "  If department membership is the subject, show the department.\n"
        "- For orders: show order ID, product, status, date, and who placed it.\n"
        "- For comparisons or rankings: state the winner/answer first, then supporting detail.\n"
        "- Currency: $X,XXX.XX format. Dates: show as-is.\n"
        "- Do not pad. Do not summarise what you're about to say. Start the answer immediately.\n"
        "- Do not say 'Based on the data', 'The results show', 'Here is', or 'The following'.\n"
        "- Under 150 words unless a long list genuinely requires more.\n"
        "- Use markdown bold (**name**) for names and key values.\n"
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-5-20251001",
                "max_tokens": 600,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        answer = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ).strip()
        if answer:
            logger.info("Claude synthesis completed (%d chars)", len(answer))
            return answer
        return None

    except requests.RequestException as exc:
        logger.warning("Claude synthesis failed: %s — falling back to Ollama", exc)
        return None


def _ollama_synthesize(user_question: str, context: str, model_cfg: dict) -> Optional[str]:
    """
    Fallback synthesis via local Ollama model when Claude API is unavailable.
    """
    prompt = (
        "You are a concise data assistant. Convert the DB results below into a "
        "natural, direct answer to the question. Follow every rule exactly.\n\n"
        "RULES:\n"
        "1. Start immediately — no preamble, no 'Based on the data', "
        "no 'Here is', no 'The following'.\n"
        "2. Single-fact questions → ONE sentence maximum.\n"
        "3. List questions → clean numbered list, one item per line, "
        "only the fields the question asks about.\n"
        "4. Never say 'database', 'records', 'query', 'data provided', "
        "'the above', or 'results'.\n"
        "5. Currency: $X,XXX.XX. Dates: show as-is.\n"
        "6. Total answer under 120 words unless a long list requires more.\n\n"
        f"DB RESULTS:\n{context}\n\n"
        f"QUESTION: {user_question}\n\n"
        "ANSWER:"
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
                    "num_predict": 350,
                    "num_ctx": 2048,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        if answer:
            logger.info("Ollama synthesis completed (%d chars)", len(answer))
            return answer
        return None

    except requests.RequestException as exc:
        logger.warning("Ollama synthesis failed: %s — falling back to template", exc)
        return None


def _llm_synthesize(user_question: str, context: str, model_cfg: dict) -> Optional[str]:
    """
    Primary synthesis dispatcher.
    Tries Claude API first (high quality), falls back to Ollama (local),
    then to the template renderer if both are unavailable.
    """
    answer = _claude_synthesize(user_question, context)
    if answer:
        return answer
    return _ollama_synthesize(user_question, context, model_cfg)


def _format_single_row(row: Dict, user_question: str) -> Optional[str]:
    q = user_question.lower()
    order_id = _get(row, "order_id")
    if order_id is not None:
        status     = _get(row, "status", "order_status") or "unknown"
        product    = _get(row, "product_name", "product")
        quantity   = _get(row, "quantity")
        order_date = _get(row, "order_date")
        ordered_by = _get(row, "employee_name", "ordered_by")
        parts = [f"Order **#{order_id}**"]
        if product:    parts.append(f"for **{product}**")
        if quantity:   parts.append(f"(qty: {quantity})")
        parts.append(f"is **{status}**")
        if order_date: parts.append(f"· Placed: {order_date}")
        if ordered_by: parts.append(f"· By: {ordered_by}")
        return " ".join(parts) + "."

    name   = _get(row, "employee_name", "name")
    salary = _get(row, "salary")
    role   = _get(row, "employee_role", "role")
    dept   = _get(row, "department_name")
    if name and salary is not None:
        wants_salary = any(w in q for w in (
            "salary", "pay", "earn", "earns", "paid", "compensation",
        ))
        parts = [f"**{name}**"]
        if role: parts.append(f"({role})")
        if dept: parts.append(f"in **{dept}**")
        if wants_salary:
            parts.append(f"earns **{_fmt_currency(salary)}** per year")
        return " ".join(parts) + "."

    pname = _get(row, "product_name", "name")
    price = _get(row, "price")
    stock = _get(row, "stock_quantity")
    cat   = _get(row, "category")
    if pname and price is not None:
        parts = [f"**{pname}**"]
        if cat:  parts.append(f"({cat})")
        parts.append(f"costs **{_fmt_currency(price)}**")
        if stock is not None: parts.append(f"· {stock} in stock")
        return " ".join(parts) + "."

    proj_name   = _get(row, "project_name")
    proj_status = _get(row, "project_status")
    mgr         = _get(row, "manager_name")
    if proj_name:
        parts = [f"**{proj_name}**"]
        if mgr:         parts.append(f"is managed by **{mgr}**")
        if proj_status: parts.append(f"· Status: {proj_status}")
        return " ".join(parts) + "."

    return None


def _template_fallback(user_question: str, db_result: DBResult) -> str:
    rows = db_result.rows
    if not rows:
        return "No matching data found."
    if len(rows) == 1:
        single = _format_single_row(rows[0], user_question)
        if single:
            return single
    row_keys = set(rows[0].keys())
    lines = []
    if "order_id" in row_keys:
        for i, row in enumerate(rows, 1):
            oid  = _get(row, "order_id") or i
            prod = _get(row, "product_name", "product")
            qty  = _get(row, "quantity")
            st   = _get(row, "status", "order_status") or "unknown"
            dt   = _get(row, "order_date")
            by   = _get(row, "employee_name", "ordered_by")
            p = [f"#{oid}"]
            if prod: p.append(prod)
            if qty:  p.append(f"qty {qty}")
            p.append(st)
            if dt:   p.append(str(dt))
            if by:   p.append(f"by {by}")
            lines.append(f"{i}. " + " · ".join(p))
        return f"Found {len(rows)} orders:\n" + "\n".join(lines)

    if "employee_name" in row_keys or "salary" in row_keys:
        seen: Dict[str, Dict] = {}
        pm: Dict[str, List[str]] = {}
        for row in rows:
            n = _get(row, "employee_name", "name") or "Unknown"
            proj = _get(row, "project_name")
            if n not in seen:
                seen[n] = row; pm[n] = []
            if proj and proj not in pm[n]:
                pm[n].append(proj)
        for i, (n, row) in enumerate(seen.items(), 1):
            role = _get(row, "employee_role", "role")
            dept = _get(row, "department_name")
            sal  = _get(row, "salary")
            p = [n]
            if role: p.append(f"({role})")
            if dept: p.append(f"— {dept}")
            if pm[n]: p.append(f"→ {', '.join(pm[n])}")
            if sal is not None: p.append(f"— {_fmt_currency(sal)}")
            lines.append(f"{i}. " + " ".join(p))
        return f"Found {len(seen)} employees:\n" + "\n".join(lines)

    for i, row in enumerate(rows, 1):
        parts = [f"{k.replace('_',' ')}: {v}" for k, v in row.items() if v is not None]
        lines.append(f"{i}. " + " | ".join(parts))
    return f"Found {len(rows)} records:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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

    # Build a lean, deduplicated context — only relevant fields per query
    lean_context = _build_context(db_result.rows, user_question)

    # Always route through LLM synthesis for natural answers
    model_cfg = _load_model_config()
    llm_answer = _llm_synthesize(user_question, lean_context, model_cfg)
    if llm_answer:
        return llm_answer

    # Template fallback — only if LLM is unreachable
    logger.warning("LLM unavailable — using template fallback")
    return _template_fallback(user_question, db_result)
