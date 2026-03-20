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


# ---------------------------------------------------------------------------
# Row-shape detectors
# These functions look at the keys present in the first result row to decide
# which context builder to use.  All detection is structural (key presence),
# not based on question text — so they work for any domain.
# ---------------------------------------------------------------------------

def _is_delta_row(row: Dict) -> bool:
    """True when the row contains a computed delta result."""
    keys = set(row.keys())
    # Delta rows always have exactly one of the *_gap aliases
    return any(k.endswith("_gap") for k in keys) or (
        any(k.startswith("highest_") for k in keys)
        and any(k.startswith("lowest_") for k in keys)
    )


def _is_having_count_row(row: Dict) -> bool:
    """True when the row contains a having_count result (employee + project_count)."""
    keys = set(row.keys())
    return "project_count" in keys or (
        "employee_name" in keys and any(k.endswith("_count") for k in keys)
    )


def _is_temporal_row(row: Dict) -> bool:
    """
    True ONLY when the row is the direct result of a temporal_filter query.

    Employee+project rows also carry start_date/end_date as project
    attributes — they must NOT be routed to the temporal context builder.
    Two structural signals distinguish a real temporal row:

    * employee_name present  -> employee-centric result; dates are project
      attributes, not the filter axis. Route to employee branch.
    * project_budget present -> plain project lookup/list row. The temporal
      query builder never selects project_budget. Route to generic fallback.
    """
    keys = set(row.keys())
    has_date = "start_date" in keys or "end_date" in keys or "order_date" in keys
    if not has_date:
        return False
    if "employee_name" in keys:   # employee+project join row
        return False
    if "project_budget" in keys:  # plain project lookup row
        return False
    return True


# ---------------------------------------------------------------------------
# Specialised context builders for new question types
# Each produces a terse, LLM-readable summary of the rows.
# ---------------------------------------------------------------------------

def _build_delta_context(rows: List[Dict], user_question: str) -> str:
    """
    Context for computed_delta results.
    Expected row shape: {highest_X, lowest_X, X_gap} — one row.
    Formats currency values automatically by detecting is_currency from column names.
    """
    if not rows:
        return "(no results)"

    row = rows[0]
    lines = []
    for k, v in row.items():
        if v is None:
            continue
        label = k.replace("_", " ")
        # Currency detection: column name contains salary, price, budget, gap, highest, lowest
        is_currency = any(
            word in k.lower()
            for word in ("salary", "price", "budget", "gap", "highest", "lowest", "pay")
        )
        formatted = _fmt_currency(v) if is_currency else str(v)
        lines.append(f"- {label}: {formatted}")

    return "\n".join(lines) if lines else "(no results)"


def _build_having_count_context(rows: List[Dict], user_question: str) -> str:
    """
    Context for having_count results.
    Expected row shape: {employee_name, project_count, employee_role?, department_name?}
    Produces one line per person with their count and supporting attributes.
    """
    if not rows:
        return "(no results)"

    lines = []
    for row in rows:
        name  = _get(row, "employee_name", "name")
        count_val = None
        count_label = "items"
        for k, v in row.items():
            if k.endswith("_count") and v is not None:
                count_val   = v
                count_label = k.replace("_count", "s").replace("_", " ")
                break
        role = _get(row, "employee_role", "role")
        dept = _get(row, "department_name")

        parts = []
        if name:
            parts.append(name)
        if count_val is not None:
            parts.append(f"{count_val} {count_label}")
        if role:
            parts.append(f"({role})")
        if dept:
            parts.append(f"— {dept}")
        lines.append("- " + " | ".join(parts) if parts else "- (unknown)")

    return "\n".join(lines)


def _build_temporal_context(rows: List[Dict], user_question: str) -> str:
    """
    Context for temporal_filter results.
    Works for both Project rows (start_date, end_date, project_status) and
    Order rows (order_date, status, product_name).  Detection is structural.
    """
    if not rows:
        return "(no results)"

    q = user_question.lower()
    is_order = "order_id" in rows[0]
    is_project = "project_name" in rows[0] or "start_date" in rows[0]

    lines = []
    for row in rows:
        if is_order:
            oid    = _get(row, "order_id")
            prod   = _get(row, "product_name")
            status = _get(row, "status")
            date   = _get(row, "order_date")
            emp    = _get(row, "employee_name")
            parts  = [f"Order #{oid}"] if oid else []
            if prod:   parts.append(f"product: {prod}")
            if status: parts.append(f"status: {status}")
            if date:   parts.append(f"date: {date}")
            if emp:    parts.append(f"placed by: {emp}")
            lines.append("- " + " | ".join(parts))
        elif is_project:
            pname  = _get(row, "project_name")
            status = _get(row, "project_status")
            start  = _get(row, "start_date")
            end    = _get(row, "end_date")
            mgr    = _get(row, "manager_name")
            parts  = [pname] if pname else ["(unnamed)"]
            if status: parts.append(f"status: {status}")
            if start:  parts.append(f"started: {start}")
            if end:    parts.append(f"ends: {end}")
            if mgr:    parts.append(f"manager: {mgr}")
            lines.append("- " + " | ".join(parts))
        else:
            # Generic fallback for any other temporal entity
            parts = [f"{k.replace('_', ' ')}: {v}" for k, v in row.items() if v is not None]
            lines.append("- " + " | ".join(parts))

    return "\n".join(lines) if lines else "(no results)"


# ---------------------------------------------------------------------------
# Main context builder — routes to the right specialist based on row shape,
# or falls back to the original generic builder.
# ---------------------------------------------------------------------------

def _build_context(rows: List[Dict], user_question: str, question_type: str = "") -> str:
    if not rows:
        return "(no results)"

    first = rows[0]

    if question_type == "computed_delta" or _is_delta_row(first):
        return _build_delta_context(rows, user_question)

    if question_type == "having_count" or _is_having_count_row(first):
        return _build_having_count_context(rows, user_question)

    if question_type == "temporal_filter" or _is_temporal_row(first):
        return _build_temporal_context(rows, user_question)

    if question_type == "comparison":
        rows = rows[:1]

    q = user_question.lower()
    currency_fields = {"salary", "budget", "price", "project_budget"}

    wants_salary  = any(w in q for w in (
        "salary", "pay", "earn", "earns", "paid", "compensation",
        "income", "wage", "wages", "makes", "make",
        "highest paid", "lowest paid", "richest", "most expensive", "cheapest",
        "highest", "lowest", "most", "least", "rank", "difference",
        "cost", "costs", "budget", "price", "how much",
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

        is_listing_people_on_project = (
            wants_project
            and len(rows) > 1
            and "project_name" in rows[0]
            and len({r.get("project_name") for r in rows}) == 1
        )

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
                # Only append manager when showing an employee's projects list,
                # NOT when listing the people assigned to a single project —
                # in that case manager_name is the same for every row and
                # causes the LLM to write about the manager, not the people.
                if mgr and wants_project and not is_listing_people_on_project:
                    entry += f" (managed by {mgr})"
                if entry not in proj_map[emp]:
                    proj_map[emp].append(entry)

        lines = []
        for emp, row in seen_emp.items():
            role = _get(row, "employee_role", "role")
            dept = _get(row, "department_name")
            sal  = _get(row, "salary")
            mgr  = _get(row, "manager_name")

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
            # Emit manager_name when it is in the row but not already expressed
            # via a project attribution.  This surfaces the dept manager for
            # questions like "who manages the team Sarah Connor works in".
            if mgr and not proj_map[emp]:
                attrs.append(f"department manager: {mgr}")

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


# ---------------------------------------------------------------------------
# LLM synthesis — Ollama only.
# ---------------------------------------------------------------------------

def _build_subject_header(filters: Dict[str, Any]) -> str:
    # manager_name and reports_to_name are scope/filter keys — the subject
    # is the people returned, not the manager.  Including them as SUBJECT
    # causes the LLM to write about the manager instead of listing results.
    _SUBJECT_KEYS = {
        "employee_name", "product_name", "project_name", "department_name",
    }
    subjects = [str(v) for k, v in filters.items() if k in _SUBJECT_KEYS and v]
    if subjects:
        return f"SUBJECT: {', '.join(subjects)}\n"
    return ""


def _ollama_synthesize(user_question: str, context: str, row_count: int, model_cfg: dict) -> Optional[str]:
    prompt = (
        "You are a data assistant. The DB RESULTS below are ground truth — "
        "they come directly from a live database query. "
        "Your only job is to express those results as a clear answer.\n\n"
        "RULES:\n"
        "1. NEVER contradict the DB RESULTS. If the results show rows, "
        "those rows exist — do not say 'no records' or 'not found'.\n"
        "2. The SUBJECT line names who or what the question is about — "
        "always mention the subject by name in your answer.\n"
        f"3. There are {row_count} row(s) in DB RESULTS — you MUST mention "
        f"ALL {row_count} items by name. Do not skip any.\n"
        "4. Start immediately — no preamble, no 'Based on the data', "
        "no 'Here is', no 'The following'.\n"
        "5. Single-fact questions → ONE sentence maximum.\n"
        "6. List questions → clean numbered list, one item per line.\n"
        "7. Never say 'database', 'records', 'query', or 'results'.\n"
        "8. Currency values are already formatted — copy them exactly as shown.\n"
        "9. Answer only from the DB RESULTS — do not add outside knowledge.\n"
        "10. For people lists: list every person's NAME from the DB RESULTS. "
        "Do not list roles instead of names.\n\n"
        f"DB RESULTS:\n{context}\n\n"
        f"QUESTION: {user_question}\n\n"
        "ANSWER:"
    )

    timeout = model_cfg.get("ollama_timeout", 180)
    try:
        resp = requests.post(
            f"{model_cfg['ollama_base_url']}/api/generate",
            json={
                "model": model_cfg["synthesis_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": model_cfg.get("synthesis_temperature", 0.1),
                    "num_predict": model_cfg.get("max_tokens_synthesis", 350),
                    "num_ctx": model_cfg.get("synthesis_num_ctx", 2048),
                },
                "keep_alive": model_cfg.get("keep_alive", "10m"),
            },
            timeout=timeout,
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


def _llm_synthesize(user_question: str, context: str, row_count: int, model_cfg: dict) -> Optional[str]:
    return _ollama_synthesize(user_question, context, row_count, model_cfg)


def _extract_context_names(context: str) -> List[str]:
    """
    Extract meaningful name tokens from context lines for completeness checking.

    Handles four line shapes produced by _build_context:

    1. Employee lines:  "- Don Draper — role: ..."
       → leading token before " — " = "Don Draper"

    2. Generic/product: "- product name: Laptop Pro 15 | ..."
       → leading token = "product name: Laptop Pro 15"
       → strip "key: " prefix → "Laptop Pro 15"

    3. Project lines:   "- Platform Migration | status: ..."
       → leading token = "Platform Migration"

    4. Order lines:     "- Order #12 | product: Wireless Mouse | ... | placed by: Dwight Schrute"
       → the order number is NOT reliable (LLM may omit it); instead extract
         the product name and placed-by name from the full line segments.

    Returns a deduplicated list of name strings that must appear in the answer.
    """
    import re as _re
    names: List[str] = []
    seen: set = set()

    def _add(val: str) -> None:
        v = val.strip()
        if v and len(v) > 2 and v.lower() not in seen:
            seen.add(v.lower())
            names.append(v)

    for line in context.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        content = line[1:].strip()

        # ── Order lines: leading token is "Order #N" ─────────────────────
        # Extract product name and placed-by name from the full segment list
        if _re.match(r"Order #\d+", content):
            for seg in content.split(" | "):
                seg = seg.strip()
                if seg.startswith("product:"):
                    _add(seg.split(":", 1)[1].strip())
                elif seg.startswith("placed by:"):
                    _add(seg.split(":", 1)[1].strip())
            continue

        # ── Employee / project / generic lines ───────────────────────────
        # Take the leading token (before first " — " or " | ")
        leading = content
        for sep in [" — ", " | "]:
            if sep in content:
                leading = content[:content.index(sep)]
                break
        candidate = leading.strip()

        # Strip "key: " prefix from generic fallback format
        if ": " in candidate:
            candidate = candidate.split(": ", 1)[1].strip()

        _add(candidate)

    return names


def _extract_currency_amounts(context: str) -> List[str]:
    """
    Extract all currency strings like $400,000.00 from context.
    Returns them as plain substrings that should appear in the answer.
    """
    import re
    return re.findall(r'\$[\d,]+(?:\.\d+)?', context)


def _check_completeness(llm_answer: str, context: str, row_count: int) -> tuple[bool, str]:
    """
    Returns (is_complete, reason).
    Rejects the LLM answer when:
    - Any name/identifier extracted from context lines is absent (all row counts).
    - Any currency amount from context is absent when only 1 row.

    Applying the name check to single-row results as well catches cases like
    Q11 where the LLM returns just a person name instead of a full order answer,
    and Q18 where it hallucinates a wrong currency figure.
    """
    answer_lower = llm_answer.lower()

    # Currency check (single-row only — avoids false positives on salary lists)
    if row_count == 1:
        amounts = _extract_currency_amounts(context)
        for amt in amounts:
            digits = amt.replace("$", "").replace(",", "")
            if digits not in llm_answer.replace(",", ""):
                return False, f"currency {amt} missing from single-row answer"

    # Name/identifier check — applies to ALL row counts
    names = _extract_context_names(context)
    if not names:
        return True, ""

    missing = [n for n in names if n.lower() not in answer_lower]
    if missing:
        return False, f"names missing from answer: {missing}"

    return True, ""


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Template fallback — used only when LLM is unavailable or returns nothing.
# Extended with delta / having_count / temporal handlers.
# ---------------------------------------------------------------------------

def _format_single_row(row: Dict, user_question: str) -> Optional[str]:
    q = user_question.lower()

    # Delta result (single row with gap/max/min columns)
    if _is_delta_row(row):
        parts = []
        for k, v in row.items():
            if v is None:
                continue
            label = k.replace("_", " ")
            is_currency = any(
                word in k.lower()
                for word in ("salary", "price", "budget", "gap", "highest", "lowest", "pay")
            )
            parts.append(f"{label}: {_fmt_currency(v) if is_currency else v}")
        return " · ".join(parts) + "." if parts else None

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


def _template_fallback(user_question: str, db_result: DBResult, question_type: str = "") -> str:
    rows = db_result.rows
    if not rows:
        return "No matching data found."

    # ── Delta: single-row MAX/MIN/gap result ──────────────────────────────
    if question_type == "computed_delta" or (rows and _is_delta_row(rows[0])):
        row = rows[0]
        parts = []
        for k, v in row.items():
            if v is None:
                continue
            label = k.replace("_", " ")
            is_currency = any(
                word in k.lower()
                for word in ("salary", "price", "budget", "gap", "highest", "lowest", "pay")
            )
            parts.append(f"{label}: {_fmt_currency(v) if is_currency else v}")
        return " · ".join(parts) + "." if parts else "No delta data found."

    # ── Having count: employee + count list ───────────────────────────────
    if question_type == "having_count" or (rows and _is_having_count_row(rows[0])):
        lines = []
        for i, row in enumerate(rows, 1):
            name = _get(row, "employee_name", "name") or "Unknown"
            count_val   = None
            count_label = "items"
            for k, v in row.items():
                if k.endswith("_count") and v is not None:
                    count_val   = v
                    count_label = k.replace("_count", "s").replace("_", " ")
                    break
            role = _get(row, "employee_role", "role")
            dept = _get(row, "department_name")
            p = [name]
            if count_val is not None: p.append(f"({count_val} {count_label})")
            if role: p.append(f"— {role}")
            if dept: p.append(f"— {dept}")
            lines.append(f"{i}. " + " ".join(p))
        return f"Found {len(rows)} employees:\n" + "\n".join(lines)

    # ── Temporal: project or order rows with date context ─────────────────
    if question_type == "temporal_filter" or (rows and _is_temporal_row(rows[0])):
        first = rows[0]
        if "order_id" in first:
            lines = []
            for i, row in enumerate(rows, 1):
                oid  = _get(row, "order_id") or i
                prod = _get(row, "product_name")
                date = _get(row, "order_date")
                st   = _get(row, "status") or ""
                by   = _get(row, "employee_name")
                p = [f"#{oid}"]
                if prod: p.append(prod)
                if date: p.append(str(date))
                if st:   p.append(st)
                if by:   p.append(f"by {by}")
                lines.append(f"{i}. " + " · ".join(p))
            return f"Found {len(rows)} orders:\n" + "\n".join(lines)
        else:
            lines = []
            for i, row in enumerate(rows, 1):
                pname  = _get(row, "project_name") or f"Project {i}"
                start  = _get(row, "start_date")
                end    = _get(row, "end_date")
                status = _get(row, "project_status")
                p = [pname]
                if start:  p.append(f"started {start}")
                if end:    p.append(f"ends {end}")
                if status: p.append(status)
                lines.append(f"{i}. " + " · ".join(p))
            return f"Found {len(rows)} projects:\n" + "\n".join(lines)

    # ── Original generic fallback (unchanged) ─────────────────────────────
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
    question_type: str = "",
    filters: Optional[Dict[str, Any]] = None,
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

    lean_context = _build_context(db_result.rows, user_question, question_type)

    subject_header = _build_subject_header(filters or {})
    if subject_header:
        lean_context = subject_header + lean_context

    model_cfg = _load_model_config()
    llm_answer = _llm_synthesize(user_question, lean_context, db_result.row_count, model_cfg)

    if llm_answer:
        _DENIAL_PHRASES = (
            "no employees", "not found", "not mentioned", "no records",
            "no data", "none found", "no results", "not in the", "no one",
            "nobody", "no projects", "no orders", "not directly",
            "there are no", "none of the", "cannot find",
            "not provided", "amount not provided", "$x",
        )
        llm_lower = llm_answer.lower()

        denial_contradiction = (
            db_result.row_count > 0
            and any(p in llm_lower for p in _DENIAL_PHRASES)
        )

        numeric_contradiction = False
        if db_result.row_count > 1:
            for wrong_num in ("1 person", "1 employee", "1 record", "1 result", "only one"):
                if wrong_num in llm_lower:
                    numeric_contradiction = True
                    break

        # Completeness guard: verify all context names/amounts appear in answer.
        # Only runs on the lean_context (without subject header prefix) so the
        # subject name itself doesn't become a required match.
        completeness_ok, completeness_reason = _check_completeness(
            llm_answer,
            _build_context(db_result.rows, user_question, question_type),
            db_result.row_count,
        )

        if denial_contradiction or numeric_contradiction or not completeness_ok:
            if denial_contradiction:
                reason = "denial phrase"
            elif numeric_contradiction:
                reason = "numeric mismatch"
            else:
                reason = f"completeness ({completeness_reason})"
            logger.warning(
                "Synthesis rejected (%s): row_count=%d — using template fallback",
                reason, db_result.row_count,
            )
        else:
            return llm_answer

    logger.warning("LLM unavailable or contradicted — using template fallback")
    return _template_fallback(user_question, db_result, question_type)
