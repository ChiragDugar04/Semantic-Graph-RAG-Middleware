"""
middleware/entity_extractor.py

Replaces intent_classifier.py + param_extractor.py in the new pipeline.

Instead of classifying into a named intent, this module identifies:
  - Which database ENTITIES the question involves (Employee, Department, etc.)
  - What FILTERS to apply (name="Sarah", department="Engineering")
  - What PROJECTIONS the user wants (salary, role, budget, etc.)
  - What QUESTION TYPE it is (lookup / list / comparison / aggregation / cross_entity)

Two-tier approach (same philosophy as the old system):
  Tier 1 — Rule-based fast path (zero LLM cost, instant)
  Tier 2 — LLM extraction using qwen2.5:1.5b with structured JSON output

The output EntityExtractionResult feeds directly into:
  semantic_graph.py  → to find the JOIN path
  graph_query_builder.py → to construct the SQL
"""

from __future__ import annotations

import re
import time
import json
import requests
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional

from middleware.models import EntityExtractionResult


# ============================================================
# CONFIG LOADER
# ============================================================

def _load_model_config() -> dict:
    """Load model settings from intents.yaml (reused from old system)."""
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


# ============================================================
# NORMALIZATION (same as old intent_classifier)
# ============================================================

def _normalize(text: str) -> str:
    """Lowercase and normalize Unicode punctuation."""
    return (
        text
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2014", "-").replace("\u2013", "-")
        .lower().strip()
    )


# ============================================================
# KNOWN VOCABULARY — for rule-based tier
# ============================================================

# Exact employee names from seed data (lowercase)
_KNOWN_EMPLOYEES = [
    "sarah connor", "john reese", "alan turing", "grace hopper",
    "linus torvalds", "michael scott", "jim halpert", "dwight schrute",
    "pam beesly", "toby flenderson", "kelly kapoor", "don draper",
    "peggy olson", "roger sterling", "leslie knope", "ben wyatt",
    "tom haverford",
]

# Exact department names from seed data (lowercase)
_KNOWN_DEPARTMENTS = [
    "engineering", "sales", "human resources", "hr", "marketing", "operations",
]

# Exact project names from seed data (lowercase)
_KNOWN_PROJECTS = [
    "api gateway rebuild", "platform migration", "q1 marketing campaign",
    "hr system upgrade", "ops automation initiative", "sales crm integration",
]

# Known product names (lowercase)
_KNOWN_PRODUCTS = [
    "laptop pro 15", "wireless mouse", "standing desk", "ergonomic chair",
    "usb-c hub", "mechanical keyboard", "monitor 27 inch", "whiteboard 4x6",
    "notebook pack", "webcam hd", "desk lamp", "printer paper a4",
]

# Known product categories
_KNOWN_CATEGORIES = [
    "electronics", "furniture", "office supply", "office supplies",
]

# Keywords that signal each question type
_COMPARISON_KEYWORDS = [
    "highest", "lowest", "most", "least", "best", "worst", "top",
    "maximum", "minimum", "richest", "rank", "ranked", "ranking",
    "who earns", "who makes", "who gets paid", "highest paid",
    "lowest paid", "most expensive", "cheapest",
]

_AGGREGATION_KEYWORDS = [
    "how many", "count", "total", "average", "avg", "sum",
    "number of", "how much total", "overall",
]

_LIST_KEYWORDS = [
    "list", "show", "all", "every", "which employees", "who works",
    "who is in", "members of", "give me all", "show me all",
    "what are all",
]

_ORDER_KEYWORDS = [
    "order", "orders", "purchase", "purchases", "bought", "ordered",
]

_ORDER_STATUS_SYNONYMS = {
    "not completed": "pending", "not complete": "pending",
    "in progress": "processing", "in-progress": "processing",
    "in transit": "shipped", "in-transit": "shipped",
    "on the way": "shipped", "out for delivery": "shipped",
    "completed": "delivered", "done": "delivered",
    "canceled": "cancelled", "called off": "cancelled",
}

_KNOWN_ORDER_STATUSES = [
    "pending", "processing", "shipped", "delivered", "cancelled",
]


# ============================================================
# TIER 1 — RULE-BASED EXTRACTION
# ============================================================

def _rule_based_extraction(question: str) -> Optional[EntityExtractionResult]:
    """Fast-path extraction using deterministic string matching.

    Handles single-entity questions and common patterns without
    calling the LLM. Returns None if rules don't confidently match,
    triggering Tier 2 LLM extraction.

    Args:
        question: Raw user question.

    Returns:
        EntityExtractionResult if rules matched, else None.
    """
    q = _normalize(question)
    entities: List[str] = []
    filters: Dict[str, Any] = {}
    projections: List[str] = []
    question_type = "lookup"

    # ── FIX 1: Detect "employee/employees" word FIRST ────────
    # Before any entity matching, check if the question is talking
    # about employees generically (e.g. "which employees in Engineering")
    # so Employee is always added even without a specific name.
    _EMPLOYEE_SUBJECT_PHRASES = [
        "employees", "employee", "who works", "who is in", "who are in",
        "which employees", "staff", "team members", "people in",
        "workers", "assigned to", "working on", "members of",
    ]
    employee_subject_detected = any(phrase in q for phrase in _EMPLOYEE_SUBJECT_PHRASES)
    if employee_subject_detected:
        entities.append("Employee")

    # ── Detect "managed by X" BEFORE employee name ───────────
    # For queries like "projects managed by Sarah" or "employees
    # working on projects managed by Don", the named person is the
    # PROJECT MANAGER, not an employee being listed. Store separately
    # so the query builder can build a subquery instead of filtering
    # the listed employees by name.
    import re as _re
    _managed_by_match = _re.search(
        r"managed by\s+([a-z]+(?:\s+[a-z]+)?)", q
    )
    matched_manager: Optional[str] = None
    if _managed_by_match:
        candidate = _managed_by_match.group(1).strip()
        # Confirm it's a known employee name
        for emp in sorted(_KNOWN_EMPLOYEES, key=len, reverse=True):
            if emp.startswith(candidate) or candidate in emp:
                matched_manager = emp
                filters["manager_name"] = emp.title()
                if "Project" not in entities:
                    entities.append("Project")
                break

    # ── Detect employee name ──────────────────────────────────
    matched_employee: Optional[str] = None
    for emp in sorted(_KNOWN_EMPLOYEES, key=len, reverse=True):
        if emp in q:
            # Skip — this name is already captured as manager_name
            if matched_manager and emp == matched_manager:
                continue
            matched_employee = emp
            if "Employee" not in entities:
                entities.append("Employee")
            # Reconstruct title-cased name
            filters["employee_name"] = emp.title()
            break

    # ── Detect department name ────────────────────────────────
    matched_dept: Optional[str] = None
    for dept in sorted(_KNOWN_DEPARTMENTS, key=len, reverse=True):
        if dept in q:
            matched_dept = dept
            if "Department" not in entities:
                entities.append("Department")
            # Normalize HR
            dept_value = "Human Resources" if dept == "hr" else dept.title()
            filters["department_name"] = dept_value
            break

    # ── Detect project name ───────────────────────────────────
    for proj in sorted(_KNOWN_PROJECTS, key=len, reverse=True):
        if proj in q:
            if "Project" not in entities:
                entities.append("Project")
            filters["project_name"] = proj.title()
            break

    # ── FIX 2: Detect "project/projects" word generically ────
    # "working on projects", "assigned to projects" etc.
    _PROJECT_SUBJECT_PHRASES = ["projects", "project managed", "project led"]
    if any(phrase in q for phrase in _PROJECT_SUBJECT_PHRASES):
        if "Project" not in entities:
            entities.append("Project")

    # ── Detect product name ───────────────────────────────────
    for prod in sorted(_KNOWN_PRODUCTS, key=len, reverse=True):
        if prod in q:
            if "Product" not in entities:
                entities.append("Product")
            filters["product_name"] = prod.title()
            break

    # ── Detect product category ───────────────────────────────
    for cat in _KNOWN_CATEGORIES:
        if cat in q:
            if "Product" not in entities:
                entities.append("Product")
            # Normalize "office supplies" → "Office Supply"
            cat_value = "Office Supply" if "office suppl" in cat else cat.title()
            filters["category"] = cat_value
            break

    # ── FIX 3: Detect "product" word generically ─────────────
    # "most expensive product", "which product" etc.
    if "product" in q or "products" in q or "item" in q:
        if "Product" not in entities:
            entities.append("Product")

    # ── Detect order status ───────────────────────────────────
    order_status: Optional[str] = None
    for synonym, canonical in sorted(
        _ORDER_STATUS_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if synonym in q:
            order_status = canonical
            break
    if order_status is None:
        for status in _KNOWN_ORDER_STATUSES:
            if status in q:
                order_status = status
                break

    if order_status or any(kw in q for kw in _ORDER_KEYWORDS):
        if "Order" not in entities:
            entities.append("Order")
        if order_status:
            filters["order_status"] = order_status


    # ── Detect project status ────────────────────────────────
    _PROJECT_STATUS_MAP = {
        "in progress": "active", "ongoing": "active", "current": "active",
        "active": "active", "completed": "completed", "finished": "completed",
        "planned": "planned", "upcoming": "planned",
    }
    for _pkw, _pval in sorted(_PROJECT_STATUS_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if _pkw in q and "Project" in entities:
            filters["project_status"] = _pval
            break

    # ── Detect question type ──────────────────────────────────
    # FIX 4: Check aggregation first, then comparison (longest phrases first)
    for kw in _AGGREGATION_KEYWORDS:
        if kw in q:
            question_type = "aggregation"
            break

    if question_type == "lookup":
        # Check comparison keywords sorted longest-first to avoid
        # "most" matching before "most expensive"
        for kw in sorted(_COMPARISON_KEYWORDS, key=len, reverse=True):
            if kw in q:
                question_type = "comparison"
                break

    if question_type == "lookup":
        for kw in _LIST_KEYWORDS:
            if kw in q:
                question_type = "list"
                break

    # ── Detect projections from question ─────────────────────
    if "salary" in q or "earn" in q or "pay" in q or "compensation" in q:
        projections.append("salary")
    if "budget" in q:
        projections.append("budget")
    if "role" in q or "position" in q or "title" in q:
        projections.append("role")
    if "stock" in q or "inventory" in q or "units" in q or "quantity" in q:
        projections.append("stock_quantity")
    if "price" in q or "cost" in q or "expensive" in q:
        projections.append("price")

    # ── FIX 5: Cross-entity detection + entity promotion ─────
    entity_set = set(entities)

    # CRITICAL: If question is a comparison/list involving a Department,
    # the user almost always wants EMPLOYEE data within that department
    # (e.g. "who earns the most in Marketing?" → needs Employee + Department).
    # Promote Employee into entities so the JOIN is built correctly.
    if "Department" in entity_set and "Employee" not in entity_set:
        if question_type in ("comparison", "list", "aggregation"):
            # Salary/role comparisons within a dept require Employee rows
            if any(kw in q for kw in ["earn", "salary", "paid", "pay",
                                       "earns", "makes", "compensation",
                                       "who", "list", "show", "employees"]):
                entities.append("Employee")
                entity_set.add("Employee")

    # Employee + Department → cross_entity when both specifically named
    if "Employee" in entity_set and "Department" in entity_set:
        if matched_employee and matched_dept:
            # Both a specific person AND a dept → truly cross-entity
            question_type = "cross_entity"
        # Otherwise keep list/comparison — it's "employees IN dept" pattern

    # Employee + Project → always cross_entity
    if "Employee" in entity_set and "Project" in entity_set:
        question_type = "cross_entity"

    # Department + Project → cross_entity
    if "Department" in entity_set and "Project" in entity_set:
        question_type = "cross_entity"

    # 3+ entities → always cross_entity
    if len(entity_set) >= 3:
        question_type = "cross_entity"

    # ── Decide: return rules result or fall through to LLM ───
    if not entities:
        # No entities detected by rules — must use LLM
        return None

    # Rules confidently matched at least one entity
    return EntityExtractionResult(
        entities=entities,
        filters=filters,
        projections=projections,
        question_type=question_type,
        extraction_method="rules",
        latency_ms=0.0,
    )


# ============================================================
# TIER 2 — LLM EXTRACTION
# ============================================================

def _llm_extraction(question: str) -> EntityExtractionResult:
    """Use qwen2.5:1.5b to extract entities, filters, and question type.

    Sends a carefully structured prompt that instructs the LLM to
    return ONLY a JSON object with no prose. Parses the response
    and falls back to a safe default on any error.

    Args:
        question: Raw user question.

    Returns:
        EntityExtractionResult populated from LLM output.
    """
    model_cfg = _load_model_config()
    fast_model: str    = model_cfg["fast_model"]
    base_url: str      = model_cfg["ollama_base_url"]
    temperature: float = model_cfg["intent_temperature"]

    prompt = f"""You are a database entity extractor. Analyze the user question and return a JSON object.

AVAILABLE ENTITIES:
- Employee   (table: employees)   fields: name, role, salary, hire_date, email
- Department (table: departments) fields: name, budget, location
- Product    (table: products)    fields: name, category, price, stock_quantity, supplier
- Order      (table: orders)      fields: id, quantity, order_date, status
- Project    (table: projects)    fields: name, description, budget, status, start_date

QUESTION TYPES:
- lookup       : find details about one specific thing (e.g. "What is Sarah's salary?")
- list         : list multiple items (e.g. "List all employees in Engineering")
- comparison   : find highest/lowest/best (e.g. "Who earns the most?")
- aggregation  : count/sum/average (e.g. "How many employees are in each department?")
- cross_entity : involves 2+ entity types in one question (e.g. "Which Engineering employees work on projects managed by Sarah?")
- other        : cannot be answered from the database

FILTER KEY NAMES to use:
  For Employee:   employee_name, employee_role
  For Department: department_name
  For Product:    product_name, category
  For Order:      order_status
  For Project:    project_name, project_status

INSTRUCTIONS:
1. Return ONLY a JSON object. No explanation, no markdown, no code fences.
2. "entities" must be a list of entity names from the list above.
3. "filters" must be a dict of filter_key: extracted_value pairs.
4. "projections" must be a list of column names the user wants to see.
5. "question_type" must be exactly one of the types above.

EXAMPLE INPUT:  "Which employees in Engineering are working on projects managed by Sarah?"
EXAMPLE OUTPUT: {{"entities": ["Employee", "Department", "Project"], "filters": {{"department_name": "Engineering", "employee_name": "Sarah"}}, "projections": ["employee_name", "project_name", "assignment_role"], "question_type": "cross_entity"}}

User question: "{question}"

JSON:"""

    start_time = time.time()

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": fast_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 300,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        raw: str = response.json().get("response", "").strip()

    except requests.RequestException as e:
        latency_ms = (time.time() - start_time) * 1000
        return EntityExtractionResult(
            entities=[],
            filters={},
            projections=[],
            question_type="other",
            extraction_method="llm_error",
            latency_ms=round(latency_ms, 2),
        )

    latency_ms = (time.time() - start_time) * 1000

    # ── Parse JSON from LLM response ─────────────────────────
    parsed = _parse_llm_json(raw)

    valid_entities = {"Employee", "Department", "Product", "Order", "Project"}
    valid_types    = {"lookup", "list", "comparison", "aggregation", "cross_entity", "other"}

    entities    = [e for e in parsed.get("entities", []) if e in valid_entities]
    filters     = parsed.get("filters", {})
    projections = parsed.get("projections", [])
    q_type      = parsed.get("question_type", "lookup")

    if q_type not in valid_types:
        q_type = "lookup"

    # Safety: if 2+ entities detected and not already cross_entity, upgrade
    if len(set(entities)) >= 2 and q_type == "lookup":
        q_type = "cross_entity"

    return EntityExtractionResult(
        entities=entities,
        filters={k: str(v) for k, v in filters.items() if v},
        projections=projections,
        question_type=q_type,
        extraction_method="llm",
        latency_ms=round(latency_ms, 2),
    )


def _parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output.

    Strips markdown fences, finds the first { } block, and
    attempts json.loads. Returns empty dict on any failure.

    Args:
        raw: Raw string from LLM response.

    Returns:
        dict: Parsed JSON or {} on failure.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Find the JSON object
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1

    if start == -1 or end == 0:
        return {}

    json_str = cleaned[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common LLM JSON mistakes
        try:
            # Replace single quotes with double quotes
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {}


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def extract_entities(question: str) -> EntityExtractionResult:
    """Extract entities, filters, and question type from a user question.

    Runs Tier 1 (rules) first. If rules confidently identify at
    least one entity, returns immediately. Otherwise calls Tier 2
    (LLM) for complex or unrecognized questions.

    Args:
        question: Raw user question as typed.

    Returns:
        EntityExtractionResult with all fields populated.

    Example:
        >>> result = extract_entities("What is Sarah Connor's salary?")
        >>> result.entities
        ['Employee']
        >>> result.filters
        {'employee_name': 'Sarah Connor'}
        >>> result.question_type
        'lookup'
    """
    start_time = time.time()

    # ── Tier 1: Rules ─────────────────────────────────────────
    rule_result = _rule_based_extraction(question)

    if rule_result is not None:
        rule_result.latency_ms = round((time.time() - start_time) * 1000, 2)
        return rule_result

    # ── Tier 2: LLM ───────────────────────────────────────────
    llm_result = _llm_extraction(question)
    llm_result.latency_ms = round((time.time() - start_time) * 1000, 2)
    return llm_result
