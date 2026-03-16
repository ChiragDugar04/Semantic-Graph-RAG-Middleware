from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import spacy
import yaml

from middleware.models import EntityExtractionResult, ExtractionConfidence

logger = logging.getLogger(__name__)

_nlp: Optional[spacy.language.Language] = None
_schema_cache: Optional[Dict[str, Any]] = None


def _load_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model en_core_web_sm loaded")
        except OSError:
            logger.warning("spaCy model not found — running without dependency parsing")
            _nlp = None
    return _nlp


def _load_model_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


def _load_schema() -> Dict[str, Any]:
    global _schema_cache
    if _schema_cache is None:
        schema_path = Path(__file__).parent.parent / "config" / "graph_schema.yaml"
        with open(schema_path, "r") as f:
            _schema_cache = yaml.safe_load(f)
    return _schema_cache


def _get_valid_entities() -> List[str]:
    return list(_load_schema()["nodes"].keys())


def _get_schema_keywords() -> Dict[str, Any]:
    schema = _load_schema()
    keywords: Dict[str, Any] = {
        "entity_triggers": {},
        "filter_column_labels": {},
        "projection_labels": {},
    }

    # T1-A: trigger_words are now read from graph_schema.yaml node definitions.
    # To add synonyms for an entity, edit the trigger_words list in the schema —
    # no Python change required. New entities added to the schema are automatically
    # detectable here as long as they carry a trigger_words list.
    # Fallback: if a node has no trigger_words (e.g. a minimally-defined node),
    # use [entity.lower()] so detection still fires on the exact entity name.
    for entity, node_data in schema["nodes"].items():
        triggers = node_data.get("trigger_words", [entity.lower()])
        keywords["entity_triggers"][entity] = triggers

        for col_def in node_data.get("selectable_columns", []):
            label = col_def.get("label", "").lower()
            alias = col_def.get("alias", "")
            col = col_def.get("column", "")
            if label:
                keywords["projection_labels"][label] = alias
            if alias:
                keywords["projection_labels"][alias] = alias
            if col:
                keywords["projection_labels"][col] = alias

    return keywords


_QUESTION_TYPE_PATTERNS = [
    (re.compile(r"\b(how many|count of|number of|total number)\b", re.I), "aggregation"),
    (re.compile(r"\b(average|avg|mean|sum|total)\b", re.I), "aggregation"),
    (re.compile(r"\b(highest|lowest|most|least|top|best|worst|maximum|minimum|richest|cheapest|priciest|most expensive)\b", re.I), "comparison"),
    (re.compile(r"\b(who earns|who makes|who gets paid|highest paid|lowest paid|best paid)\b", re.I), "comparison"),
    (re.compile(r"\b(list|show|all|every|display|give me all)\b", re.I), "list"),
    (re.compile(r"\b(what is|what are|tell me|find|get|show me the|which is|who is)\b", re.I), "lookup"),
]

_OFF_TOPIC_PATTERNS = re.compile(
    r"\b(weather|joke|news|sports|stock market|recipe|movie|song|capital of|"
    r"population of|how to cook|translate|what time|current time|define |meaning of)\b",
    re.I,
)

_MANAGER_PATTERNS = re.compile(
    r"\b(managed by|manages|manager of|who manages|who leads|who runs|led by|run by|head of|in charge of"
    r"|supervised by|supervises|overseen by)\b",
    re.I,
)

# N4: "reports to / reporting to" signals an org-chart hierarchy query.
# This is intentionally SEPARATE from _MANAGER_PATTERNS because it routes
# to a different path (Employee→Department via department manager FK)
# rather than the project-manager path that manager_name triggers.
# T1.2: Added "under" with a Title-Case lookahead so "everyone under Toby Flenderson"
# maps to reports_to_name. The (?=[A-Z]) lookahead prevents "under contract",
# "under $100", "under review" from firing — only Title-Case proper nouns match.
_REPORTS_TO_PATTERNS = re.compile(
    r"\b(reports to|reporting to|who does .{1,40} report to|direct reports of"
    r"|under\s+(?=[A-Z]))\b",
    re.I,
)

_ASSIGNMENT_PATTERNS = re.compile(
    r"\b(assigned to|working on|works on|on the|member of|part of|involved in|participating in)\b",
    re.I,
)

# T1-C: _STATUS_MAP removed. Status values are now read from
# Order.filterable_columns.status.enum_values and
# Project.filterable_columns.status.enum_values in graph_schema.yaml.
# To add a new status value, edit the schema — zero Python changes required.

_PROJECTION_KEYWORDS = {
    "salary": "salary",
    "salaries": "salary",
    "pay": "salary",
    "earnings": "salary",
    "compensation": "salary",
    "earn": "salary",
    "makes": "salary",
    "budget": "budget",
    "stock": "stock_quantity",
    "inventory": "stock_quantity",
    "units": "stock_quantity",
    "price": "price",
    "cost": "price",
    "location": "location",
    "email": "email",
    "role": "role",
    "hire date": "hire_date",
    "hired": "hire_date",
    "headcount": "headcount",
    "manager": "manager_name",
    "manages": "manager_name",
    "managed by": "manager_name",
    "status": "status",
    "description": "description",
}


def _compute_confidence(
    entities: List[str],
    filters: Dict[str, Any],
    question_type: str,
    has_clear_structure: bool,
    projections: List[str],
    extraction_layer: str,
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    entity_score = 0.3 if entities else 0.0
    breakdown["entity_recognized"] = entity_score

    filter_score = 0.2 if filters else 0.0
    breakdown["filter_extracted"] = filter_score

    type_score = 0.2 if question_type != "lookup" else 0.1
    breakdown["question_type_resolved"] = type_score

    structure_score = 0.15 if has_clear_structure else 0.0
    breakdown["grammatical_structure_clear"] = structure_score

    projection_score = 0.15 if projections else 0.0
    breakdown["projections_identified"] = projection_score

    if extraction_layer == "pattern":
        breakdown["layer_bonus"] = 0.1
    elif extraction_layer == "spacy":
        breakdown["layer_bonus"] = 0.05
    else:
        breakdown["layer_bonus"] = 0.0

    total = min(1.0, sum(breakdown.values()))
    return total, breakdown


# ---------------------------------------------------------------------------
# Noise words that should never be treated as entity name values
# ---------------------------------------------------------------------------
_NAME_NOISE = frozenset({
    "all", "every", "any", "each", "the", "a", "an",
    "list", "show", "display", "find", "get", "give",
    "what", "who", "which", "how", "where", "when", "why",
    "me", "us", "i", "my", "our", "their",
    "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "for", "by", "to", "from",
    "and", "or", "but", "not", "no", "with", "about",
    "assigned", "working", "works", "managed", "manages",
    "currently", "recently", "active", "latest",
})

# Entity-type keywords that signal what the following / preceding
# proper noun refers to.  Order matters: more-specific first.
_ENTITY_CONTEXT_SIGNALS: List[Tuple[str, str]] = [
    # (trigger word in question lowercase, filter_key to assign)
    ("project",     "project_name"),
    ("initiative",  "project_name"),
    ("program",     "project_name"),
    ("product",     "product_name"),
    ("item",        "product_name"),
    ("order",       "product_name"),   # D4: "orders for [Standing Desk]"
    ("department",  "department_name"),
    ("dept",        "department_name"),
    ("team",        "department_name"),
    ("division",    "department_name"),
]

# Directional preposition signals (D4/D8): only apply when the trigger
# appears in the text BEFORE (to the left of) the candidate.
# This prevents "in [Operations] ... project" from mapping Operations to
# project_name — "in" fires as a department signal from the left side.
_BEFORE_CANDIDATE_SIGNALS: List[Tuple[str, str]] = [
    ("in",   "department_name"),   # "employees in [Sales]"
    ("for",  "product_name"),      # "orders for [Standing Desk]"
    ("of",   "product_name"),      # "units of [Webcam HD]"
]


def _scan_proper_noun_candidates(question: str) -> List[str]:
    """
    Extract all Title-Case word sequences from the original question.

    Works on the raw question (not lowercased) so capitalisation is
    preserved.  Single-word stop-words are filtered out.
    Returns longest-match candidates first.
    """
    # Match 1-to-4 consecutive Title-Case tokens (e.g. "Platform Migration",
    # "Webcam HD", "Leslie Knope", "API Gateway Rebuild")
    raw_candidates = re.findall(
        r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3})\b",
        question,
    )

    # Question-opener and auxiliary words that are Title-Case at sentence
    # start but must never become entity name candidates (D3).
    _QUESTION_STARTERS = frozenset({
        "who", "which", "what", "how", "where", "when", "why",
        "are", "is", "do", "does", "did", "have", "has",
    })

    seen: set = set()
    result: List[str] = []
    for cand in raw_candidates:
        cand = cand.strip()
        if cand.lower() in _NAME_NOISE:
            continue
        # must be at least 2 chars and contain a real letter
        if len(cand) < 2 or not re.search(r"[a-zA-Z]", cand):
            continue
        # D3: reject multi-word spans where any individual word is noise,
        # or the span opens with a question/auxiliary verb word.
        # Single-word candidates are already filtered by _NAME_NOISE above.
        words = cand.split()
        if len(words) > 1:
            if any(w.lower() in _NAME_NOISE for w in words):
                continue
            if words[0].lower() in _QUESTION_STARTERS:
                continue
        if cand not in seen:
            seen.add(cand)
            result.append(cand)

    # Sort longest first so "Platform Migration" beats "Platform"
    result.sort(key=len, reverse=True)
    return result


def _extract_named_values(question: str) -> Dict[str, Any]:
    """
    Contextual Proper Noun Extractor — schema-aware, case-insensitive.

    Extraction pipeline
    -------------------
    Phase 0 — Status / category tokens (purely lexical, order-independent)
    Phase 1 — Manager-first: detect relational keywords and lock the
               following name as `manager_name`.  This name is then
               EXCLUDED from all subsequent general extraction to prevent
               manager/department collision.
    Phase 2 — Proper noun scan: find every Title-Case sequence in the
               original question and assign it the most appropriate filter
               key by checking what entity-context signals surround it.
    Phase 3 — Person possessive / "what is X's" fallback for employee names.
    """
    filters: Dict[str, Any] = {}
    q_lower = question.lower()

    # ------------------------------------------------------------------ #
    # Phase 0 — Status tokens (order-independent, purely lowercase scan)  #
    # ------------------------------------------------------------------ #
    # T1-C: Status values are read from Order and Project filterable_columns
    # enum_values in graph_schema.yaml. The context-routing logic (which
    # surrounding keyword decides order_status vs project_status) stays here
    # in Python — only the status vocabulary moved to the schema.
    # To add a new status value: add an enum_values entry in the schema.
    _schema_nodes = _load_schema()["nodes"]
    _order_enum   = _schema_nodes.get("Order",   {}).get("filterable_columns", {}).get("status", {}).get("enum_values", [])
    _project_enum = _schema_nodes.get("Project", {}).get("filterable_columns", {}).get("status", {}).get("enum_values", [])

    # Build a unified synonym→canonical lookup covering both entities.
    # We iterate both lists because "active" and "completed" appear in both
    # Order and Project ENUMs — the context-routing if/elif below decides
    # which filter key to write. The synonym map just normalises the spelling.
    _status_synonym_map: Dict[str, str] = {}
    for _entry in _order_enum + _project_enum:
        _canonical = _entry.get("canonical", "")
        for _syn in _entry.get("synonyms", []):
            _status_synonym_map[str(_syn).lower()] = _canonical

    for status_word, status_val in _status_synonym_map.items():
        if re.search(rf"\b{re.escape(status_word)}\b", q_lower):
            if any(w in q_lower for w in ["order", "orders", "purchase"]):
                filters["order_status"] = status_val
            elif any(w in q_lower for w in ["project", "projects", "initiative"]):
                filters["project_status"] = status_val
            # Don't break — a question can have both (rare but safe)

    # T1-B: Category tokens — built dynamically from Product.filterable_columns.category.valid_values
    # in graph_schema.yaml. To add a new category (e.g. "Clothing"), add a valid_values entry
    # to the schema with canonical DB value + synonyms. Zero Python changes required.
    # Filter key "product_category" is schema-derived: entity_lower + "_" + col_key = product_category.
    # Uses exact match in WHERE (match_type: exact), so canonical must match the DB value precisely.
    _cat_col = _load_schema()["nodes"].get("Product", {}).get("filterable_columns", {}).get("category", {})
    for _cat_entry in _cat_col.get("valid_values", []):
        _canonical = _cat_entry.get("canonical", "")
        for _synonym in _cat_entry.get("synonyms", []):
            if re.search(rf"\b{re.escape(_synonym)}\b", q_lower):
                filters["product_category"] = _canonical
                break
        if "product_category" in filters:
            break

    # R5: Supplier filter extraction.
    # "supplied by TechSupply Co" / "from supplier OfficeWorld"
    # Product.supplier is now a filterable_column → filter key "product_supplier".
    # T1.5: Tightened to require a prepositional left-boundary (supplied by / from supplier /
    # by supplier / supplier is|:). Bare "supplier" as a sentence subject (e.g. "which
    # supplier provides…") no longer triggers the pattern, preventing the predicate of the
    # sentence from being captured as the supplier value.
    # Additional guard: captured value must not start with a lowercase verb word
    # (provides, gives, sells, makes, is, has, etc.) — these are question predicates,
    # not supplier names.
    _SUPPLIER_VERB_STARTERS = frozenset({
        "provides", "provide", "gives", "give", "sells", "sell",
        "makes", "make", "offers", "offer", "has", "have", "is", "are",
        "supplies", "supply", "produces", "produce",
    })
    supplier_match = re.search(
        r"\b(?:supplied by|from supplier|by supplier|supplier(?:\s+is|\s*:))"
        r"\s+([A-Za-z][A-Za-z0-9\s]{1,30}?)(?:\s*$|\s*[,\?.])",
        question,  # use original case to preserve proper noun capitalisation
    )
    if not supplier_match:
        # Fallback: "made by / manufactured by [Supplier]"
        supplier_match = re.search(
            r"\b(?:made by|manufactured by)\s+"
            r"([A-Za-z][A-Za-z0-9\s]{1,30}?)(?:\s*$|\s*[,\?.])",
            question,
        )
    if supplier_match:
        sup_val = supplier_match.group(1).strip()
        first_word = sup_val.split()[0].lower() if sup_val else ""
        if (sup_val.lower() not in _NAME_NOISE
                and len(sup_val) > 1
                and first_word not in _SUPPLIER_VERB_STARTERS):
            filters["product_supplier"] = sup_val
            logger.debug("Phase 0 supplier extraction: product_supplier=%s", sup_val)

    # R6: Department location extraction.
    # "Building A", "Building 3", "Floor 2" etc. — the schema's Department.location
    # filterable_column generates filter key "department_location" in _build_filter_map.
    # Single-letter/digit identifiers (e.g. "A") are stripped by _scan_proper_noun_candidates
    # (min 2 chars), so we capture them explicitly here in Phase 0 before any noun scan.
    # Run on ORIGINAL question (not q_lower) so "Building A" (uppercase A) is matched.
    loc_match = re.search(
        r"\b(Building\s+[A-Z0-9]\w*|Floor\s+\d+|Site\s+[A-Z0-9]\w*|Campus\s+[A-Z0-9]\w*)",
        question,
    )
    if not loc_match:
        # Fallback: "located in <phrase>"
        loc_match = re.search(
            r"\blocated\s+in\s+([A-Za-z0-9][A-Za-z0-9\s]{1,20}?)(?:\s*$|\s*[,\?.])",
            question,
        )
    if loc_match:
        raw_loc = loc_match.group(1).strip() if loc_match.lastindex and loc_match.lastindex >= 1 else loc_match.group(0).strip()
        raw_loc = re.sub(r"^located\s+in\s+", "", raw_loc, flags=re.I).strip()
        cap_loc = " ".join(w.capitalize() for w in raw_loc.split())
        if cap_loc and cap_loc.lower() not in _NAME_NOISE:
            filters["department_location"] = cap_loc
            logger.debug("Phase 0 location extraction: department_location=%s", cap_loc)

    # ------------------------------------------------------------------ #
    # Phase 0.5 — Subject-first person lock                             #
    # Handles two patterns:                                             #
    #   (a) "[Name] is in which project" — name at sentence START       #
    #   (b) "in which project is Sarah Connor" — name at sentence END   #
    # Without this lock Phase 2 sees "project" in context around the   #
    # name and maps it to project_name, causing 0-row queries.         #
    # ------------------------------------------------------------------ #
    # N1 FIX: These patterns must NOT use re.I.
    # Python's re.I makes [A-Z] match lowercase letters, so "which employees"
    # would match [A-Z][a-z]+ and "managed by Alan Turing" would match as a name.
    # Person names are always Title-Case proper nouns — no case-insensitive flag needed.
    _SUBJECT_RELATIONAL_START = re.compile(
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+"
        r"(?:is|was|are|were)\s+"
        r"(?:in|on|assigned|part of|working|involved|a member)",
        # NO re.I — [A-Z] must match uppercase only
    )
    # Pattern (b): question ends with a name after a relational question word
    # "in which [entity] is [Name]?" or "which [entity] is [Name] in?"
    _SUBJECT_RELATIONAL_END = re.compile(
        r"\b(?:is|are|was)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*\??\s*$",
        # NO re.I — [A-Z] must match uppercase only
    )
    # T1.1: Pattern (c): mid-sentence subject relational — "[Name] belong(s) to",
    # "[Name] is on [entity]". Fires when a Title-Case name precedes a relational
    # verb that signals the name is the subject (the person), not the destination.
    # Negative lookahead (?![A-Z]) after the relational phrase prevents "Sarah Connor
    # is on Platform Migration" (a Title-Case noun follows "on") from misfiring.
    # NO re.I — [A-Z] must enforce uppercase only.
    _SUBJECT_RELATIONAL_MID = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
        r"(?:belongs? to|is on(?!\s+[A-Z])|was on(?!\s+[A-Z])"
        r"|is a member of|works? (?:in|on|for)|sit(?:s|ting) (?:in|on))",
        # NO re.I — [A-Z] must match uppercase only
    )

    subj_match = _SUBJECT_RELATIONAL_START.match(question.strip())
    if not subj_match:
        subj_match = _SUBJECT_RELATIONAL_END.search(question.strip())
    if not subj_match:
        subj_match = _SUBJECT_RELATIONAL_MID.search(question.strip())

    if subj_match:
        subj_name = subj_match.group(1).strip()
        if subj_name.lower() not in _NAME_NOISE and subj_name not in _get_valid_entities():
            filters["employee_name"] = subj_name
            locked_names_early = {subj_name.lower()}
            logger.debug("Phase 0.5 subject lock: employee_name=%s", subj_name)
        else:
            locked_names_early = set()
    else:
        locked_names_early = set()

    # ------------------------------------------------------------------ #
    # Phase 1 — Manager-first relational lock                             #
    # ------------------------------------------------------------------ #
    locked_names: set = locked_names_early   # inherit early locks

    manager_match = _MANAGER_PATTERNS.search(question)
    if manager_match:
        after_manager = question[manager_match.end():].strip()
        # Allow up to 4-word proper noun after the relational keyword
        name_match = re.search(
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", after_manager
        )
        if name_match:
            mgr_name = name_match.group(1).strip()
            if mgr_name.lower() not in _NAME_NOISE:
                filters["manager_name"] = mgr_name
                locked_names.add(mgr_name.lower())
                logger.debug("Manager-first lock: manager_name=%s", mgr_name)

    # N4: Org-chart hierarchy lock — "reports to [Name]" / "reporting to [Name]"
    # This is distinct from manager_name (project manager). It means:
    # "find employees whose department is managed by [Name]".
    # Stores the result in reports_to_name which pipeline and builder handle separately.
    reports_to_match = _REPORTS_TO_PATTERNS.search(question)
    if reports_to_match:
        after_rt = question[reports_to_match.end():].strip()
        rt_name_match = re.search(
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", after_rt
        )
        if rt_name_match:
            rt_name = rt_name_match.group(1).strip()
            if rt_name.lower() not in _NAME_NOISE:
                filters["reports_to_name"] = rt_name
                locked_names.add(rt_name.lower())
                logger.debug("Reports-to lock: reports_to_name=%s", rt_name)

    # T1.3: Possessive-team lock — "[Name]'s team/group/department/division"
    # "Leslie Knope's team" → reports_to_name=Leslie Knope.
    # This is a Phase 1 relational lock: the possessive signals the name is a
    # manager whose direct reports we want, NOT a department name value.
    # Guard: candidate must be at least 2 Title-Case words (prevents "Engineering's team"
    # from firing — "Engineering" is a single word that doesn't end with [a-z]+\s+[A-Z]).
    # NO re.I — [A-Z] must enforce uppercase only.
    _POSSESSIVE_TEAM = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'s\s+"
        r"(?:team|group|department|division|dept|people|staff|reports|direct reports)",
        # NO re.I
    )
    poss_match = _POSSESSIVE_TEAM.search(question)
    if poss_match and "reports_to_name" not in filters:
        poss_name = poss_match.group(1).strip()
        if poss_name.lower() not in _NAME_NOISE and poss_name not in _get_valid_entities():
            filters["reports_to_name"] = poss_name
            locked_names.add(poss_name.lower())
            logger.debug("Possessive-team lock: reports_to_name=%s", poss_name)

    # ------------------------------------------------------------------ #
    # Phase 2 — Proper noun scan with entity-context assignment           #
    # ------------------------------------------------------------------ #
    candidates = _scan_proper_noun_candidates(question)

    # Build a small window-based context checker:
    # For each candidate, look at the words within ±4 tokens around it
    # in the lowercased question to decide which filter key to use.

    def _context_around(needle: str, haystack_lower: str, window: int = 40) -> str:
        idx = haystack_lower.find(needle.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(haystack_lower), idx + len(needle) + window)
        return haystack_lower[start:end]

    # T1-E: load role_suffixes from schema once before the loop — avoids a
    # repeated _load_schema() call on every candidate iteration.
    # Fallback to empty set if Employee node has no role_suffixes defined.
    _role_suffixes_schema = frozenset(
        _load_schema()["nodes"].get("Employee", {}).get("role_suffixes", [])
    )

    for cand in candidates:
        cand_lower = cand.lower()

        # Skip names already locked by Phase 1
        if cand_lower in locked_names:
            continue
        # Skip single-word noise
        if cand_lower in _NAME_NOISE:
            continue
        # Skip schema entity names themselves ("Employee", "Project", …)
        if cand in _get_valid_entities():
            continue
        # Skip sub-spans of values already assigned in this phase.
        # e.g. if "API Gateway Rebuild" -> project_name, skip "Gateway Rebuild"
        # which is a subset and would incorrectly become employee_name.
        if any(cand_lower in str(v).lower() and cand_lower != str(v).lower()
               for v in filters.values()):
            continue

        context = _context_around(cand, q_lower)

        # Step 1: check BEFORE_CANDIDATE_SIGNALS (left-context only).
        # Each signal has its own window: tight for ambiguous prepositions like
        # "of" (only fires for "units of [Product]", not "emails of everyone"),
        # wider for unambiguous ones like "for" and "in".
        # R5 guard: if product_supplier was already captured in Phase 0, skip
        # the "for" product_name signal so the supplier name is not ALSO assigned
        # to product_name (e.g. "stock for all items supplied by TechSupply Co").
        assigned_key: Optional[str] = None
        idx = q_lower.find(cand_lower)
        for trigger, fkey, window in [
            ("for",  "product_name",    30),   # "orders for [Standing Desk]"
            ("of",   "product_name",     8),   # "units of [Webcam HD]" — TIGHT to avoid "emails of everyone"
            ("in",   "department_name", 30),   # "employees in [Sales]"
        ]:
            if fkey == "product_name" and "product_supplier" in filters:
                continue  # R5: don't double-assign supplier name to product_name
            left_ctx = q_lower[max(0, idx - window):idx]
            if re.search(rf"\b{re.escape(trigger)}\b", left_ctx):
                assigned_key = fkey
                break

        # Step 2: bidirectional entity-context signals (±40 chars) as fallback.
        if assigned_key is None:
            for trigger, fkey in _ENTITY_CONTEXT_SIGNALS:
                if re.search(rf"\b{re.escape(trigger)}\b", context):
                    assigned_key = fkey
                    break

        # If no entity-context signal found, fall through to person heuristic
        if assigned_key is None:
            # R4: Before treating a 2-word Title-Case as a person name, check
            # if it ends with a known occupational suffix — those are role titles
            # (e.g. "Senior Engineer", "Operations Manager") that belong in
            # employee_role, not employee_name.
            # Suffixes are derived from the roles present in the seed data and
            # schema — NOT hardcoded arbitrary strings.
            # T1-E: role_suffixes now read from graph_schema.yaml Employee node
            # (hoisted to _role_suffixes_schema before this loop — see above).
            cand_words = cand.split()
            last_word = cand_words[-1].lower() if cand_words else ""
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", cand) and last_word in _role_suffixes_schema:
                assigned_key = "employee_role"
            # Two-word Title-Case with no entity signal → likely a person name
            elif re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", cand):
                assigned_key = "employee_name"
            else:
                # Single capitalised word or ambiguous multi-word — skip
                continue

        # Don't overwrite an already-locked / higher-priority filter
        if assigned_key not in filters:
            filters[assigned_key] = cand
            logger.debug("Proper noun assigned: %s → %s", cand, assigned_key)

    # ------------------------------------------------------------------ #
    # Phase 2.5 — Case-insensitive fallback for product & project names   #
    # that appear lowercase in the question (e.g. "webcam hd",            #
    # "platform migration project")                                        #
    # ------------------------------------------------------------------ #
    _PRODUCT_CONTEXT = {"stock", "units", "inventory", "price", "cost",
                        "how many", "in stock", "available"}
    _PROJECT_CONTEXT = {"project", "initiative", "assigned to", "working on",
                        "on the", "part of"}

    # Only run if we haven't already captured the value via Phase 2.
    # N3 FIX: Also skip if product_supplier is already set — a supplier query like
    # "stock for all items supplied by TechSupply Co" has "of the current stock"
    # which would match and extract "The Current" as a phantom product_name.
    if "product_name" not in filters and "product_supplier" not in filters:
        if any(kw in q_lower for kw in _PRODUCT_CONTEXT):
            # Extract any 1-3 word sequence after "of", "for", "about" or
            # before "stock", "units", "inventory", "price" — case-insensitive
            prod_match = re.search(
                r"\b(?:of|for|about)\s+([a-zA-Z0-9][a-zA-Z0-9\s]{1,30}?)"
                r"\s+(?:stock|units|inventory|price|cost|in stock|available|are|is)\b",
                q_lower,
            )
            if not prod_match:
                prod_match = re.search(
                    r"\b([a-zA-Z0-9][a-zA-Z0-9\s]{1,30}?)\s+"
                    r"(?:stock|units|inventory|price|cost)\b",
                    q_lower,
                )
            if prod_match:
                raw_val = prod_match.group(1).strip()
                # T1.4: Reject spans whose first token is a comparison or question word.
                # "most expensive item we have in stock" would otherwise capture
                # "most expensive item we have in" as a product name. We also cap at
                # 4 words — real product names (Webcam HD, Standing Desk, USB-C Hub)
                # are never longer. This guard must not use re.I on [A-Z] checks.
                _COMPARISON_STARTERS = frozenset({
                    "most", "least", "highest", "lowest", "best", "worst",
                    "cheapest", "expensive", "priciest", "top", "bottom",
                    "maximum", "minimum", "any", "all", "every", "which",
                    "what", "who", "how", "the", "we", "our", "i", "you",
                    "have", "has", "do", "does", "more", "less", "greater",
                })
                span_words = raw_val.split()
                first_token = span_words[0].lower() if span_words else ""
                if (first_token not in _COMPARISON_STARTERS
                        and len(span_words) <= 4):
                    cap_val = " ".join(w.capitalize() for w in span_words)
                    if cap_val.lower() not in _NAME_NOISE and len(cap_val) > 2:
                        filters["product_name"] = cap_val
                        logger.debug(
                            "Phase 2.5 product name (case-insensitive): %s", cap_val
                        )

    if "project_name" not in filters:
        if any(kw in q_lower for kw in _PROJECT_CONTEXT):
            # D3b: Collect all 1–4 word spans that immediately precede "project"
            # using finditer (to get ALL matches, not just the leftmost).  Pick
            # the longest span whose words are all non-noise — this gives
            # "platform migration" from "...to the platform migration project"
            # rather than the greedy leftmost "to the platform migration".
            _ph25_starters = frozenset({
                "the", "a", "an", "this", "that", "some", "any", "active",
                "all", "every", "which", "who", "what", "how", "are", "is",
                "do", "does", "did", "have", "has", "where", "when", "why",
            })
            best_proj: Optional[str] = None
            for n_extra in range(4):  # 1-word up to 4-word spans
                pat = (
                    r"\b([a-zA-Z][a-zA-Z0-9]*"
                    + r"(?:\s+[a-zA-Z][a-zA-Z0-9]*)" * n_extra
                    + r")\s+project\b"
                )
                for m in re.finditer(pat, q_lower):
                    span = m.group(1).strip()
                    words = span.split()
                    if (len(words) <= 4
                            and words[0].lower() not in _ph25_starters
                            and not any(w.lower() in _NAME_NOISE for w in words)
                            and len(span) > 3):
                        if best_proj is None or len(span) > len(best_proj):
                            best_proj = span

            if best_proj is None:
                # Fallback: "project [NAME]" form
                proj_after = re.search(
                    r"\bproject\s+([a-zA-Z][a-zA-Z\s]{2,40}?)\s*(?:$|\?|,|\.|\band\b)",
                    q_lower,
                )
                if proj_after:
                    raw_val = proj_after.group(1).strip()
                    words = raw_val.split()
                    if (len(words) <= 4
                            and words[0].lower() not in _ph25_starters
                            and not any(w.lower() in _NAME_NOISE for w in words)
                            and len(raw_val) > 3):
                        best_proj = raw_val

            if best_proj is not None:
                cap_val = " ".join(w.capitalize() for w in best_proj.split())
                filters["project_name"] = cap_val
                logger.debug("Phase 2.5 project name (case-insensitive): %s", cap_val)

    # ------------------------------------------------------------------ #
    # Phase 3 — Possessive / "what is X's" person fallback               #
    # D2 guard: build a set of all values already assigned by Phase 2.   #
    # If the Phase 3 candidate matches any existing filter value, it was  #
    # already correctly classified — skip it to prevent a ghost AND-      #
    # constraint (e.g. employee_name = "Platform Migration").             #
    # ------------------------------------------------------------------ #
    _captured_values = {str(v).lower() for v in filters.values()}

    def _already_captured(candidate: str) -> bool:
        """Return True if candidate is an exact match OR a sub-string of any
        already-assigned filter value (D2 guard — prevents ghost employee_name
        for sub-spans like 'Gateway Rebuild' ⊂ 'API Gateway Rebuild')."""
        c = candidate.lower()
        return c in _captured_values or any(c in val for val in _captured_values)

    if "employee_name" not in filters:
        # Pattern: "What is Sarah Connor's salary" / "Sarah Connor's pay"
        possessive = re.search(
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\'s)?\s+"
            r"(?:salary|pay|role|email|department|budget|project|assigned|hire)",
            question,
        )
        if possessive:
            candidate = possessive.group(1).strip()
            if (candidate.lower() not in _NAME_NOISE
                    and candidate not in _get_valid_entities()
                    and not _already_captured(candidate)):  # D2 guard
                filters["employee_name"] = candidate

    if "employee_name" not in filters:
        # Pattern: "What is/are [Name]'s ..."
        whose = re.search(
            r"\bwhat\s+(?:is|are)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\'s)?\s+\w+",
            question,
        )
        if whose:
            candidate = whose.group(1).strip()
            if (candidate.lower() not in _NAME_NOISE
                    and candidate not in _get_valid_entities()
                    and not _already_captured(candidate)):  # D2 guard
                filters["employee_name"] = candidate

    return filters


def _extract_projections(question: str) -> List[str]:
    q_lower = question.lower()
    projections = []
    for keyword, projection in _PROJECTION_KEYWORDS.items():
        if keyword in q_lower and projection not in projections:
            projections.append(projection)
    return projections


def _detect_entities_from_keywords(question: str) -> List[str]:
    schema_keywords = _get_schema_keywords()
    q_lower = question.lower()
    detected = []

    # T1-A: priority_order is now derived from schema node iteration order rather
    # than a hardcoded list. YAML preserves insertion order (Python 3.7+ dicts do
    # too), so the order in graph_schema.yaml nodes: is the detection priority.
    # This means a new entity added to the schema is automatically checked here
    # without any Python change — completing the T1-A zero-Python-change guarantee.
    priority_order = list(_load_schema()["nodes"].keys())
    for entity in priority_order:
        triggers = schema_keywords["entity_triggers"].get(entity, [])
        for trigger in triggers:
            if re.search(rf"\b{re.escape(trigger)}\b", q_lower):
                if entity not in detected:
                    detected.append(entity)
                break

    return detected


def _detect_question_type(question: str) -> Tuple[str, bool]:
    q_lower = question.lower()
    for pattern, q_type in _QUESTION_TYPE_PATTERNS:
        if pattern.search(q_lower):
            return q_type, True
    return "lookup", False


def _layer1_pattern_extraction(question: str) -> Optional[EntityExtractionResult]:
    q_lower = question.lower()

    if _OFF_TOPIC_PATTERNS.search(q_lower):
        return EntityExtractionResult(
            entities=[],
            filters={},
            projections=[],
            question_type="other",
            extraction_method="pattern",
            confidence_score=ExtractionConfidence.HIGH,
            confidence_breakdown={"off_topic_detected": 1.0},
            escalation_reason="",
        )

    # T2: Graceful decline for query patterns that require SQL capabilities
    # (HAVING, cross-row aggregation subqueries, temporal comparisons) that the
    # current GraphQueryBuilder cannot generate correctly.
    # Detection is purely structural — no entity names are hardcoded.
    _UNSUPPORTED_PATTERNS = re.compile(
        r"\b(more than (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:active\s+)?projects?"  # HAVING COUNT > N
        r"|assigned to (?:more|over|greater) than (?:\d+|one|two|three|four|five)"  # HAVING variant
        r"|(?:higher|more|greater|cheaper|less) than (?:the )?average"              # subquery AVG
        r"|cost more than (?:the )?average"                                          # product subquery
        r"|started after .{3,60}(?:was|were) (?:completed|finished|done)"           # temporal
        r"|after .{3,60}(?:ended|was completed|was finished)"                        # temporal variant
        r")\b",
        re.I,
    )
    if _UNSUPPORTED_PATTERNS.search(q_lower):
        logger.info("T2: unsupported query pattern detected — returning graceful decline")
        return EntityExtractionResult(
            entities=[],
            filters={},
            projections=[],
            question_type="other",
            extraction_method="pattern",
            confidence_score=ExtractionConfidence.HIGH,
            confidence_breakdown={"unsupported_pattern": 1.0},
            escalation_reason=(
                "This question requires comparing across multiple rows "
                "(e.g. HAVING, subquery average, or date comparison between records). "
                "Try rephrasing: e.g. 'Show employees on more than one project' "
                "or 'List projects that started in 2025'."
            ),
        )

    entities = _detect_entities_from_keywords(question)

    # Auto-infer entity from named filter values when no keyword trigger fired.
    # e.g. "What is Alan Turing salary?" has no 'employee/staff/who' keyword but
    # Phase 3 extraction captures employee_name=Alan Turing — infer Employee.
    # Similarly for product_name, department_name, project_name filters.
    if not entities:
        filters_early = _extract_named_values(question)
        _FILTER_TO_ENTITY = {
            "employee_name": "Employee",
            "department_name": "Department",
            "product_name": "Product",
            "project_name": "Project",
            "manager_name": "Employee",
        }
        for fkey, entity in _FILTER_TO_ENTITY.items():
            if fkey in filters_early and entity not in entities:
                entities.append(entity)
                logger.debug("Entity auto-inferred from filter key %s -> %s", fkey, entity)

    if not entities:
        return None

    filters = _extract_named_values(question)
    projections = _extract_projections(question)
    question_type, type_resolved = _detect_question_type(question)

    if len(entities) >= 2:
        question_type = "cross_entity" if question_type == "lookup" else question_type

    # R2: A "list all X" query with entities but no named filter is structurally
    # complete — it's a deliberate request for all records of that entity type.
    # Penalising it for missing filters causes unnecessary LLM escalation (20s+).
    is_filterless_list = question_type == "list" and bool(entities) and not filters and not projections
    has_clear_structure = bool(entities and (filters or projections)) or is_filterless_list
    confidence, breakdown = _compute_confidence(
        entities, filters, question_type,
        has_clear_structure, projections, "pattern"
    )

    if confidence < ExtractionConfidence.TRUST_THRESHOLD:
        return None

    logger.debug("Layer 1 extraction: entities=%s confidence=%.2f", entities, confidence)

    return EntityExtractionResult(
        entities=entities,
        filters=filters,
        projections=projections,
        question_type=question_type,
        extraction_method="pattern",
        confidence_score=confidence,
        confidence_breakdown=breakdown,
        escalation_reason="",
    )


def _layer2_spacy_extraction(question: str) -> Optional[EntityExtractionResult]:
    nlp = _load_nlp()
    if nlp is None:
        return None

    doc = nlp(question)
    entities = _detect_entities_from_keywords(question)
    filters = _extract_named_values(question)
    projections = _extract_projections(question)
    question_type, type_resolved = _detect_question_type(question)

    has_manager_relation = False
    has_assignment_relation = False

    for token in doc:
        if token.lemma_ in {"manage", "lead", "run", "head", "oversee"}:
            has_manager_relation = True
            if "Project" not in entities:
                entities.append("Project")
            subj_tokens = [t for t in token.subtree if t.dep_ in {"nsubj", "nsubjpass"}]
            obj_tokens = [t for t in token.subtree if t.dep_ in {"dobj", "pobj", "attr"}]

            for obj_tok in obj_tokens:
                span_text = obj_tok.text
                if obj_tok.ent_type_ == "PERSON" or (obj_tok.text[0].isupper() and len(obj_tok.text) > 2):
                    if "manager_name" not in filters:
                        filters["manager_name"] = span_text

            for subj_tok in subj_tokens:
                if subj_tok.ent_type_ == "PERSON":
                    if "manager_name" not in filters:
                        filters["manager_name"] = subj_tok.text

        if token.lemma_ in {"assign", "work", "involve", "participate"}:
            has_assignment_relation = True
            if "Employee" not in entities:
                entities.append("Employee")

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if "employee_name" not in filters and "manager_name" not in filters:
                if _MANAGER_PATTERNS.search(question[:ent.start_char]):
                    filters["manager_name"] = ent.text
                else:
                    filters["employee_name"] = ent.text
        elif ent.label_ == "ORG":
            if "department_name" not in filters:
                valid_entities = _get_valid_entities()
                if ent.text not in valid_entities:
                    filters["department_name"] = ent.text

    if len(entities) >= 2:
        question_type = "cross_entity" if question_type == "lookup" else question_type

    has_clear_structure = has_manager_relation or has_assignment_relation or bool(doc.ents)
    confidence, breakdown = _compute_confidence(
        entities, filters, question_type,
        has_clear_structure, projections, "spacy"
    )

    if confidence < ExtractionConfidence.TRUST_THRESHOLD:
        return None

    logger.debug("Layer 2 spaCy extraction: entities=%s confidence=%.2f", entities, confidence)

    return EntityExtractionResult(
        entities=entities,
        filters=filters,
        projections=projections,
        question_type=question_type,
        extraction_method="spacy",
        confidence_score=confidence,
        confidence_breakdown=breakdown,
        escalation_reason="",
    )


@lru_cache(maxsize=256)
def _cached_llm_extraction(question_normalized: str) -> Optional[str]:
    model_cfg = _load_model_config()
    valid_entities = _get_valid_entities()
    schema = _load_schema()

    entity_descriptions = []
    for entity, node_data in schema["nodes"].items():
        filterable_cols = list(node_data.get("filterable_columns", {}).keys())
        entity_descriptions.append(f"  - {entity}: filterable by {filterable_cols}")

    entity_desc_str = "\n".join(entity_descriptions)

    filter_key_docs = (
        "employee_name, department_name, product_name, project_name, "
        "order_status (pending/processing/shipped/delivered/cancelled), "
        "project_status (planning/active/completed/on_hold), "
        "manager_name, project_department, category, employee_role"
    )

    prompt = f"""You are a database entity extractor. Return ONLY a valid JSON object, nothing else.

AVAILABLE ENTITIES AND THEIR FILTERABLE FIELDS:
{entity_desc_str}

VALID ENTITY NAMES (use exactly these): {valid_entities}

QUESTION TYPES: lookup, list, comparison, aggregation, cross_entity, other

FILTER KEYS: {filter_key_docs}

RULES:
1. Return ONLY JSON. No markdown, no explanation.
2. entities: list of entity names from the valid list above only.
3. filters: dict of filter_key to value. Only include explicitly stated filters.
4. projections: list of field names the user wants to see.
5. question_type: exactly one type from the list above.
6. If question cannot be answered from this database schema, use question_type "other" with empty entities.
7. "managed by X" → filter key "manager_name" with value X.
8. "projects in X department" → filter key "project_department" with value X.
9. For list questions with 2+ entities, use question_type "cross_entity".

EXAMPLES:
Q: "What is Sarah Connor salary?" → {{"entities": ["Employee"], "filters": {{"employee_name": "Sarah Connor"}}, "projections": ["salary", "role"], "question_type": "lookup"}}
Q: "Who manages API Gateway Rebuild?" → {{"entities": ["Project"], "filters": {{"project_name": "API Gateway Rebuild"}}, "projections": ["manager_name"], "question_type": "lookup"}}
Q: "Employees on projects managed by Don Draper" → {{"entities": ["Employee", "Project"], "filters": {{"manager_name": "Don Draper"}}, "projections": ["employee_name", "project_name", "assignment_role"], "question_type": "cross_entity"}}
Q: "What is the weather?" → {{"entities": [], "filters": {{}}, "projections": [], "question_type": "other"}}

User question: "{question_normalized}"

JSON:"""

    try:
        resp = requests.post(
            f"{model_cfg['ollama_base_url']}/api/generate",
            json={
                "model": model_cfg["fast_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 200,
                    "num_ctx": 2048,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


def _parse_and_validate_llm_response(raw: str) -> Optional[EntityExtractionResult]:
    if not raw:
        return None

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("LLM response contains no JSON object")
        return None

    json_str = cleaned[start:end]

    parsed = None
    for attempt in [
        lambda s: json.loads(s),
        lambda s: json.loads(s.replace("'", '"')),
        lambda s: json.loads(re.sub(r",\s*([}\]])", r"\1", s)),
    ]:
        try:
            parsed = attempt(json_str)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if parsed is None:
        logger.warning("All JSON parse attempts failed for LLM response")
        return None

    valid_entities = set(_get_valid_entities())
    valid_types = {"lookup", "list", "comparison", "aggregation", "cross_entity", "other"}

    raw_entities = parsed.get("entities", [])
    entities = [e for e in raw_entities if e in valid_entities]

    invalid = set(raw_entities) - valid_entities
    if invalid:
        logger.warning("LLM returned invalid entities (rejected): %s", invalid)

    filters = {k: str(v) for k, v in parsed.get("filters", {}).items() if v}
    projections = [str(p) for p in parsed.get("projections", [])]
    q_type = parsed.get("question_type", "lookup")

    if q_type not in valid_types:
        logger.warning("LLM returned invalid question_type '%s', defaulting to lookup", q_type)
        q_type = "lookup"

    if len(set(entities)) >= 2 and q_type in ("lookup", "list"):
        q_type = "cross_entity"

    has_clear_structure = bool(entities and (filters or projections))
    confidence, breakdown = _compute_confidence(
        entities, filters, q_type, has_clear_structure, projections, "llm"
    )

    return EntityExtractionResult(
        entities=entities,
        filters=filters,
        projections=projections,
        question_type=q_type,
        extraction_method="llm",
        confidence_score=confidence,
        confidence_breakdown=breakdown,
        escalation_reason="",
    )


def _layer3_llm_extraction(question: str) -> EntityExtractionResult:
    question_normalized = re.sub(r"\s+", " ", question.strip().lower())

    start = time.time()
    raw = _cached_llm_extraction(question_normalized)
    latency_ms = round((time.time() - start) * 1000, 2)

    result = _parse_and_validate_llm_response(raw)

    if result is None:
        logger.warning("LLM extraction produced no valid result for: '%s'", question)
        return EntityExtractionResult(
            entities=[],
            filters={},
            projections=[],
            question_type="other",
            extraction_method="llm_failed",
            latency_ms=latency_ms,
            confidence_score=0.0,
            confidence_breakdown={"llm_parse_failed": 0.0},
            escalation_reason="LLM returned unparseable response",
        )

    result.latency_ms = latency_ms

    if result.confidence_score < ExtractionConfidence.TRUST_THRESHOLD and result.question_type != "other":
        logger.warning(
            "LLM extraction confidence %.2f below threshold for: '%s'",
            result.confidence_score, question
        )
        result.escalation_reason = (
            f"LLM confidence {result.confidence_score:.2f} below trust threshold "
            f"{ExtractionConfidence.TRUST_THRESHOLD}"
        )

    logger.debug(
        "Layer 3 LLM extraction: entities=%s confidence=%.2f method=%s",
        result.entities, result.confidence_score, result.extraction_method
    )
    return result


def extract_entities(question: str) -> EntityExtractionResult:
    start = time.time()
    question = question.strip()

    result = _layer1_pattern_extraction(question)
    if result is not None:
        result.latency_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            "Extraction via pattern: entities=%s type=%s confidence=%.2f latency=%.0fms",
            result.entities, result.question_type, result.confidence_score, result.latency_ms
        )
        return result

    result = _layer2_spacy_extraction(question)
    if result is not None:
        result.latency_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            "Extraction via spaCy: entities=%s type=%s confidence=%.2f latency=%.0fms",
            result.entities, result.question_type, result.confidence_score, result.latency_ms
        )
        return result

    logger.info("Layers 1+2 insufficient (confidence below threshold), escalating to LLM")
    result = _layer3_llm_extraction(question)
    result.latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(
        "Extraction via LLM: entities=%s type=%s confidence=%.2f latency=%.0fms",
        result.entities, result.question_type, result.confidence_score, result.latency_ms
    )
    return result
