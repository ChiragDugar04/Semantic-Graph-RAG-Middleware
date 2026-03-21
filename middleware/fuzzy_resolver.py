"""
fuzzy_resolver.py — In-Memory Fuzzy Name Resolution
=====================================================
Fix 5: Resolves extracted filter values to their canonical DB strings
BEFORE the SQL query is built.

Problem it solves:
  - Small LLMs (1.5b) truncate names: "Leslie Knop" → "Leslie Knope"
  - Slight misspellings: "Turing" instead of "Alan Turing" still resolves
  - Possessive artifacts: "Draper's" → "Don Draper"

How it works:
  1. On first use, loads all name/text values from DB into an in-memory dict
  2. When a filter value is extracted, runs rapidfuzz WRatio against known values
  3. If score >= threshold (85), replaces with canonical DB string
  4. Logs the substitution so it's traceable in debug output

Usage in pipeline.py / entity_extractor.py:
    from middleware.fuzzy_resolver import resolve_filter_values
    extraction.filters = resolve_filter_values(extraction.filters)

Design:
  - Lazy-loaded: DB is only queried once per process lifetime
  - Threshold is conservative (85) to avoid false matches
  - Only resolves text/name fields — skips enum fields (status, category, etc.)
  - Works without rapidfuzz installed: silently falls back to no-op

No schema changes required. Adding a new name-type table requires only
adding it to _NAME_QUERIES below.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded name index
# ---------------------------------------------------------------------------
_name_index: Optional[List[str]] = None  # all known name strings from DB
_index_loaded = False

# Queries that pull name/text values from the DB.
# Each returns a single string column of canonical names.
_NAME_QUERIES = [
    "SELECT name FROM employees",
    "SELECT name FROM departments",
    "SELECT name FROM projects",
    "SELECT name FROM products",
]

# Filter keys that are name-type (should be fuzzy-resolved).
# Enum-type and numeric keys are excluded — we don't fuzzy-match "pending" or "active".
_NAME_FILTER_KEYS = {
    "employee_name",
    "manager_name",
    "reports_to_name",
    "department_name",
    "project_name",
    "product_name",
}

# Minimum WRatio score to accept a fuzzy match (0-100).
# 85 = accepts "Knop" → "Leslie Knope", rejects accidental cross-entity matches.
FUZZY_THRESHOLD = 85


def _load_name_index() -> List[str]:
    """Load all name strings from DB into memory. Called once per process."""
    global _name_index, _index_loaded
    if _index_loaded:
        return _name_index or []

    _index_loaded = True
    try:
        from middleware.query_executor import _get_connection
        conn = _get_connection()
        cursor = conn.cursor()
        names: List[str] = []
        for q in _NAME_QUERIES:
            try:
                cursor.execute(q)
                rows = cursor.fetchall()
                names.extend(row[0] for row in rows if row[0])
            except Exception as e:
                logger.warning("fuzzy_resolver: skipped query '%s': %s", q, e)
        cursor.close()
        conn.close()
        _name_index = names
        logger.info("fuzzy_resolver: loaded %d names into index", len(names))
    except Exception as e:
        logger.warning("fuzzy_resolver: failed to load name index: %s — fuzzy matching disabled", e)
        _name_index = []

    return _name_index or []


def resolve_filter_values(filters: Dict[str, str]) -> Dict[str, str]:
    """
    Resolve name-type filter values to their canonical DB strings using fuzzy matching.

    Args:
        filters: dict of filter_key → extracted_value from LLM/regex

    Returns:
        New dict with name-type values corrected to canonical DB strings where
        a high-confidence match exists. Non-name keys are passed through unchanged.

    Examples:
        {"employee_name": "Leslie Knop"} → {"employee_name": "Leslie Knope"}
        {"reports_to_name": "Draper"}    → {"reports_to_name": "Don Draper"}
        {"order_status": "pending"}      → {"order_status": "pending"}  (unchanged)
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        logger.debug("rapidfuzz not installed — fuzzy name resolution disabled")
        return filters

    if not filters:
        return filters

    name_index = _load_name_index()
    if not name_index:
        return filters

    resolved = dict(filters)
    for key, value in filters.items():
        if key not in _NAME_FILTER_KEYS:
            continue  # skip enum/numeric fields

        # Strip common possessive artifacts before matching
        clean_value = value.replace("'s", "").strip()

        # rapidfuzz.process.extractOne returns (match, score, index) or None
        result = process.extractOne(
            clean_value,
            name_index,
            scorer=fuzz.WRatio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result is None:
            continue

        canonical, score, _ = result
        if canonical != value:
            logger.info(
                "fuzzy_resolver: '%s'='%s' → '%s' (score=%d)",
                key, value, canonical, score,
            )
            resolved[key] = canonical

    return resolved


def reset_index() -> None:
    """Force reload of name index on next call (e.g. after DB seed changes)."""
    global _name_index, _index_loaded
    _name_index = None
    _index_loaded = False