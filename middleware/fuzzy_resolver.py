from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_name_index: Optional[List[str]] = None  # all known name strings from DB
_index_loaded = False

_NAME_QUERIES = [
    "SELECT name FROM employees",
    "SELECT name FROM departments",
    "SELECT name FROM projects",
    "SELECT name FROM products",
]

_NAME_FILTER_KEYS = {
    "employee_name",
    "manager_name",
    "reports_to_name",
    "department_name",
    "project_name",
    "product_name",
}

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
            continue  
        clean_value = value.replace("'s", "").strip()

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
    global _name_index, _index_loaded
    _name_index = None
    _index_loaded = False
