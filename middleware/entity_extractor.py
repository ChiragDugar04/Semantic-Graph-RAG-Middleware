from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from middleware.models import EntityExtractionResult, ExtractionConfidence

logger = logging.getLogger(__name__)


_schema_cache: Optional[Dict[str, Any]] = None
_schema_mtime: float = 0.0
_schema_prompt_cache: Optional[str] = None


def _load_schema() -> Dict[str, Any]:
    global _schema_cache, _schema_mtime, _schema_prompt_cache
    schema_path = Path(__file__).parent.parent / "config" / "graph_schema.yaml"
    try:
        mtime = os.path.getmtime(schema_path)
    except OSError:
        mtime = 0.0
    if _schema_cache is None or mtime != _schema_mtime:
        with open(schema_path, "r") as f:
            _schema_cache = yaml.safe_load(f)
        _schema_mtime = mtime
        _schema_prompt_cache = None
        logger.info("Schema loaded from %s", schema_path)
    return _schema_cache


def _load_model_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


def _get_valid_entities() -> List[str]:
    return list(_load_schema()["nodes"].keys())


def _build_valid_filter_keys(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    valid: Dict[str, Dict[str, Any]] = {}
    nodes = schema.get("nodes", {})
    edges = schema.get("edges", [])

    for entity, node_data in nodes.items():
        entity_lower = entity.lower()
        for col_key, col_meta in node_data.get("filterable_columns", {}).items():
            fkey = f"{entity_lower}_{col_key}"
            enum_vals = col_meta.get("enum_values", [])
            valid_vals = col_meta.get("valid_values", [])
            allowed: Optional[List[str]] = None
            if enum_vals:
                allowed = [e.get("canonical", "") for e in enum_vals]
            elif valid_vals:
                allowed = [v.get("canonical", "") for v in valid_vals]
            valid[fkey] = {
                "match": col_meta.get("match_type", "like"),
                "allowed_values": allowed,
            }
        for supp_key in node_data.get("filter_supplements", {}):
            valid[supp_key] = {"match": "subquery", "allowed_values": None}

    for edge in edges:
        for supp_key in edge.get("filter_supplements", {}):
            valid[supp_key] = {"match": "subquery", "allowed_values": None}

    return valid


def _build_schema_prompt() -> str:
    global _schema_prompt_cache
    if _schema_prompt_cache is not None:
        return _schema_prompt_cache

    schema = _load_schema()
    nodes = schema.get("nodes", {})
    edges = schema.get("edges", [])

    lines: List[str] = []

    lines.append("TABLES:")
    for entity, node_data in nodes.items():
        alias = node_data.get("alias", entity.lower()[0])
        table = node_data.get("table", entity.lower())
        col_parts: List[str] = []
        for col_key, col_meta in node_data.get("filterable_columns", {}).items():
            fkey = f"{entity.lower()}_{col_key}"
            enum_vals = col_meta.get("enum_values", [])
            valid_vals = col_meta.get("valid_values", [])
            is_numeric = col_meta.get("is_numeric", False)
            sqd = col_meta.get("scalar_subquery_defaults")
            if enum_vals:
                vals = "|".join(e.get("canonical", "") for e in enum_vals)
                col_parts.append(f"{fkey}[{vals}]")
            elif valid_vals:
                vals = "|".join(v.get("canonical", "") for v in valid_vals)
                col_parts.append(f"{fkey}[{vals}]")
            elif sqd:
                col_parts.append(f"{fkey}(numeric,avg_compare_ok)")
            elif is_numeric:
                col_parts.append(f"{fkey}(numeric)")
            else:
                col_parts.append(fkey)
        supp_keys = list(node_data.get("filter_supplements", {}).keys())
        for edge in edges:
            for sk in edge.get("filter_supplements", {}):
                if sk not in supp_keys:
                    supp_keys.append(sk)
        if supp_keys:
            col_parts.append("cross:" + ",".join(supp_keys))
        lines.append(f"  {entity}({alias}/{table}): {' | '.join(col_parts)}")

    lines.append("")
    lines.append("JOINS:")
    for edge in edges:
        fn = edge["from_node"]
        tn = edge["to_node"]
        jt = edge.get("junction_table")
        if jt:
            lines.append(f"  {fn}->{tn} via {jt}")
        else:
            lines.append(f"  {fn}->{tn}")

    lines.append("")
    lines.append("QUESTION TYPES: lookup|list|comparison|aggregation|cross_entity|having_count|temporal_filter|computed_delta|group_rank|other")
    lines.append("")
    lines.append("RULES:")
    lines.append("  Use only entity names and filter keys shown above.")
    lines.append("  For enum fields use only the listed canonical values exactly.")
    lines.append("  Never invent filter values not stated in the question.")
    lines.append("  having_count: question asks who appears in MORE THAN N related records.")
    lines.append("  temporal_filter: question asks for records AFTER or BEFORE a date.")
    lines.append("  computed_delta: question asks for the GAP or DIFFERENCE between highest and lowest of a numeric column.")
    lines.append("  group_rank: question asks which GROUP has the MOST or LEAST of something.")
    lines.append("  comparison: question asks for highest/lowest/most/least of one entity.")
    lines.append("  cross_entity: question involves two or more entities joined together.")
    lines.append("  Return other if the question is off-topic or asks for something unsupported.")

    _schema_prompt_cache = "\n".join(lines)
    return _schema_prompt_cache


def _build_prompt(question: str) -> str:
    schema_ctx = _build_schema_prompt()
    prompt = (
        'You are a JSON extractor. Parse the question into exactly this JSON structure:\n'
        '{"entities":[],"filters":{},"question_type":"","projections":[]}\n\n'
        + schema_ctx
        + "\n\nEXAMPLES:\n"
        'Q:"What is Alan Turing salary" -> {"entities":["Employee"],"filters":{"employee_name":"Alan Turing"},"question_type":"lookup","projections":["salary"]}\n'
        'Q:"Show me all pending orders" -> {"entities":["Order"],"filters":{"order_status":"pending"},"question_type":"list","projections":[]}\n'
        'Q:"which department does Sarah Connor belong to" -> {"entities":["Employee","Department"],"filters":{"employee_name":"Sarah Connor"},"question_type":"lookup","projections":[]}\n'
        'Q:"which employees are in Leslie Knope\'s team" -> {"entities":["Employee"],"filters":{"reports_to_name":"Leslie Knope"},"question_type":"list","projections":[]}\n'
        'Q:"who placed the order for the Standing Desk" -> {"entities":["Order"],"filters":{"product_name":"Standing Desk"},"question_type":"lookup","projections":[]}\n'
        'Q:"show everyone working on projects managed by Don Draper" -> {"entities":["Employee","Project"],"filters":{"manager_name":"Don Draper"},"question_type":"list","projections":[]}\n'
        'Q:"Engineering employees on active projects" -> {"entities":["Employee","Project"],"filters":{"department_name":"Engineering","project_status":"active"},"question_type":"cross_entity","projections":[]}\n'
        'Q:"who reports to Michael Scott" -> {"entities":["Employee"],"filters":{"reports_to_name":"Michael Scott"},"question_type":"lookup","projections":[]}\n'
        'Q:"what is the most expensive product" -> {"entities":["Product"],"filters":{},"question_type":"comparison","projections":[]}\n'
        'Q:"show me all electronics we have in stock" -> {"entities":["Product"],"filters":{"product_category":"Electronics"},"question_type":"list","projections":[]}\n'
        'Q:"list all office supply products" -> {"entities":["Product"],"filters":{"product_category":"Office Supply"},"question_type":"list","projections":[]}\n'
        'Q:"show me all furniture items" -> {"entities":["Product"],"filters":{"product_category":"Furniture"},"question_type":"list","projections":[]}\n'
        'Q:"who manages the team that Sarah Connor works in" -> {"entities":["Employee","Department"],"filters":{"employee_name":"Sarah Connor"},"question_type":"lookup","projections":[]}\n'
        'Q:"who is the manager of the department Grace Hopper belongs to" -> {"entities":["Employee","Department"],"filters":{"employee_name":"Grace Hopper"},"question_type":"lookup","projections":[]}\n'
        'Q:"how many employees are assigned to more than one project" -> {"entities":["Employee"],"filters":{"having_threshold":"1"},"question_type":"having_count","projections":[]}\n'
        'Q:"rank departments by total salary budget" -> {"entities":["Department"],"filters":{},"question_type":"aggregation","projections":[]}\n'
        'Q:"show me delivered orders placed by people in Sales" -> {"entities":["Order"],"filters":{"order_status":"delivered","department_name":"Sales"},"question_type":"list","projections":[]}\n'
        'Q:"list all pending orders for furniture items" -> {"entities":["Order","Product"],"filters":{"order_status":"pending","product_category":"Furniture"},"question_type":"list","projections":[]}\n'
        'Q:"what did Dwight Schrute order recently" -> {"entities":["Order"],"filters":{"employee_name":"Dwight Schrute"},"question_type":"lookup","projections":[]}\n'
        'Q:"what has Jim Halpert purchased" -> {"entities":["Order"],"filters":{"employee_name":"Jim Halpert"},"question_type":"lookup","projections":[]}\n'
        'Q:"are there any employees in Operations who are on an Engineering project" -> {"entities":["Employee","Project"],"filters":{"department_name":"Operations"},"question_type":"cross_entity","projections":[]}\n'
        'Q:"what is the weather" -> {"entities":[],"filters":{},"question_type":"other","projections":[]}\n'
        f'\nQ:"{question}" -> '
    )
    return prompt


def _call_ollama(prompt: str) -> Optional[str]:
    model_cfg = _load_model_config()
    timeout = model_cfg.get("ollama_timeout", 180)
    try:
        resp = requests.post(
            f"{model_cfg['ollama_base_url']}/api/generate",
            json={
                "model": model_cfg.get("extraction_model", model_cfg.get("fast_model", "phi3:mini")),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": model_cfg.get("extraction_temperature", 0.0),
                    "num_predict": model_cfg.get("max_tokens_extraction", 250),
                    "num_ctx": model_cfg.get("extraction_num_ctx", 2048),
                },
                "keep_alive": model_cfg.get("keep_alive", "10m"),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        logger.debug("Ollama extraction raw: %s", raw[:300])
        return raw
    except requests.RequestException as exc:
        logger.warning("Ollama extraction call failed: %s", exc)
        return None


def _validate_and_build(
    raw: Optional[str],
    question_normalized: str = "",
) -> Optional[EntityExtractionResult]:
    if not raw:
        return None

    schema = _load_schema()
    valid_entities = set(schema["nodes"].keys())
    valid_types = {
        "lookup", "list", "comparison", "aggregation", "cross_entity",
        "group_rank", "having_count", "temporal_filter", "computed_delta", "other",
    }
    valid_filter_keys = _build_valid_filter_keys(schema)

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("No JSON object in LLM response: %s", raw[:120])
        return None

    json_str = cleaned[start:end]
    parsed: Optional[Dict] = None
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
        logger.warning("JSON parse failed after 3 attempts: %s", json_str[:120])
        return None

    raw_entities = [e for e in parsed.get("entities", []) if isinstance(e, str)]
    entities = [e for e in raw_entities if e in valid_entities]
    invalid_ents = set(raw_entities) - valid_entities
    if invalid_ents:
        logger.warning("Stripping invalid entities: %s", invalid_ents)

    def _coerce(v: Any) -> str:
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v)

    raw_filters = {k: _coerce(v) for k, v in parsed.get("filters", {}).items() if v}
    filters: Dict[str, str] = {}
    for fkey, fval in raw_filters.items():
        if fkey == "having_threshold":
            try:
                int(fval)
                filters[fkey] = fval
            except ValueError:
                logger.warning("having_threshold value not integer: %s", fval)
            continue
        if fkey not in valid_filter_keys:
            logger.warning("Stripping unknown filter key: '%s'", fkey)
            continue
        allowed = valid_filter_keys[fkey].get("allowed_values")
        if allowed is not None and fval not in allowed:
            logger.warning(
                "Stripping invalid enum '%s' for key '%s' — allowed: %s",
                fval, fkey, allowed,
            )
            continue
        filters[fkey] = fval

    _SUPERLATIVE_INDICATORS = {
        "most", "highest", "lowest", "least", "best", "worst",
        "top", "bottom", "rank", "ranked", "ranking", "expensive",
        "cheapest", "maximum", "minimum", "max", "min",
    }
    q_words = set(question_normalized.split()) if question_normalized else set()
    if q_words & _SUPERLATIVE_INDICATORS:
        numeric_keys: set = set()
        for _ent, _nd in schema.get("nodes", {}).items():
            for _ck, _cm in _nd.get("filterable_columns", {}).items():
                if _cm.get("is_numeric"):
                    numeric_keys.add(f"{_ent.lower()}_{_ck}")
        for fkey in list(filters.keys()):
            if fkey in numeric_keys and valid_filter_keys.get(fkey, {}).get("allowed_values") is None:
                logger.info("Ranking guard: stripped %s='%s'", fkey, filters[fkey])
                del filters[fkey]

    q_type = parsed.get("question_type", "lookup")
    if q_type not in valid_types:
        logger.warning("Invalid question_type '%s' — defaulting to lookup", q_type)
        q_type = "lookup"

    if len(set(entities)) >= 2 and q_type in ("lookup", "list"):
        q_type = "cross_entity"

    projections = [str(p) for p in parsed.get("projections", []) if p]

    if entities or q_type == "other":
        confidence = ExtractionConfidence.HIGH
        breakdown = {"llm_schema_validated": 1.0}
    else:
        confidence = ExtractionConfidence.LOW
        breakdown = {"llm_no_entities": 0.4}

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


_MONTH_MAP: Dict[str, str] = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _extract_date_value(question_normalized: str) -> Optional[str]:
    q = question_normalized

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(
        r"\b(" + "|".join(_MONTH_MAP.keys()) + r")\s+(\d{1,2}),?\s+(\d{4})\b", q
    )
    if m:
        return f"{m.group(3)}-{_MONTH_MAP[m.group(1)]}-{m.group(2).zfill(2)}"

    m = re.search(
        r"\b(\d{1,2})\s+(" + "|".join(_MONTH_MAP.keys()) + r"),?\s+(\d{4})\b", q
    )
    if m:
        return f"{m.group(3)}-{_MONTH_MAP[m.group(2)]}-{m.group(1).zfill(2)}"

    m = re.search(r"\b(" + "|".join(_MONTH_MAP.keys()) + r")\s+(\d{4})\b", q)
    if m:
        return f"{m.group(2)}-{_MONTH_MAP[m.group(1)]}-01"

    m = re.search(r"\b(?:mid-?|early |late )(\d{4})\b", q)
    if m:
        if "mid" in q:
            return f"{m.group(1)}-07-01"
        if "early" in q:
            return f"{m.group(1)}-01-01"
        return f"{m.group(1)}-10-01"

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return f"{m.group(1)}-01-01"

    return None


def _extract_threshold(question_normalized: str, threshold_signals: Dict[str, int]) -> int:
    for word, value in sorted(threshold_signals.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(word) + r"\b", question_normalized):
            return value
    return 1


def _get_embedding(text: str) -> Optional[List[float]]:
    model_cfg = _load_model_config()
    embed_model = model_cfg.get("embed_model", "nomic-embed-text:latest")
    timeout = model_cfg.get("ollama_timeout", 180)
    try:
        resp = requests.post(
            f"{model_cfg['ollama_base_url']}/api/embeddings",
            json={"model": embed_model, "prompt": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception as exc:
        logger.warning("Embedding call failed: %s", exc)
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _filter_values_present_in_question(
    filters: Dict[str, Any],
    question_normalized: str,
) -> bool:
    for fkey, fval in filters.items():
        if fkey in ("having_threshold",):
            continue
        if any(fkey.endswith(s) for s in ("_direction", "_column_sql", "_after_op", "_before_op")):
            continue
        if fkey.startswith("delta_"):
            continue
        fval_lower = str(fval).lower()
        if fval_lower and fval_lower not in question_normalized:
            logger.info(
                "Session cache REJECTED: filter value '%s'='%s' not found in question '%s'",
                fkey, fval, question_normalized[:80],
            )
            return False
    return True


def _session_cache_lookup(
    question_normalized: str,
    embedding: List[float],
    session_cache: List[Dict[str, Any]],
) -> Optional[EntityExtractionResult]:
    model_cfg = _load_model_config()
    threshold = model_cfg.get("semantic_cache_threshold", 0.95)

    best_score = 0.0
    best_entry = None

    for entry in session_cache:
        cached_emb = entry.get("embedding")
        if not cached_emb:
            continue
        score = _cosine_similarity(embedding, cached_emb)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None or best_score < threshold:
        return None

    cached_filters = best_entry.get("extraction", {}).get("filters", {})
    if not _filter_values_present_in_question(cached_filters, question_normalized):
        return None

    logger.info(
        "Session cache HIT: score=%.4f question='%s' matched='%s'",
        best_score, question_normalized[:60], best_entry.get("question", "")[:60],
    )
    best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
    ext_data = best_entry["extraction"]
    return EntityExtractionResult(
        entities=ext_data.get("entities", []),
        filters=ext_data.get("filters", {}),
        projections=ext_data.get("projections", []),
        question_type=ext_data.get("question_type", "lookup"),
        extraction_method="cache",
        confidence_score=ExtractionConfidence.HIGH,
        confidence_breakdown={"session_cache": round(best_score, 4)},
        escalation_reason="",
    )


def _session_cache_store(
    question_normalized: str,
    embedding: List[float],
    result: EntityExtractionResult,
    session_cache: List[Dict[str, Any]],
) -> None:
    entry = {
        "question": question_normalized,
        "embedding": embedding,
        "extraction": {
            "entities": result.entities,
            "filters": result.filters,
            "projections": result.projections,
            "question_type": result.question_type,
        },
        "hit_count": 0,
    }
    session_cache.append(entry)
    logger.info(
        "Session cache stored: '%s' (%d entries)",
        question_normalized[:60], len(session_cache),
    )


@lru_cache(maxsize=256)
def _cached_llm_extract(question_normalized: str) -> Optional[EntityExtractionResult]:
    prompt = _build_prompt(question_normalized)
    raw = _call_ollama(prompt)
    return _validate_and_build(raw, question_normalized)


def _post_process_having_count(
    result: EntityExtractionResult,
    question_normalized: str,
) -> EntityExtractionResult:
    if result.question_type != "having_count":
        return result
    if "having_threshold" in result.filters:
        return result
    schema = _load_schema()
    anchor = result.entities[0] if result.entities else "Employee"
    node_data = schema.get("nodes", {}).get(anchor, {})
    hcd = node_data.get("having_count_defaults", {})
    threshold = _extract_threshold(
        question_normalized,
        hcd.get("threshold_signals", {}),
    )
    new_filters = dict(result.filters)
    new_filters["having_threshold"] = str(threshold)
    return result.model_copy(update={"filters": new_filters})


def _post_process_temporal(
    result: EntityExtractionResult,
    question_normalized: str,
) -> EntityExtractionResult:
    if result.question_type != "temporal_filter":
        return result

    schema = _load_schema()
    anchor = result.entities[0] if result.entities else None
    if not anchor:
        return result

    node_data = schema.get("nodes", {}).get(anchor, {})
    temporal_columns = node_data.get("temporal_columns", [])
    if not temporal_columns:
        return result

    date_val = _extract_date_value(question_normalized)
    if not date_val:
        logger.warning("temporal_filter detected but no date found in: '%s'", question_normalized)
        return result.model_copy(update={"question_type": "list"})

    direction = "after"
    before_keywords = ["before", "until", "up to", "ending before", "due before"]
    for kw in before_keywords:
        if kw in question_normalized:
            direction = "before"
            break

    tc = temporal_columns[0]
    for col in temporal_columns:
        col_alias = col.get("column_alias", "")
        if col_alias in question_normalized or col.get("column_sql", "") in question_normalized:
            tc = col
            break

    param_key = tc["param_key"]
    new_filters = {
        param_key: date_val,
        f"{param_key}_direction": direction,
        f"{param_key}_column_sql": tc["column_sql"],
        f"{param_key}_after_op": tc.get("after_operator", ">="),
        f"{param_key}_before_op": tc.get("before_operator", "<="),
    }
    return result.model_copy(update={"filters": new_filters})


def _post_process_delta(
    result: EntityExtractionResult,
    question_normalized: str,
) -> EntityExtractionResult:
    if result.question_type != "computed_delta":
        return result

    if "delta_column_sql" in result.filters:
        return result

    schema = _load_schema()
    anchor = result.entities[0] if result.entities else None
    if not anchor:
        return result

    node_data = schema.get("nodes", {}).get(anchor, {})
    delta_defaults = node_data.get("delta_defaults", [])
    if not delta_defaults:
        return result

    dd = delta_defaults[0]
    new_filters = dict(result.filters)
    new_filters["delta_column_sql"] = dd["column_sql"]
    new_filters["delta_max_alias"] = dd["max_alias"]
    new_filters["delta_min_alias"] = dd["min_alias"]
    new_filters["delta_alias"] = dd["delta_alias"]
    new_filters["delta_is_currency"] = str(dd.get("is_currency", False))
    return result.model_copy(update={"filters": new_filters})


def _post_process_project_manager_lookup(
    result: EntityExtractionResult,
    question_normalized: str,
) -> EntityExtractionResult:
    """
    Detects "who manages / who leads / who is in charge of [project]" questions
    and rewrites the extraction so the query builder returns only the manager,
    NOT all project_assignment rows.

    Detection: schema-driven via Project.manager_lookup_triggers in graph_schema.yaml.
    Action: inject filter key "project_manager_only"="true" and collapse entities
            to ["Project"] only (single-node lookup — no junction join needed).

    The query builder (_build_project_manager_query) reads project_manager_only
    and emits:
        SELECT mgr.name AS manager_name, proj.name AS project_name, ...
        FROM projects proj
        JOIN employees mgr ON mgr.id = proj.manager_id
        WHERE proj.name LIKE %(project_name)s
    """
    schema = _load_schema()
    triggers = (
        schema.get("nodes", {})
              .get("Project", {})
              .get("manager_lookup_triggers", [])
    )
    if not triggers:
        return result

    matched = any(trigger in question_normalized for trigger in triggers)
    if not matched:
        return result

    # Only fire when a project_name filter exists (otherwise it's too broad)
    has_project_filter = "project_name" in result.filters

    new_filters = dict(result.filters)
    new_filters["project_manager_only"] = "true"

    logger.info(
        "post_process: project_manager_lookup detected — "
        "injecting project_manager_only filter (project_name present=%s)",
        has_project_filter,
    )

    return result.model_copy(update={
        "entities": ["Project"],
        "filters": new_filters,
        "question_type": "lookup",
    })


def _schema_aggregation_fallback(
    question_normalized: str,
) -> Optional[EntityExtractionResult]:
    """
    Schema-driven fallback when the LLM extractor returns None.

    Reads every schema node that declares aggregation_defaults and checks
    whether the question contains that node's table name, display name, or
    entity key.  If it also contains an aggregation trigger word (rank, total,
    most, least, biggest, highest, lowest, compare, count, sum, average, avg)
    it returns an aggregation EntityExtractionResult for that node.

    Completely schema-driven: adding a new entity with aggregation_defaults
    to graph_schema.yaml automatically participates here — no Python changes.
    """
    _AGG_TRIGGERS = {
        "rank", "total", "most", "least", "biggest", "highest", "lowest",
        "compare", "count", "sum", "average", "avg", "budget", "salary",
        "payroll", "spending", "cost",
    }
    q = question_normalized
    has_agg_word = any(w in q for w in _AGG_TRIGGERS)
    if not has_agg_word:
        return None

    schema = _load_schema()
    for entity_name, node_data in schema.get("nodes", {}).items():
        if "aggregation_defaults" not in node_data:
            continue
        table   = node_data.get("table", "").lower()
        display = node_data.get("display_name", "").lower()
        entity_lower = entity_name.lower()
        # Match if question contains table name, display name, or entity key
        # (e.g. "departments", "department", or "Department")
        if table in q or display in q or entity_lower in q:
            logger.info(
                "Schema aggregation fallback: matched entity=%s for question='%s'",
                entity_name, q[:60],
            )
            return EntityExtractionResult(
                entities=[entity_name],
                filters={},
                projections=[],
                question_type="aggregation",
                extraction_method="llm_failed",
                latency_ms=0.0,
                confidence_score=ExtractionConfidence.MEDIUM,
                confidence_breakdown={"schema_aggregation_fallback": 0.7},
                escalation_reason="",
            )
    return None



def extract_entities(
    question: str,
    session_cache: Optional[List[Dict[str, Any]]] = None,
) -> EntityExtractionResult:
    start = time.time()
    question = question.strip()
    question_normalized = re.sub(r"\s+", " ", question.lower())

    embedding = _get_embedding(question_normalized) if session_cache is not None else None

    if embedding is not None and session_cache:
        cached = _session_cache_lookup(question_normalized, embedding, session_cache)
        if cached is not None:
            cached.latency_ms = round((time.time() - start) * 1000, 2)
            return cached

    result = _cached_llm_extract(question_normalized)

    latency_ms = round((time.time() - start) * 1000, 2)

    if result is None:
        logger.warning("LLM extraction failed for: '%s'", question)
        # Schema-driven aggregation fallback: handles questions like
        # "rank departments by salary" when the LLM times out or fails.
        fallback = _schema_aggregation_fallback(question_normalized)
        if fallback is not None:
            fallback.latency_ms = latency_ms
            return fallback
        return EntityExtractionResult(
            entities=[],
            filters={},
            projections=[],
            question_type="other",
            extraction_method="llm_failed",
            latency_ms=latency_ms,
            confidence_score=ExtractionConfidence.UNCERTAIN,
            confidence_breakdown={"llm_failed": 0.0},
            escalation_reason=(
                "I could not understand that question well enough to query the database. "
                "Try rephrasing — for example: 'Show me all employees in Engineering' "
                "or 'What is Alan Turing salary?'"
            ),
        )

    result = _post_process_having_count(result, question_normalized)
    result = _post_process_temporal(result, question_normalized)
    result = _post_process_delta(result, question_normalized)
    result = _post_process_project_manager_lookup(result, question_normalized)

    result.latency_ms = latency_ms

    if embedding is not None and session_cache is not None and result.question_type != "other" and result.entities:
        _session_cache_store(question_normalized, embedding, result, session_cache)

    logger.info(
        "Extraction: entities=%s filters=%s type=%s method=%s confidence=%.2f latency=%.0fms",
        result.entities,
        list(result.filters.keys()),
        result.question_type,
        result.extraction_method,
        result.confidence_score,
        latency_ms,
    )

    return result


_INJECTION_SIGNAL_REGISTRY: dict = {}
