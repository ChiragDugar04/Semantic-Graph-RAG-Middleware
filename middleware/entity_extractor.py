"""
entity_extractor.py — Schema-Driven LLM Extraction
====================================================
Architecture: single Ollama call, no regex phases, no spaCy.

The graph_schema.yaml is the ONLY source of domain knowledge:
  - Every valid entity name comes from schema["nodes"]
  - Every valid filter key is derived as <entity_lower>_<col_key>
  - Every allowed enum value comes from filterable_columns[col].enum_values
  - Every cross-table supplement key comes from filter_supplements
  - Unsupported query patterns come from query_limits.unsupported_patterns

Adding a new database, entity, column, or relationship requires
only a schema YAML edit — zero Python changes.

Flow:
    extract_entities(question)
        → _build_schema_context()     # compile schema → prompt section (cached)
        → _build_prompt(question)     # combine schema context + question
        → _call_ollama(prompt)        # single LLM call, temperature=0.0
        → _validate_and_build(raw)    # parse JSON, validate ALL keys against schema
        → EntityExtractionResult
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from middleware.models import EntityExtractionResult, ExtractionConfidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema loader — cached by file mtime so changes are picked up without restart
# ---------------------------------------------------------------------------
_schema_cache: Optional[Dict[str, Any]] = None
_schema_mtime: float = 0.0
_schema_context_cache: Optional[str] = None  # compiled prompt section


def _load_schema() -> Dict[str, Any]:
    global _schema_cache, _schema_mtime, _schema_context_cache
    schema_path = Path(__file__).parent.parent / "config" / "graph_schema.yaml"
    try:
        mtime = os.path.getmtime(schema_path)
    except OSError:
        mtime = 0.0

    if _schema_cache is None or mtime != _schema_mtime:
        with open(schema_path, "r") as f:
            _schema_cache = yaml.safe_load(f)
        _schema_mtime = mtime
        _schema_context_cache = None  # invalidate compiled prompt on schema change
        logger.info("Schema loaded/reloaded from %s", schema_path)

    return _schema_cache


def _get_valid_entities() -> List[str]:
    return list(_load_schema()["nodes"].keys())


def _load_model_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


# ---------------------------------------------------------------------------
# Schema-to-prompt compiler
# Reads graph_schema.yaml and produces a deterministic, self-contained
# prompt section that gives the LLM the complete closed-world view of the
# database. Cached until the schema file changes on disk.
# ---------------------------------------------------------------------------

def _build_schema_context() -> str:
    """
    Compile graph_schema.yaml into a compact, machine-readable prompt section.
    Result is cached until the schema mtime changes.

    Sections emitted (all derived from YAML, zero hardcoded strings):
      1. ENTITIES AND FILTER KEYS      — entity names + filter keys + enum constraints
      2. CROSS-TABLE KEY USAGE         — semantic_signals per supplement key (new)
      3. MULTI-ENTITY RULES            — multi_entity_triggers per node (new)
      4. DISAMBIGUATION                — disambiguation_note per filterable_column (new)
      5. RANKING RULE                  — no_filter_signals from query_limits (new)
      6. RETURN "other" IF CONTAINS    — linguistic_signals from unsupported_patterns (new)
    """
    global _schema_context_cache
    if _schema_context_cache is not None:
        return _schema_context_cache

    schema = _load_schema()
    nodes = schema.get("nodes", {})
    edges = schema.get("edges", [])
    query_limits = schema.get("query_limits", {})

    lines: List[str] = []

    # ── Section 1: Entities with their filter keys (one line per entity) ─────
    # Format: EntityName: key1, key2, key3=[v1,v2,v3], key4
    lines.append("ENTITIES AND FILTER KEYS:")
    for entity, node_data in nodes.items():
        filterable = node_data.get("filterable_columns", {})
        parts: List[str] = []
        for col_key, col_meta in filterable.items():
            fkey = f"{entity.lower()}_{col_key}"
            enum_vals   = col_meta.get("enum_values",  [])
            valid_vals  = col_meta.get("valid_values", [])
            if enum_vals:
                vals = ",".join(e.get("canonical", "") for e in enum_vals)
                parts.append(f"{fkey}=[{vals}]")
            elif valid_vals:
                vals = ",".join(v.get("canonical", "") for v in valid_vals)
                parts.append(f"{fkey}=[{vals}]")
            else:
                parts.append(fkey)
        lines.append(f"  {entity}: {', '.join(parts)}" if parts else f"  {entity}: (no filters)")

    lines.append("")

    # ── Section 2: Cross-table key usage with semantic signals ───────────────
    # For each filter_supplement that declares semantic_signals, emit a one-line
    # guidance entry so the LLM knows WHEN to use that key vs similar keys.
    # Reads: nodes[*].filter_supplements[*].semantic_signals
    #        edges[*].filter_supplements[*].semantic_signals
    signal_lines: List[str] = []
    seen_keys: set = set()
    for node_data in nodes.values():
        for supp_key, supp_meta in node_data.get("filter_supplements", {}).items():
            sigs = supp_meta.get("semantic_signals", [])
            if sigs and supp_key not in seen_keys:
                sigs_str = ", ".join(f'"{s}"' for s in sigs)
                signal_lines.append(f"  {supp_key:<20} → {sigs_str}")
                seen_keys.add(supp_key)
    for edge in edges:
        for supp_key, supp_meta in edge.get("filter_supplements", {}).items():
            sigs = supp_meta.get("semantic_signals", [])
            if sigs and supp_key not in seen_keys:
                sigs_str = ", ".join(f'"{s}"' for s in sigs)
                signal_lines.append(f"  {supp_key:<20} → {sigs_str}")
                seen_keys.add(supp_key)

    # Also list cross-table keys that have no signals (so LLM still knows they're valid)
    all_cross_keys: List[str] = []
    for node_data in nodes.values():
        for supp_key in node_data.get("filter_supplements", {}):
            if supp_key not in seen_keys and supp_key not in all_cross_keys:
                all_cross_keys.append(supp_key)
    for edge in edges:
        for supp_key in edge.get("filter_supplements", {}):
            if supp_key not in seen_keys and supp_key not in all_cross_keys:
                all_cross_keys.append(supp_key)

    if signal_lines:
        lines.append("CROSS-TABLE KEY USAGE — use the right key for these phrasings:")
        lines.extend(signal_lines)
        if all_cross_keys:
            lines.append(f"  also valid (no specific phrasing): {', '.join(all_cross_keys)}")
        lines.append("")
    elif all_cross_keys:
        lines.append(f"CROSS-TABLE KEYS (also valid): {', '.join(all_cross_keys)}")
        lines.append("")

    # ── Section 3: Multi-entity rules ────────────────────────────────────────
    # When question contains any trigger phrase for a node's multi_entity_triggers,
    # entities[] must include BOTH that node AND the co-entity (e.g. Employee+Department).
    # Reads: nodes[*].multi_entity_triggers
    # The co-entity pairing is inferred: any node with multi_entity_triggers implies
    # it is the secondary entity; Employee is always the primary co-entity.
    multi_rules: List[str] = []
    for entity, node_data in nodes.items():
        triggers = node_data.get("multi_entity_triggers", [])
        if triggers:
            trigger_str = ", ".join(f'"{t}"' for t in triggers)
            multi_rules.append(
                f"  If question contains [{trigger_str}] → entities must include Employee AND {entity}"
            )

    if multi_rules:
        lines.append("MULTI-ENTITY RULES (both entities required):")
        lines.extend(multi_rules)
        lines.append("")

    # ── Section 4: Disambiguation notes ──────────────────────────────────────
    # For filterable columns that have a disambiguation_note, emit a single line
    # clarifying when to use that key vs a similar one.
    # Reads: nodes[*].filterable_columns[*].disambiguation_note
    disambig_lines: List[str] = []
    for entity, node_data in nodes.items():
        for col_key, col_meta in node_data.get("filterable_columns", {}).items():
            note = col_meta.get("disambiguation_note", "")
            if note:
                fkey = f"{entity.lower()}_{col_key}"
                disambig_lines.append(f"  {fkey}: {note}")

    if disambig_lines:
        lines.append("DISAMBIGUATION:")
        lines.extend(disambig_lines)
        lines.append("")

    # ── Section 5: Ranking / superlative rule ─────────────────────────────────
    # For questions containing a superlative/ranking word about a numeric field,
    # emit filters:{} (empty) and question_type:"comparison". Never invent a value.
    # Reads: query_limits.no_filter_signals
    nfs = query_limits.get("no_filter_signals", {})
    nfs_words = nfs.get("trigger_words", [])
    nfs_fields = nfs.get("applies_to_fields", [])
    if nfs_words and nfs_fields:
        words_str  = ", ".join(f'"{w}"' for w in nfs_words)
        fields_str = ", ".join(nfs_fields)
        lines.append(
            f"RANKING RULE: If question contains [{words_str}] about [{fields_str}] "
            f"→ set filters:{{}} and question_type:\"comparison\". Never invent a filter value."
        )
        lines.append("")

    # ── Section 6: Linguistic signals for unsupported patterns ───────────────
    # Flat list of all linguistic_signals across all unsupported patterns.
    # The LLM sees these as exact substrings to match → return question_type:"other".
    # Reads: query_limits.unsupported_patterns[*].linguistic_signals
    all_signals: List[str] = []
    for pattern in query_limits.get("unsupported_patterns", []):
        all_signals.extend(pattern.get("linguistic_signals", []))

    if all_signals:
        signals_str = ", ".join(f'"{s}"' for s in all_signals)
        lines.append(f'RETURN "other" IF QUESTION CONTAINS ANY OF: {signals_str}')
        lines.append("")

    _schema_context_cache = "\n".join(lines)
    return _schema_context_cache


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str) -> str:
    schema_context = _build_schema_context()

    # Single-line example format saves ~200 tokens vs multi-line blocks.
    # Examples are kept because they calibrate question_type classification
    # and cross-table key usage — the model needs at least 3–4 to generalise.
    prompt = (
        'Parse the question into JSON: {"entities":[...],"filters":{},"question_type":"...","projections":[...]}\n\n'
        + schema_context
        + "QUESTION TYPES: lookup|list|comparison|aggregation|cross_entity|other\n"
        "RULES: only use entity names and filter keys listed above. "
        "For enum fields use only the listed values. "
        "Return other if off-topic or unsupported.\n\n"
        "EXAMPLES:\n"
        'Q:"Show me all pending orders"->{"entities":["Order"],"filters":{"order_status":"pending"},"question_type":"list","projections":[]}\n'
        'Q:"Alan Turing salary"->{"entities":["Employee"],"filters":{"employee_name":"Alan Turing"},"question_type":"lookup","projections":["salary"]}\n'
        'Q:"Don Draper is managing"->{"entities":["Employee","Project"],"filters":{"manager_name":"Don Draper"},"question_type":"list","projections":[]}\n'
        'Q:"Engineering employees active projects"->{"entities":["Employee","Project"],"filters":{"department_name":"Engineering","project_status":"active"},"question_type":"cross_entity","projections":[]}\n'
        'Q:"who reports to Michael Scott"->{"entities":["Employee"],"filters":{"reports_to_name":"Michael Scott"},"question_type":"lookup","projections":[]}\n'
        'Q:"Grace Hopper work involved in"->{"entities":["Employee","Project"],"filters":{"employee_name":"Grace Hopper"},"question_type":"list","projections":[]}\n'
        'Q:"products from OfficeWorld"->{"entities":["Product"],"filters":{"product_supplier":"OfficeWorld"},"question_type":"list","projections":[]}\n'
        'Q:"orders placed by people in Sales delivered"->{"entities":["Order"],"filters":{"order_status":"delivered","department_name":"Sales"},"question_type":"list","projections":[]}\n'
        'Q:"more than one project"->{"entities":[],"filters":{},"question_type":"other","projections":[]}\n'
        'Q:"weather"->{"entities":[],"filters":{},"question_type":"other","projections":[]}\n\n'
        f'Q:"{question}"->'
    )
    return prompt


# ---------------------------------------------------------------------------
# Ollama caller
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> Optional[str]:
    model_cfg = _load_model_config()
    try:
        resp = requests.post(
            f"{model_cfg['ollama_base_url']}/api/generate",
            json={
                "model": model_cfg["fast_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 150,   # JSON output is ~80 tokens; 150 is safe ceiling
                    "num_ctx":    2048,   # compressed prompt is ~450 tokens; 2048 is ample
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        logger.debug("Ollama raw response: %s", raw[:300])
        return raw
    except requests.RequestException as exc:
        logger.warning("Ollama call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Schema-driven validator — primary anti-hallucination layer
# ---------------------------------------------------------------------------

def _build_valid_filter_keys(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build complete valid filter key set from schema.
    Convention: <entity_lower>_<col_key> — identical to graph_query_builder.
    Includes cross-table supplement keys from filter_supplements on nodes/edges.
    """
    valid: Dict[str, Dict[str, Any]] = {}
    nodes = schema.get("nodes", {})
    edges = schema.get("edges", [])

    for entity, node_data in nodes.items():
        entity_lower = entity.lower()
        for col_key, col_meta in node_data.get("filterable_columns", {}).items():
            fkey = f"{entity_lower}_{col_key}"
            match = col_meta.get("match_type", "like")
            enum_vals = col_meta.get("enum_values", [])
            valid_vals = col_meta.get("valid_values", [])
            allowed: Optional[List[str]] = None
            if enum_vals:
                allowed = [e.get("canonical", "") for e in enum_vals]
            elif valid_vals:
                allowed = [v.get("canonical", "") for v in valid_vals]
            valid[fkey] = {"match": match, "allowed_values": allowed}

        for supp_key in node_data.get("filter_supplements", {}):
            valid[supp_key] = {"match": "subquery", "allowed_values": None}

    for edge in edges:
        for supp_key in edge.get("filter_supplements", {}):
            valid[supp_key] = {"match": "subquery", "allowed_values": None}

    return valid


def _validate_and_build(raw: Optional[str], question_normalized: str = "") -> Optional[EntityExtractionResult]:
    """
    Parse LLM JSON and validate every field against the schema.

    Validation:
      1. JSON parsing (3-attempt with repair)
      2. Entity names   — must be in schema nodes
      3. Filter keys    — must be in _build_valid_filter_keys()
      4. Enum values    — enforced for [exact] fields with allowed_values
      5. Ranking guard  — strips invented values for no_filter_signals fields (Fix 2)
      6. question_type  — must be in valid set

    Invalid items are stripped with a warning — partial results beat nothing.
    """
    if not raw:
        return None

    schema = _load_schema()
    valid_entities = set(schema["nodes"].keys())
    valid_types = {"lookup", "list", "comparison", "aggregation", "cross_entity", "other"}
    valid_filter_keys = _build_valid_filter_keys(schema)

    # JSON parsing
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("LLM response has no JSON object: %s", raw[:120])
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
        logger.warning("JSON parse failed: %s", json_str[:120])
        return None

    # Entity validation
    raw_entities = parsed.get("entities", [])
    entities = [e for e in raw_entities if e in valid_entities]
    invalid_ents = set(raw_entities) - valid_entities
    if invalid_ents:
        logger.warning("Stripping invalid entities: %s", invalid_ents)

    # Filter key + enum validation
    raw_filters = {k: str(v) for k, v in parsed.get("filters", {}).items() if v}
    filters: Dict[str, str] = {}
    for fkey, fval in raw_filters.items():
        if fkey not in valid_filter_keys:
            logger.warning("Stripping unknown filter key: '%s'", fkey)
            continue
        allowed = valid_filter_keys[fkey].get("allowed_values")
        if allowed is not None and fval not in allowed:
            logger.warning(
                "Stripping invalid enum value '%s' for key '%s' — allowed: %s",
                fval, fkey, allowed,
            )
            continue
        filters[fkey] = fval

    # ------------------------------------------------------------------ #
    # Fix 2 — Ranking guard: strip invented filter values for numeric     #
    # fields when the question is superlative/ranking.                    #
    #                                                                     #
    # Why validator not prompt: Q13/Q25 proved the RANKING RULE prompt    #
    # instruction is ignored by the 1.5B model — it always emits some    #
    # value for a field it has identified. This guard runs unconditionally#
    # after the LLM. It reads no_filter_signals from the schema:          #
    #   - applies_to_fields: keys that must never have an invented value  #
    #   - trigger_words:     words that signal a superlative question     #
    # A filter is stripped only when BOTH conditions hold:               #
    #   (a) question contains a ranking trigger word                      #
    #   (b) the field has no allowed_values (i.e. it's a free numeric)   #
    # This preserves legitimate string filters on the same fields.        #
    # Multi-DB safe: reads only from schema; zero hardcoded field names.  #
    # ------------------------------------------------------------------ #
    nfs = schema.get("query_limits", {}).get("no_filter_signals", {})
    nfs_fields   = set(nfs.get("applies_to_fields", []))
    nfs_words    = nfs.get("trigger_words", [])
    q_has_ranking = (
        bool(question_normalized)
        and any(w in question_normalized for w in nfs_words)
    )
    if q_has_ranking and nfs_fields:
        for fkey in list(filters.keys()):
            if fkey in nfs_fields:
                allowed = valid_filter_keys.get(fkey, {}).get("allowed_values")
                if allowed is None:
                    # No valid string values exist for this field — any string
                    # value the LLM emitted is invented. Strip it.
                    logger.info(
                        "Ranking guard: stripped filter %s='%s' "
                        "(superlative question, field has no valid string values)",
                        fkey, filters[fkey],
                    )
                    del filters[fkey]

    # question_type validation
    q_type = parsed.get("question_type", "lookup")
    if q_type not in valid_types:
        logger.warning("Invalid question_type '%s' — defaulting to lookup", q_type)
        q_type = "lookup"

    if len(set(entities)) >= 2 and q_type in ("lookup", "list"):
        q_type = "cross_entity"

    projections = [str(p) for p in parsed.get("projections", []) if p]

    # Confidence: binary — HIGH if we have usable output, LOW otherwise
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


# ---------------------------------------------------------------------------
# Extraction cache — keyed on normalized question text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cached_extract(question_normalized: str) -> Optional[EntityExtractionResult]:
    prompt = _build_prompt(question_normalized)
    raw = _call_ollama(prompt)
    return _validate_and_build(raw, question_normalized)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_entities(question: str) -> EntityExtractionResult:
    """
    Extract entities, filters, question_type, and projections from a
    natural language question using schema-driven LLM extraction.

    graph_schema.yaml is the sole source of domain knowledge.
    No regex, no spaCy, no hardcoded vocabulary.

    Pre-filter: checks query_limits.unsupported_patterns[*].linguistic_signals
    against the normalised question BEFORE the LLM call. Any substring match
    returns question_type="other" immediately — saves the full Ollama latency
    for T2 queries and ensures reliable decline even for small models.
    This loop reads only from the schema; zero hardcoded strings here.
    """
    start = time.time()
    question = question.strip()
    question_normalized = re.sub(r"\s+", " ", question.lower())

    # ── T2 pre-filter: schema linguistic_signals → instant "other" return ────
    # Runs before _cached_extract so T2 questions never burn an LLM call.
    schema = _load_schema()
    for pattern in schema.get("query_limits", {}).get("unsupported_patterns", []):
        for signal in pattern.get("linguistic_signals", []):
            if signal.lower() in question_normalized:
                latency_ms = round((time.time() - start) * 1000, 2)
                user_msg = pattern.get(
                    "user_message",
                    "This query pattern requires SQL capabilities (HAVING/subquery) "
                    "that this system does not support. Try a simpler phrasing.",
                )
                logger.info(
                    "T2 pre-filter matched signal '%s' in '%s' — returning other",
                    signal, question_normalized,
                )
                return EntityExtractionResult(
                    entities=[],
                    filters={},
                    projections=[],
                    question_type="other",
                    extraction_method="llm",
                    latency_ms=latency_ms,
                    confidence_score=ExtractionConfidence.HIGH,
                    confidence_breakdown={"t2_prefilter": 1.0},
                    escalation_reason=user_msg,
                )

    result = _cached_extract(question_normalized)

    latency_ms = round((time.time() - start) * 1000, 2)

    if result is None:
        logger.warning("LLM extraction failed for: '%s'", question)
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

    result.latency_ms = latency_ms

    logger.info(
        "Extraction: entities=%s filters=%s type=%s confidence=%.2f latency=%.0fms",
        result.entities,
        list(result.filters.keys()),
        result.question_type,
        result.confidence_score,
        latency_ms,
    )

    return result


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# pipeline.py imported _INJECTION_SIGNAL_REGISTRY from this module.
# The LLM now returns complete entity lists directly, making the signal
# loop in pipeline.py a no-op. Exporting an empty dict prevents ImportError
# during the transition period.
# ---------------------------------------------------------------------------
_INJECTION_SIGNAL_REGISTRY: dict = {}
