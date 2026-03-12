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

    entity_trigger_map = {
        "Employee": ["employee", "employees", "staff", "worker", "workers", "person", "people", "who", "member"],
        "Department": ["department", "departments", "dept", "team", "division", "group"],
        "Product": ["product", "products", "item", "items", "stock", "inventory", "unit", "units"],
        "Order": ["order", "orders", "purchase", "purchases", "bought", "ordered"],
        "Project": ["project", "projects", "initiative", "initiatives", "program", "programs"],
    }

    for entity, node_data in schema["nodes"].items():
        triggers = entity_trigger_map.get(entity, [entity.lower()])
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
    r"\b(managed by|manages|manager of|who manages|who leads|who runs|led by|run by|head of|in charge of)\b",
    re.I,
)

_ASSIGNMENT_PATTERNS = re.compile(
    r"\b(assigned to|working on|works on|on the|member of|part of|involved in|participating in)\b",
    re.I,
)

_STATUS_MAP = {
    "pending": "pending",
    "processing": "processing",
    "shipped": "shipped",
    "delivered": "delivered",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "active": "active",
    "completed": "completed",
    "complete": "completed",
    "planning": "planning",
    "planned": "planning",
    "on hold": "on_hold",
    "on_hold": "on_hold",
    "paused": "on_hold",
}

_CATEGORY_MAP = {
    "electronics": "Electronics",
    "electronic": "Electronics",
    "furniture": "Furniture",
    "office supply": "Office Supply",
    "office supplies": "Office Supply",
    "supply": "Office Supply",
    "supplies": "Office Supply",
}

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


def _extract_filter_value_for_entity(
    question: str,
    entity: str,
) -> Optional[str]:
    schema = _load_schema()
    node_data = schema["nodes"].get(entity, {})
    filterable = node_data.get("filterable_columns", {})

    name_patterns = [
        r"\b(?:in|for|of|about|called|named|the)\s+([A-Z][a-zA-Z\s&\-]+?)(?:\s+(?:department|dept|team|project|product|employee|staff))?\b",
        r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:department|dept|team|project|product)",
    ]

    if "name" in filterable:
        for pattern in name_patterns:
            match = re.search(pattern, question)
            if match:
                value = match.group(1).strip()
                if len(value) > 1 and value.lower() not in {
                    "all", "every", "any", "the", "a", "an", "list", "show",
                    "what", "who", "which", "how", "where", "when", "why"
                }:
                    return value

    return None


def _extract_named_values(question: str) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}

    for status_word, status_val in _STATUS_MAP.items():
        if re.search(rf"\b{re.escape(status_word)}\b", question, re.I):
            if any(w in question.lower() for w in ["order", "orders", "purchase"]):
                filters["order_status"] = status_val
            elif any(w in question.lower() for w in ["project", "projects", "initiative"]):
                filters["project_status"] = status_val
            break

    for cat_word, cat_val in _CATEGORY_MAP.items():
        if re.search(rf"\b{re.escape(cat_word)}\b", question, re.I):
            filters["category"] = cat_val
            break

    manager_match = _MANAGER_PATTERNS.search(question)
    if manager_match:
        after_manager = question[manager_match.end():].strip()
        name_match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", after_manager)
        if name_match:
            filters["manager_name"] = name_match.group(1).strip()

    name_in_dept = re.search(
        r"\b(?:in|from|of|within)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+(?:department|dept|team|division))?\s*(?:$|\?|,)",
        question, re.I
    )
    if name_in_dept:
        candidate = name_in_dept.group(1).strip()
        if candidate.lower() not in {"all", "every", "each", "the"}:
            if not any(w in candidate.lower() for w in ["project", "order", "product"]):
                filters["department_name"] = candidate

    project_name_match = re.search(
        r"\b(?:project|initiative|program)\s+(?:called\s+|named\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+project)?\s*(?:$|\?|,|\.|and)",
        question, re.I
    )
    if project_name_match:
        candidate = project_name_match.group(1).strip()
        if len(candidate) > 2:
            filters["project_name"] = candidate

    product_name_match = re.search(
        r"\b(?:product\s+)?([A-Z][a-zA-Z\s]+\s+(?:Pro|HD|Hub|Desk|Chair|Keyboard|Monitor|Mouse|Lamp|Pack|Paper|Whiteboard)[\w\s]*)\b",
        question
    )
    if product_name_match:
        filters["product_name"] = product_name_match.group(1).strip()

    person_name_match = re.search(
        r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\'s)?\s+(?:salary|pay|role|email|department|budget|project|assigned)",
        question
    )
    if person_name_match:
        filters["employee_name"] = person_name_match.group(1).strip()

    if not filters.get("employee_name"):
        whose_match = re.search(
            r"\bwhat\s+(?:is|are)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\'s)?\s+\w+",
            question
        )
        if whose_match:
            candidate = whose_match.group(1).strip()
            schema_entities = _get_valid_entities()
            if candidate not in schema_entities:
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

    priority_order = ["Employee", "Department", "Product", "Order", "Project"]
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

    entities = _detect_entities_from_keywords(question)
    if not entities:
        return None

    filters = _extract_named_values(question)
    projections = _extract_projections(question)
    question_type, type_resolved = _detect_question_type(question)

    if len(entities) >= 2:
        question_type = "cross_entity" if question_type == "lookup" else question_type

    has_clear_structure = bool(entities and (filters or projections))
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
