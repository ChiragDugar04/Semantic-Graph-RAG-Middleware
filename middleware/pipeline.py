from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Tuple

import networkx as nx

from middleware.models import (
    EntityExtractionResult,
    ExtractedParameters,
    ExtractionConfidence,
    GraphTraversal,
    IntentClassification,
    MiddlewareTrace,
    QueryTemplate,
)
from middleware.entity_extractor import extract_entities
from middleware.semantic_graph import SemanticGraph, EntityNotFoundError, NoPathError, JoinStep
from middleware.graph_query_builder import GraphQueryBuilder, QueryBuildError
from middleware.query_executor import execute_query
from middleware.context_formatter import format_context
from middleware.answer_synthesizer import synthesize_answer
from middleware.fuzzy_resolver import resolve_filter_values

logger = logging.getLogger(__name__)

_graph = SemanticGraph()
_builder = GraphQueryBuilder(_graph)

_SUPPLEMENT_ENTITY_MAP: dict = {}
for _edge in _graph._edges:
    for _supp_key, _supp_meta in _edge.get("filter_supplements", {}).items():
        _inject = _supp_meta.get("supplement_injects_entity")
        if _inject:
            _SUPPLEMENT_ENTITY_MAP[_supp_key] = _inject

logger.debug("_SUPPLEMENT_ENTITY_MAP built at startup: %s", _SUPPLEMENT_ENTITY_MAP)


def run_pipeline(question: str, session_cache: Optional[List] = None) -> MiddlewareTrace:
    pipeline_start = time.time()
    trace = MiddlewareTrace(user_question=question)
    trace.semantic_mode = True

    logger.info("Pipeline start: '%s'", question)

    extraction = extract_entities(question, session_cache=session_cache)

    if extraction.filters:
        resolved_filters = resolve_filter_values(extraction.filters)
        if resolved_filters != extraction.filters:
            extraction = extraction.model_copy(update={"filters": resolved_filters})

    trace.entity_extraction = extraction
    trace.pipeline_stage_reached = "extraction"

    logger.info(
        "Extraction complete: entities=%s type=%s confidence=%.2f method=%s",
        extraction.entities,
        extraction.question_type,
        extraction.confidence_score,
        extraction.extraction_method,
    )

    if extraction.is_genuinely_off_topic:
        trace.final_answer = (
            "I can only answer questions about employees, departments, products, "
            "orders, and projects from the company database. "
            "Please ask something related to that data."
        )
        trace.pipeline_stage_reached = "complete"
        trace.total_latency_ms = round((time.time() - pipeline_start) * 1000, 2)
        _populate_legacy_fields(trace, extraction, None)
        logger.info("Off-topic question short-circuited at extraction stage")
        return trace

    if (extraction.question_type == "other"
            and extraction.escalation_reason
            and "unsupported_pattern" in extraction.confidence_breakdown):
        trace.final_answer = extraction.escalation_reason
        trace.pipeline_stage_reached = "complete"
        trace.total_latency_ms = round((time.time() - pipeline_start) * 1000, 2)
        _populate_legacy_fields(trace, extraction, None)
        logger.info("T2: unsupported pattern short-circuited")
        return trace

    if not extraction.is_trustworthy and not extraction.entities:
        logger.warning(
            "Extraction confidence %.2f below threshold with no entities — "
            "cannot build query. Reason: %s",
            extraction.confidence_score,
            extraction.escalation_reason,
        )
        trace.final_answer = (
            "I understood your question but could not identify what data to look for. "
            "Could you rephrase? For example: "
            "'Show me all employees in Engineering' or "
            "'What is Sarah Connor's salary?'"
        )
        trace.pipeline_stage_reached = "complete"
        trace.total_latency_ms = round((time.time() - pipeline_start) * 1000, 2)
        _populate_legacy_fields(trace, extraction, None)
        return trace

    _NAMED_FILTERS: set = set()
    for _ns_entity, _ns_node in _graph._nodes.items():
        _ns_entity_lower = _ns_entity.lower()
        for _ns_col_key, _ns_col_meta in _ns_node.get("filterable_columns", {}).items():
            if _ns_col_meta.get("is_named_scope"):
                _NAMED_FILTERS.add(f"{_ns_entity_lower}_{_ns_col_key}")
    # Supplement keys that represent named-scope references (non-schema column keys)
    _NAMED_FILTERS.update({"manager_name", "reports_to_name"})

    # D5: only downgrade aggregation — never touch the 3 new types.
    _NON_DOWNGRADEABLE = {"having_count", "temporal_filter", "computed_delta"}
    if (extraction.question_type == "aggregation"
            and extraction.question_type not in _NON_DOWNGRADEABLE
            and _NAMED_FILTERS & set(extraction.filters.keys())):
        logger.info(
            "D5: aggregation downgraded to lookup (named filter present: %s)",
            list(_NAMED_FILTERS & set(extraction.filters.keys())),
        )
        extraction = extraction.model_copy(update={"question_type": "lookup"})

    traversal, join_chain = _run_graph_traversal(extraction, question)
    trace.graph_traversal = traversal
    trace.pipeline_stage_reached = "traversal"

    _populate_legacy_fields(trace, extraction, traversal)

    if not extraction.entities or extraction.question_type == "other":
        template, query_params = _build_fallback_template()
    else:
        try:
            template, query_params = _builder.build_query(
                extraction=extraction,
                traversal=traversal,
                join_chain=join_chain,
                effective_entities=traversal.path_taken if traversal.path_taken else extraction.entities,
            )
        except QueryBuildError as qbe:
            logger.error("Query build failed: %s", qbe)
            template, query_params = _build_fallback_template()
        except Exception as exc:
            logger.error("Unexpected error building query: %s", exc, exc_info=True)
            template, query_params = _build_fallback_template()

    trace.pipeline_stage_reached = "query_built"

    exec_params = ExtractedParameters(
        params=query_params,
        missing_required=[],
        extraction_method=extraction.extraction_method,
        latency_ms=extraction.latency_ms,
    )

    trace.db_result = execute_query(
        template=template,
        parameters=exec_params,
        user_question=question,
    )
    trace.pipeline_stage_reached = "db"

    if trace.db_result.self_healing_triggered:
        trace.self_healing_triggered = True

    context = format_context(db_result=trace.db_result, template=template)

    trace.final_answer = synthesize_answer(
        user_question=question,
        context=context,
        db_result=trace.db_result,
        question_type=extraction.question_type,
        filters=extraction.filters,
    )
    trace.pipeline_stage_reached = "complete"

    trace.total_latency_ms = round((time.time() - pipeline_start) * 1000, 2)

    logger.info(
        "Pipeline complete: %.0fms total, stage=%s, rows=%d",
        trace.total_latency_ms,
        trace.pipeline_stage_reached,
        trace.db_result.row_count if trace.db_result else 0,
    )

    return trace


def _populate_legacy_fields(
    trace: MiddlewareTrace,
    extraction: EntityExtractionResult,
    traversal: Optional[GraphTraversal],
) -> None:
    path_str = "→".join(traversal.path_taken) if traversal and traversal.path_taken else "other"
    confidence_label = (
        "high" if extraction.confidence_score >= 0.85
        else "medium" if extraction.confidence_score >= ExtractionConfidence.TRUST_THRESHOLD
        else "low"
    )

    trace.intent = IntentClassification(
        intent_name=path_str,
        confidence=confidence_label,
        raw_llm_response=(
            f"entities={extraction.entities}, "
            f"type={extraction.question_type}, "
            f"score={extraction.confidence_score:.2f}"
        ),
        model_used=f"semantic_graph [{extraction.extraction_method}]",
        latency_ms=extraction.latency_ms,
    )

    trace.parameters = ExtractedParameters(
        params=extraction.filters,
        missing_required=[],
        extraction_method=extraction.extraction_method,
        latency_ms=extraction.latency_ms,
    )


def _run_graph_traversal(
    extraction: EntityExtractionResult,
    question: str = "",
) -> Tuple[GraphTraversal, List[JoinStep]]:
    start = time.time()
    entities = extraction.entities
    filters = extraction.filters

    if not entities:
        elapsed = round((time.time() - start) * 1000, 2)
        return GraphTraversal(
            path_taken=[],
            join_count=0,
            tables_involved=[],
            traversal_time_ms=elapsed,
            traversal_method="single_node",
            path_description="No entities detected",
        ), []

    unique_entities = list(dict.fromkeys(entities))
    unique_entities = _expand_entities_from_filters(
        unique_entities, filters, q_type=extraction.question_type, question=question,
    )

    if len(set(unique_entities)) == 1:
        entity = unique_entities[0]
        table = _graph.get_table_name(entity)
        elapsed = round((time.time() - start) * 1000, 2)
        return GraphTraversal(
            path_taken=[entity],
            join_count=0,
            tables_involved=[table],
            traversal_time_ms=elapsed,
            traversal_method="single_node",
            path_description=entity,
        ), []

    try:
        path = _graph.find_multi_path(unique_entities)
        join_chain = _graph.get_join_chain(path)

        tables = []
        for node in path:
            try:
                tables.append(_graph.get_table_name(node))
            except Exception:
                pass

        join_count = len(join_chain)
        method = (
            "single_node" if join_count == 0
            else "two_hop" if join_count == 1
            else "multi_hop"
        )

        elapsed = round((time.time() - start) * 1000, 2)
        path_desc = _graph.describe_path(path)

        logger.info("Graph traversal: path=%s joins=%d method=%s", path, join_count, method)

        return GraphTraversal(
            path_taken=path,
            join_count=join_count,
            tables_involved=tables,
            traversal_time_ms=elapsed,
            traversal_method=method,
            path_description=path_desc,
        ), join_chain

    except (EntityNotFoundError, NoPathError) as exc:
        logger.warning("Graph traversal failed: %s — falling back to anchor entity only", exc)
        entity = unique_entities[0]
        table = _graph.get_table_name(entity)
        elapsed = round((time.time() - start) * 1000, 2)

        return GraphTraversal(
            path_taken=[entity],
            join_count=0,
            tables_involved=[table],
            traversal_time_ms=elapsed,
            traversal_method="single_node",
            path_description=f"{entity} (path error: {str(exc)[:60]})",
        ), []


def _expand_entities_from_filters(
    entities: List[str],
    filters: dict,
    q_type: str = "lookup",
    question: str = "",
) -> List[str]:
    expanded = list(dict.fromkeys(entities))   # deduplicate, preserve order
    entity_set = set(expanded)

    if "department_name" in filters:
        dept_val = filters["department_name"]
        dept_node = _graph._nodes.get("Department", {})
        dept_trigger_words = {t.lower() for t in dept_node.get("trigger_words", [])}
        _possessive_re = re.compile(
            r"^([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)'s?\s*(\w+)",
            re.UNICODE,
        )
        m = _possessive_re.match(dept_val.strip())
        if m:
            candidate_name = m.group(1).strip()
            trailing_word  = m.group(2).lower()
            if trailing_word in dept_trigger_words:
                filters = dict(filters)
                filters["reports_to_name"] = candidate_name
                del filters["department_name"]
                expanded = [e for e in expanded if e != "Department"]
                entity_set.discard("Department")
                logger.info(
                    "Possessive correction: department_name='%s' -> "
                    "reports_to_name='%s' (trailing word '%s' in dept trigger_words)",
                    dept_val, candidate_name, trailing_word,
                )

    if question:
        q_norm = re.sub(r"\s+", " ", question.lower())
        for entity, node_data in _graph._nodes.items():
            if entity not in entity_set:
                for trigger in node_data.get("multi_entity_triggers", []):
                    if trigger in q_norm:
                        expanded.append(entity)
                        entity_set.add(entity)
                        logger.info(
                            "Path extension: injected %s via multi_entity_trigger '%s'",
                            entity, trigger,
                        )
                        break

    for supp_key, inject_entity in _SUPPLEMENT_ENTITY_MAP.items():
        if supp_key in filters and inject_entity not in entity_set:
            if "Employee" in entity_set:
                expanded.append(inject_entity)
                entity_set.add(inject_entity)
                logger.info(
                    "Path extension: injected %s via _SUPPLEMENT_ENTITY_MAP "
                    "(filter key '%s')", inject_entity, supp_key,
                )

    _schema_node_names = list(_graph._schema_node_names)
    _entity_lower_map = {e.lower(): e for e in _schema_node_names}

    _SYNTHETIC_KEY_PREFIXES = ("delta_", "having_threshold")
    _SYNTHETIC_KEY_SUFFIXES = ("_direction", "_column_sql", "_after_op", "_before_op")

    for filter_key in filters:
        if any(filter_key.startswith(p) for p in _SYNTHETIC_KEY_PREFIXES):
            continue
        if any(filter_key.endswith(s) for s in _SYNTHETIC_KEY_SUFFIXES):
            continue
        matched_entity = None
        for entity_lower, entity in _entity_lower_map.items():
            if filter_key.startswith(entity_lower + "_") or filter_key == entity_lower:
                matched_entity = entity
                break
        if matched_entity is None:
            continue
        if matched_entity in entity_set:
            continue
        col_key = filter_key[len(matched_entity.lower()) + 1:]  # strip "entity_" prefix
        col_meta = _graph._nodes.get(matched_entity, {}).get("filterable_columns", {}).get(col_key, {})
        if col_meta.get("insert_before_entity") and col_meta["insert_before_entity"] in entity_set:
            continue
        # T1.6: Department + Order path guard.
        if matched_entity == "Department" and "Order" in entity_set:
            logger.info(
                "Path extension: skipping Department injection "
                "(Order in path — department_name will be resolved via Employee subquery)"
            )
            continue
        expanded.append(matched_entity)
        entity_set.add(matched_entity)
        logger.info(
            "Path extension: injected %s generically (filter key %s)",
            matched_entity, filter_key,
        )

    for filter_key in filters:
        # Derive entity and column key from the filter key string
        matched_entity = None
        col_key = None
        for entity_lower, entity in _entity_lower_map.items():
            if filter_key.startswith(entity_lower + "_"):
                matched_entity = entity
                col_key = filter_key[len(entity_lower) + 1:]
                break
        if matched_entity is None or col_key is None:
            continue
        col_meta = _graph._nodes.get(matched_entity, {}).get("filterable_columns", {}).get(col_key, {})
        owner = col_meta.get("owner_entity")
        insert_before = col_meta.get("insert_before_entity")
        if not owner or not insert_before:
            continue
        # Fire when: owner absent from path AND insert_before IS in path
        if owner not in entity_set and insert_before in entity_set:
            idx = expanded.index(insert_before)
            expanded.insert(idx, owner)
            entity_set.add(owner)
            logger.info(
                "Path extension: injected %s before %s "
                "(filter key '%s' owner_entity + insert_before_entity rule)",
                owner, insert_before, filter_key,
            )

    for node_entity, node_data in _graph._nodes.items():
        for supp_key, supp_meta in node_data.get("filter_supplements", {}).items():
            if supp_key not in filters:
                continue
            owner = supp_meta.get("owner_entity")
            excludes = supp_meta.get("excludes_entities", [])
            cond_excl = supp_meta.get("conditional_exclude", {})

            # Ensure owner entity is in path
            if owner and owner not in entity_set:
                expanded.append(owner)
                entity_set.add(owner)
                logger.info(
                    "Path extension: injected %s (supplement owner for '%s')",
                    owner, supp_key,
                )

            # Remove excluded entities, subject to conditional_exclude
            for excl in excludes:
                if excl not in entity_set:
                    continue
                absent_filter = cond_excl.get("absent_filter")
                if absent_filter and absent_filter in filters:
                    # The condition for exclusion is NOT met — keep the entity
                    logger.debug(
                        "Supplement owner: NOT removing %s for '%s' "
                        "(conditional_exclude: absent_filter '%s' is present)",
                        excl, supp_key, absent_filter,
                    )
                    continue
                expanded = [e for e in expanded if e != excl]
                entity_set.discard(excl)
                logger.info(
                    "Path extension: removed %s (supplement '%s' excludes_entities rule)",
                    excl, supp_key,
                )

    _NO_EXPANSION_TYPES = {"group_rank", "having_count", "temporal_filter", "computed_delta"}
    _employee_has_join_partner = any(
        e for e in entity_set
        if e not in ("Employee", "Department") and _graph._graph.has_edge("Employee", e)
    )
    if (q_type not in _NO_EXPANSION_TYPES
            and "department_name" in filters
            and "Department" not in entity_set
            and "Order" not in entity_set
            and not _employee_has_join_partner):
        if "Employee" in entity_set:
            idx = expanded.index("Employee")
            expanded.insert(idx + 1, "Department")
            entity_set.add("Department")
            logger.info(
                "Path extension: injected Department (department_name filter present)"
            )

    if "Order" in entity_set and q_type not in _NO_EXPANSION_TYPES:
        # Order+Department co-presence guard: no Order->Department edge exists.
        # Remove Department; department_name is resolved via Employee's subquery.
        if "Department" in entity_set:
            has_direct_order_dept_edge = _graph._graph.has_edge("Order", "Department")
            if not has_direct_order_dept_edge:
                expanded = [e for e in expanded if e != "Department"]
                entity_set.discard("Department")
                logger.info(
                    "Path extension: removed Department "
                    "(Order in path — no Order->Department edge; "
                    "department_name resolved via Employee subquery)"
                )

        schema_order_neighbours = set(_graph._graph.successors("Order"))
        for neighbour in sorted(schema_order_neighbours):
            if neighbour not in entity_set:
                expanded.append(neighbour)
                entity_set.add(neighbour)
                logger.info(
                    "Path extension: injected %s "
                    "(Order display expansion via schema edge)", neighbour
                )

    if "Department" in entity_set and "Employee" not in entity_set and q_type not in _NO_EXPANSION_TYPES:
        if q_type in ("comparison", "list", "cross_entity", "lookup"):
            idx = expanded.index("Department")
            expanded.insert(idx, "Employee")
            entity_set.add("Employee")
            logger.info(
                "Path extension: injected Employee before Department "
                "(%s query — needs Employee->Department path for all staff)", q_type
            )
        else:
            expanded.append("Employee")
            entity_set.add("Employee")
            logger.info(
                "Path extension: injected Employee after Department "
                "(aggregation — managed_by join for dept summary)"
            )

    return _canonical_entity_order(expanded, set(expanded), filters, original_entities=entities, q_type=q_type)


def _topo_sort_entities(entity_set: set) -> List[str]:

    subgraph = _graph._graph.subgraph(entity_set)
    schema_order = list(_graph._schema_node_names)
    try:
        return list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        # Cycle in subgraph (bidirectional edges) — fall back to degree sort
        return sorted(
            entity_set,
            key=lambda n: (
                -subgraph.out_degree(n),
                schema_order.index(n) if n in schema_order else len(schema_order),
            ),
        )


def _canonical_entity_order(
    entities: List[str],
    entity_set: set,
    filters: dict,
    original_entities: Optional[List[str]] = None,
    q_type: str = "lookup",
) -> List[str]:
    
    if not entities:
        return entities

    if len(entity_set) == 1:
        return list(entity_set)

    if original_entities and len(set(original_entities)) == 1 and q_type != "comparison":
        candidate = original_entities[0]
        if candidate in entity_set:
            others = entity_set - {candidate}
            if others and all(_graph._graph.has_edge(candidate, other) for other in others):
                has_any_reverse_edge = any(
                    _graph._graph.has_edge(other, candidate) for other in others
                )
                if has_any_reverse_edge:
                    logger.debug(
                        "Anchor override suppressed for %s: reverse edges exist from injected "
                        "nodes — respecting _expand_entities_from_filters ordering",
                        candidate,
                    )
                else:
                    # Unidirectional — candidate is the only valid anchor (e.g. Order).
                    # Step 6: secondary list replaced with _topo_sort_entities().
                    ordered: List[str] = [candidate]
                    secondary = _topo_sort_entities(entity_set - {candidate})
                    for node in secondary:
                        if node != candidate:
                            ordered.append(node)
                    for node in entities:
                        if node not in ordered:
                            ordered.append(node)
                    logger.debug(
                        "Anchor override: %s leads (sole original, unidirectional edges only)",
                        candidate,
                    )
                    return ordered

    ordered = _topo_sort_entities(entity_set)

    # Append anything the subgraph missed (isolated nodes or schema gaps)
    for node in entities:
        if node not in ordered:
            ordered.append(node)

    return ordered


def _build_fallback_template() -> Tuple[QueryTemplate, dict]:
    return QueryTemplate(
        intent_name="other",
        description="No matching database query found",
        sql_template="",
        required_params=[],
        optional_params=[],
        result_description="",
    ), {}
