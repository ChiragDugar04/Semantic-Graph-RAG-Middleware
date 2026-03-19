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

# ---------------------------------------------------------------------------
# Step 4 — Schema-driven supplement→entity injection map
#
# Built once at module startup by reading every edge's filter_supplements.
# Any supplement key that carries supplement_injects_entity: <EntityName>
# in graph_schema.yaml is registered here.  The Python loop in
# _expand_entities_from_filters() reads this map — no entity names or
# filter key strings are hardcoded in Python.
#
# To change which supplement injects which entity: edit graph_schema.yaml.
# Zero Python changes required.
# ---------------------------------------------------------------------------
_SUPPLEMENT_ENTITY_MAP: dict = {}
for _edge in _graph._edges:
    for _supp_key, _supp_meta in _edge.get("filter_supplements", {}).items():
        _inject = _supp_meta.get("supplement_injects_entity")
        if _inject:
            _SUPPLEMENT_ENTITY_MAP[_supp_key] = _inject

logger.debug("_SUPPLEMENT_ENTITY_MAP built at startup: %s", _SUPPLEMENT_ENTITY_MAP)


def run_pipeline(question: str) -> MiddlewareTrace:
    pipeline_start = time.time()
    trace = MiddlewareTrace(user_question=question)
    trace.semantic_mode = True

    logger.info("Pipeline start: '%s'", question)

    extraction = extract_entities(question)

    # Fix 5: Fuzzy name resolution — correct truncated/misspelled names
    # before any pipeline logic uses the filter values. This runs after both
    # the regex fast-path and the LLM path, so it covers all extraction methods.
    # e.g. "Leslie Knop" → "Leslie Knope", "Draper" → "Don Draper"
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

    # T2: Unsupported query pattern — return helpful guidance instead of wrong SQL.
    # The extractor sets question_type="other" with a populated escalation_reason
    # for queries that need HAVING/subquery/temporal capabilities we don't support.
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

    # D5: Aggregation → lookup downgrade.
    # "How many units of Webcam HD" / "total budget for Don Draper's projects"
    # fire aggregation from "how many" / "total" keywords, but the user
    # named a specific entity — they want its attributes, not a COUNT(*).
    # Downgrade to lookup so _build_select returns real columns, not COUNT(*).
    #
    # T2-B: _NAMED_FILTERS is built dynamically from graph_schema.yaml.
    # Any filterable_column with is_named_scope: true contributes its filter
    # key (<entity_lower>_<col_key>) to the set. Adding a new named-scope
    # filter to the schema automatically includes it here — no Python change.
    # Non-schema supplement keys (manager_name, reports_to_name) are added
    # via the supplement map since they have no filterable_columns entry.
    _NAMED_FILTERS: set = set()
    for _ns_entity, _ns_node in _graph._nodes.items():
        _ns_entity_lower = _ns_entity.lower()
        for _ns_col_key, _ns_col_meta in _ns_node.get("filterable_columns", {}).items():
            if _ns_col_meta.get("is_named_scope"):
                _NAMED_FILTERS.add(f"{_ns_entity_lower}_{_ns_col_key}")
    # Supplement keys that represent named-scope references (non-schema column keys)
    _NAMED_FILTERS.update({"manager_name", "reports_to_name"})

    if (extraction.question_type == "aggregation"
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

    # Filter-aware path extension MUST run before the single-entity short-circuit.
    # Order and Department queries start with one entity but need neighbour nodes
    # injected (Product+Employee for Order, Employee for Department) so the graph
    # traversal builds the correct JOIN chain.
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
    """
    Filter-Aware Path Extension.

    If a filter references an entity that is not yet in the detected entity
    list, automatically inject that entity so the graph traversal builds the
    correct JOIN path.

    Generic injection (no Python change needed for new entities/filters):
    - _SUPPLEMENT_ENTITY_MAP loop: edge-level supplement keys that declare
      supplement_injects_entity inject that entity when the key is in filters.
    - Supplement-owner loop: node-level supplement keys with owner_entity,
      excludes_entities, and conditional_exclude enforce org-chart routing.
    - Filterable-column owner loop: filterable columns with owner_entity and
      insert_before_entity handle positional injection (e.g. employee_name
      before Project for Employee→Project JOIN direction).
    - T2-A generic loop: any filter key matching <entity_lower>_<col_key>
      injects that entity if absent (appended at end).

    Structural rules (explicit Python — encode graph topology constraints):
    - Department positional injection after Employee
    - Order display expansion (inject Order successors from schema graph)
    - Department display expansion (inject Employee before/after Department)
    - Order+Department guard (no direct Order→Department edge)
    - multi_entity_triggers membership query injection
    """
    expanded = list(dict.fromkeys(entities))   # deduplicate, preserve order
    entity_set = set(expanded)

    # ------------------------------------------------------------------ #
    # Fix B — Possessive filter correction                                 #
    # "Leslie Knope's team" → LLM may emit department_name="Leslie Knope's Team"
    # Detect by matching Title-Case name + possessive + trigger_word.
    # Correction: rewrite to reports_to_name, remove department_name.
    # Zero hardcoding: trigger words come from Department.trigger_words in schema.
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Schema-driven multi_entity_triggers injection                        #
    # When the LLM returns only ["Employee"] but the question contains a  #
    # phrase signalling a membership query ("which department", "belong to")
    # inject the secondary entity unconditionally.                         #
    # Zero hardcoding: reads multi_entity_triggers from _graph._nodes.    #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 4 — _SUPPLEMENT_ENTITY_MAP injection                           #
    # Replaces the hardcoded "manager_name → inject Project" block.       #
    # Any edge supplement that declares supplement_injects_entity in YAML  #
    # is registered at startup in _SUPPLEMENT_ENTITY_MAP.                  #
    # When that supplement key is present in filters AND the target entity #
    # is not yet in the path AND Employee is already present (supplement   #
    # keys on Employee→Project edge only fire on Employee paths), inject.  #
    # ------------------------------------------------------------------ #
    for supp_key, inject_entity in _SUPPLEMENT_ENTITY_MAP.items():
        if supp_key in filters and inject_entity not in entity_set:
            if "Employee" in entity_set:
                expanded.append(inject_entity)
                entity_set.add(inject_entity)
                logger.info(
                    "Path extension: injected %s via _SUPPLEMENT_ENTITY_MAP "
                    "(filter key '%s')", inject_entity, supp_key,
                )

    # ------------------------------------------------------------------ #
    # T2-A: Generic filter-to-entity injection loop                       #
    # Derives entity ownership from filter key prefix using the schema    #
    # node list. For any key "entity_lower_colname" where entity_lower    #
    # matches a known node, inject that entity if it is not yet in path.  #
    # ------------------------------------------------------------------ #
    _schema_node_names = list(_graph._schema_node_names)
    _entity_lower_map = {e.lower(): e for e in _schema_node_names}

    for filter_key in filters:
        matched_entity = None
        for entity_lower, entity in _entity_lower_map.items():
            if filter_key.startswith(entity_lower + "_") or filter_key == entity_lower:
                matched_entity = entity
                break
        if matched_entity is None:
            continue
        if matched_entity in entity_set:
            continue
        # Positional-injection guard: if this filter key has insert_before_entity
        # declared in the schema, skip the generic append here — the owner loop
        # below handles positional insertion correctly.
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

    # ------------------------------------------------------------------ #
    # Step 5 — Filterable-column owner/positional injection loop          #
    # Replaces the hardcoded N2 rule:                                     #
    #   "employee_name + Project → insert Employee BEFORE Project"        #
    # Any filterable column with owner_entity + insert_before_entity in   #
    # the schema participates here.                                        #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 5 — Supplement-owner loop                                      #
    # Replaces the hardcoded N4 rule:                                     #
    #   "reports_to_name → ensure Employee, remove Project unless          #
    #    manager_name also present"                                        #
    # Reads owner_entity, excludes_entities, and conditional_exclude from  #
    # each node's filter_supplements in graph_schema.yaml.                 #
    # ------------------------------------------------------------------ #
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
                # conditional_exclude: {absent_filter: <key>}
                # Only remove excl when the named filter is ABSENT from filters
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

    # --- Department injection (Semantic — positional: after Employee) ---
    # P6.1: Guard — skip if Order is already in path.
    if "department_name" in filters and "Department" not in entity_set and "Order" not in entity_set:
        if "Employee" in entity_set:
            idx = expanded.index("Employee")
            expanded.insert(idx + 1, "Department")
            entity_set.add("Department")
            logger.info(
                "Path extension: injected Department (department_name filter present)"
            )

    # --- Order display expansion (P1-B) ---
    # When Order is the anchor entity it owns two outgoing schema edges:
    #   Order -[placed_by]->  Employee  (o.employee_id = e.id)
    #   Order -[contains]->   Product   (o.product_id  = p.id)
    # Neighbours are derived from the live graph — not hardcoded strings.
    if "Order" in entity_set:
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

    # --- Department display expansion (P1-C / D7 / R7) ---
    # When Department is the sole entity, inject Employee for correct JOIN direction.
    # list/comparison/cross_entity/lookup: Employee BEFORE Department (works_in edge →
    # returns all employees in dept).
    # aggregation: Employee AFTER Department (managed_by edge → GROUP BY dept).
    if "Department" in entity_set and "Employee" not in entity_set:
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
    """
    Return entities ordered by topological sort of the induced schema subgraph.

    This is the schema-driven replacement for the hardcoded priority list
    ["Employee", "Department", "Project", "Order", "Product"].

    For DAG subgraphs (e.g. Order->Employee, Order->Product): topological sort
    gives Order first — which is correct since Order is the anchor.

    For subgraphs with cycles from bidirectional edges (Employee<->Department,
    Employee<->Project): NetworkXUnfeasible is caught and we fall back to
    sorting by out_degree descending with schema declaration order as tiebreaker.
    The schema declares nodes in Employee, Department, Product, Order, Project
    order — the same precedence as the old hardcoded list.

    Adding a new entity to the schema automatically gets the correct sort
    position from graph topology — no Python change needed.
    """
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
    """
    Enforce a stable, join-friendly ordering of entities so the graph always
    traverses in a direction that produces valid SQL joins.

    Step 6: both the default priority list and the secondary list in the anchor
    override block are replaced with _topo_sort_entities().  The sort is driven
    entirely by the schema graph topology — adding a new entity automatically
    gets the correct join order with no Python changes.

    Schema-driven anchor override: when a single entity was originally detected
    and others were injected by display expansion (e.g. Order injects Employee
    and Product), the originally-detected node must anchor the JOIN.  Gate
    conditions are purely graph-structure-driven — no entity names hardcoded.
    """
    if not entities:
        return entities

    if len(entity_set) == 1:
        return list(entity_set)

    # Schema-driven anchor override — only when expansion injected extra nodes.
    # Gate: original list had exactly one entity (the query subject).
    # D7: suppress for comparison queries — we intentionally inserted Employee
    # first so the works_in JOIN returns all employees, not just the manager.
    #
    # P1 FIX — Bidirectional-edge guard:
    # The override must NOT fire when the candidate<->injected relationship is
    # bidirectional.  When both directions exist, _expand_entities_from_filters
    # deliberately orders them to select the correct JOIN direction.
    # The override SHOULD fire for unidirectional relationships (e.g. Order->Employee,
    # Order->Product) where Order is the only valid anchor.
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

    # Step 6: default priority list replaced with _topo_sort_entities().
    # Produces identical ordering to ["Employee","Department","Project","Order","Product"]
    # for all current entity combinations, and correct ordering for any new entities.
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
