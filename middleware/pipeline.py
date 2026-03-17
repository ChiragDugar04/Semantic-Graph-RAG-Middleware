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

logger = logging.getLogger(__name__)

_graph = SemanticGraph()
_builder = GraphQueryBuilder(_graph)


def run_pipeline(question: str) -> MiddlewareTrace:
    pipeline_start = time.time()
    trace = MiddlewareTrace(user_question=question)
    trace.semantic_mode = True

    logger.info("Pipeline start: '%s'", question)

    extraction = extract_entities(question)
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
    # T2-B: _NAMED_FILTERS is now built dynamically from graph_schema.yaml.
    # Any filterable_column with is_named_scope: true contributes its filter
    # key (<entity_lower>_<col_key>) to the set. Adding a new named-scope
    # filter to the schema (e.g. supplier_country) automatically includes it
    # here — no Python change required.
    # Non-schema supplement keys (manager_name, reports_to_name) are added
    # explicitly — they have no filterable_columns entry.
    # Access schema via the module-level _graph instance (already loaded,
    # no new import or cache needed). _graph._nodes is identical to
    # _load_schema()["nodes"] — same YAML, same cache lifecycle.
    _NAMED_FILTERS: set = set()
    for _ns_entity, _ns_node in _graph._nodes.items():
        _ns_entity_lower = _ns_entity.lower()
        for _ns_col_key, _ns_col_meta in _ns_node.get("filterable_columns", {}).items():
            if _ns_col_meta.get("is_named_scope"):
                _NAMED_FILTERS.add(f"{_ns_entity_lower}_{_ns_col_key}")
    # Non-schema named-scope supplement keys
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

    T2-A: A generic loop now handles all filter keys that follow the
    <entity_lower>_<col_key> convention — these are injected automatically
    at the end of the path (lowest priority) without any Python rule.
    Semantic rules that require specific insertion positions or involve
    non-schema filter keys (manager_name, reports_to_name) are kept below.

    Refinement 1 — Signal-to-Entity Registry (T2.5):
    Each node in graph_schema.yaml can carry an injection_signals list of
    pattern names. If employee_name is in filters AND a listed signal matches
    the question AND the node is not yet in path → inject that node.
    The loop is fully generic: adding a new signal requires only a schema edit
    and a registry entry in entity_extractor._INJECTION_SIGNAL_REGISTRY.

    Generic injection (this loop — no Python change needed for new entities):
    - Any filter key matching <entity_lower>_<col_key> where entity is in
      the schema → inject that entity if absent (appended at end)

    Semantic rules (explicit Python — ordering or non-schema keys):
    - department_name → inject Department at specific position after Employee
    - manager_name → inject Project (non-schema key, cannot be derived generically)
    - employee_name + Project → inject Employee BEFORE Project (N2 position rule)
    - reports_to_name → ensure Employee, remove Project if injected (N4 org-chart)
    - Order display expansion → inject Order successors from schema graph
    - Department display expansion → inject Employee before/after Department
    """
    expanded = list(dict.fromkeys(entities))   # deduplicate, preserve order
    entity_set = set(expanded)

    # ------------------------------------------------------------------ #
    # Fix B — Possessive filter correction                                 #
    # Problem: "Leslie Knope's team" → LLM emits department_name=         #
    # "Leslie Knop's Team" instead of reports_to_name="Leslie Knope".     #
    # The possessive "X's <team-word>" is an org-chart query, not a       #
    # department-name lookup.                                              #
    #                                                                     #
    # Detection: department_name value matches Title-Case name + "'s" +   #
    # a word that appears in Department.multi_entity_triggers (e.g."team")#
    #                                                                     #
    # Correction:                                                          #
    #  1. Extract person name (everything before "'s")                     #
    #  2. Set filters['reports_to_name'] = extracted_name                  #
    #  3. Delete filters['department_name']                                #
    #  4. Remove Department from entity_set (not needed for reports_to)    #
    #                                                                     #
    # Zero hardcoding: trigger words come from Department.multi_entity_   #
    # triggers in the schema. Name extraction uses a generic Title-Case   #
    # regex. Multi-DB safe: any schema node declaring multi_entity_        #
    # triggers with team-like words participates automatically.            #
    # ------------------------------------------------------------------ #
    if "department_name" in filters:
        dept_val = filters["department_name"]
        dept_node    = _graph._nodes.get("Department", {})
        # Use trigger_words (bare word list: "team", "dept", "division", etc.)
        # not multi_entity_triggers (phrase list: "which team", "belong to", etc.)
        # The trailing word after "'s" will be a single word like "team" or "group",
        # which matches trigger_words entries, not multi-word trigger phrases.
        dept_trigger_words = {t.lower() for t in dept_node.get("trigger_words", [])}
        # Pattern: one or more Title-Case words, then 's (with or without
        # the apostrophe), then a space and one more word.
        # Handles: "Leslie Knope's Team", "Leslie Knop's Team"
        _possessive_re = re.compile(
            r"^([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)'s?\s*(\w+)",
            re.UNICODE,
        )
        m = _possessive_re.match(dept_val.strip())
        if m:
            candidate_name = m.group(1).strip()
            trailing_word  = m.group(2).lower()
            if trailing_word in dept_trigger_words:
                # Confirmed possessive org-chart query — rewrite the filter
                filters = dict(filters)          # mutable copy (caller's dict unchanged)
                filters["reports_to_name"] = candidate_name
                del filters["department_name"]
                expanded = [e for e in expanded if e != "Department"]
                entity_set.discard("Department")
                logger.info(
                    "Possessive correction: department_name='%s' → "
                    "reports_to_name='%s' (trailing word '%s' in dept trigger_words)",
                    dept_val, candidate_name, trailing_word,
                )

    # ------------------------------------------------------------------ #
    # Fix 1 — Schema-driven multi_entity_triggers injection               #
    # When the LLM returns only ["Employee"] but the question contains a  #
    # phrase that signals a membership query (e.g. "which department",    #
    # "belong to"), inject the secondary entity unconditionally.          #
    #                                                                     #
    # Why pipeline not prompt: the prompt rule was present and still      #
    # failed for sentence structures where the person name is dominant    #
    # (Q05: "Which dept does Sarah Connor belong to?"). The pipeline      #
    # enforces this after the LLM regardless of sentence structure.       #
    #                                                                     #
    # Zero hardcoding: reads multi_entity_triggers from _graph._nodes.   #
    # Any node that declares multi_entity_triggers participates here.     #
    # Multi-DB safe: schema author declares triggers per entity in YAML.  #
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
                        break  # one trigger match is sufficient per entity

    # ------------------------------------------------------------------ #
    # T2-A: Generic filter-to-entity injection loop                       #
    # Derives entity ownership from filter key prefix using the schema    #
    # node list. For any key "entity_lower_colname" where entity_lower    #
    # matches a known node, inject that entity if it is not yet in path.  #
    # Entities are appended at the end — semantic rules below handle      #
    # cases that require specific insertion positions.                     #
    # ------------------------------------------------------------------ #
    _schema_node_names = list(_graph._schema_node_names)  # ordered list from schema
    _entity_lower_map = {e.lower(): e for e in _schema_node_names}

    for filter_key in filters:
        # Match the longest entity prefix (e.g. "order" before "order_date"
        # is ambiguous — we check all nodes and take the first prefix match).
        matched_entity = None
        for entity_lower, entity in _entity_lower_map.items():
            if filter_key.startswith(entity_lower + "_") or filter_key == entity_lower:
                matched_entity = entity
                break
        if matched_entity is None:
            continue
        if matched_entity in entity_set:
            continue
        # N2 guard: if employee_name filter is present AND Project is already in
        # the path, do NOT inject Employee here — the N2 semantic rule below
        # must handle this case because it inserts Employee BEFORE Project.
        # A generic append would produce [Project, Employee] — wrong JOIN direction.
        if filter_key == "employee_name" and "Project" in entity_set:
            continue
        # T1.6: Department + Order path guard.
        # When BOTH department_name and order_status are in filters, injecting
        # Department creates a traversal the graph cannot resolve (no Dept→Order edge).
        # The correct path is Order→Employee, with department_name resolved via the
        # Employee node's filter_supplements.department_name subquery — which fires
        # automatically when Department is NOT in the path.
        # Guard fires ONLY when Order is already present; safe for all Order-free queries.
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

    # --- LLM entity injection (formerly Refinement 1 injection-signal loop) ---
    # The schema-driven LLM extractor now returns complete entity lists directly
    # (e.g. ["Employee", "Project"] for assignment queries). The regex-based
    # injection_signals mechanism is no longer needed. The structural traversal
    # rules below (Department positional injection, manager_name→Project, N2, N4,
    # Order display expansion) remain — they encode JOIN correctness constraints
    # from the graph topology, not linguistic patterns.

    # --- Department injection (Semantic — positional: after Employee) ---
    # P6.1: Guard replicated from T2-A loop — skip if Order is already in path.
    # When Order is present, department_name is resolved via Employee's
    # filter_supplements.department_name subquery (no Department JOIN needed).
    # Without this guard, Department is injected positionally → Department→Order
    # NoPathError → fallback to Employee-only → order_status filter silently dropped.
    if "department_name" in filters and "Department" not in entity_set and "Order" not in entity_set:
        if "Employee" in entity_set:
            if "Project" in entity_set:
                # Need full 3-hop: Employee → Department → Project
                # But graph path is Employee→Department and Department→Project
                # Insert Department between Employee and Project
                idx = expanded.index("Employee")
                expanded.insert(idx + 1, "Department")
            else:
                idx = expanded.index("Employee")
                expanded.insert(idx + 1, "Department")
            entity_set.add("Department")
            logger.info(
                "Path extension: injected Department (department_name filter present)"
            )

    # --- Project injection from manager_name (Semantic — non-schema key) ---
    if "manager_name" in filters and "Project" not in entity_set:
        if "Employee" in entity_set:
            expanded.append("Project")
            entity_set.add("Project")
            logger.info(
                "Path extension: injected Project (manager_name filter present)"
            )

    # N2: Employee injection for employee_name + Project path.
    # When a question names an employee AND asks about projects
    # (e.g. "Sarah Connor is in which project"), the system must traverse
    # Employee→Project via project_assignments — not single-node Project.
    # Without this rule, employee_name filter is silently dropped because
    # it doesn't exist in _build_filter_map for a Project-only path.
    if "employee_name" in filters and "Employee" not in entity_set and "Project" in entity_set:
        idx = expanded.index("Project")
        expanded.insert(idx, "Employee")
        entity_set.add("Employee")
        logger.info(
            "Path extension: injected Employee before Project "
            "(employee_name filter + Project entity — needs Employee→Project join)"
        )

    # N4: reports_to_name routing — org-chart hierarchy queries.
    # "employees who report to X" = employees in the department managed by X.
    # Route to Employee-only path; the reports_to_name subquery filter in
    # _build_filter_map resolves the manager name to a department via the
    # departments.manager_id FK — purely schema-driven, no names hardcoded.
    if "reports_to_name" in filters:
        if "Employee" not in entity_set:
            expanded.append("Employee")
            entity_set.add("Employee")
        # Remove Project from path if it was injected by manager_name logic —
        # reports_to is an employee hierarchy query, not a project query.
        if "Project" in entity_set and "manager_name" not in filters:
            expanded = [e for e in expanded if e != "Project"]
            entity_set.discard("Project")
            logger.info(
                "Path extension: removed Project from path "
                "(reports_to_name query — org-chart, not project manager)"
            )
        logger.info("Path extension: Employee path for reports_to_name filter")

    # --- Order display expansion (P1-B) ---
    # When Order is the anchor entity it owns two outgoing edges in the schema:
    #   Order -[placed_by]->  Employee  (o.employee_id = e.id)
    #   Order -[contains]->   Product   (o.product_id  = p.id)
    # Without these JOINs the SELECT has no product name or ordered_by column.
    # Neighbours are derived from the live graph object — not hardcoded strings —
    # so this stays correct if edges are ever added/renamed in graph_schema.yaml.
    if "Order" in entity_set:
        # Fix 4 — Order+Department co-presence guard                         #
        # Department has no edge to Order in the schema (no Order↔Dept join). #
        # When both appear (e.g. Q20: "Who in Operations placed orders..."),   #
        # Department must be removed from the path. The department_name filter #
        # is resolved instead via Employee's filter_supplement subquery, which #
        # fires automatically whenever Department is NOT in the path.          #
        # This guard removes Department regardless of whether it came from the #
        # LLM or from injection — the topology constraint applies either way.  #
        # Multi-DB safe: reads only from schema graph edges, no names hardcoded.#
        if "Department" in entity_set:
            # Use has_edge (direct edge only) — NOT nx.has_path which returns True
            # for any reachable multi-hop path (Order→Employee→Department exists
            # as a 2-hop but that is not a valid JOIN anchor for this query).
            # We only preserve Department in the entity list when there is a direct
            # schema edge Order→Department. In the current schema there is none.
            has_direct_order_dept_edge = _graph._graph.has_edge("Order", "Department")
            if not has_direct_order_dept_edge:
                expanded = [e for e in expanded if e != "Department"]
                entity_set.discard("Department")
                logger.info(
                    "Path extension: removed Department "
                    "(Order in path — no Order→Department edge; "
                    "department_name resolved via Employee subquery)"
                )

        schema_order_neighbours = set(_graph._graph.successors("Order"))
        for neighbour in sorted(schema_order_neighbours):  # sorted for stable order
            if neighbour not in entity_set:
                expanded.append(neighbour)
                entity_set.add(neighbour)
                logger.info(
                    "Path extension: injected %s "
                    "(Order display expansion via schema edge)", neighbour
                )

    # --- Department display expansion (P1-C / D7 / R7) ---
    # When Department is the sole entity, we must decide which edge to use:
    #
    #   Employee→Department  (works_in: e.department_id = d.id)  → ALL staff in dept
    #   Department→Employee  (managed_by: d.manager_id = e.id)   → only the 1 manager
    #
    # R7: "list" queries asking for "everyone in HR" need ALL employees, not just
    # the manager.  Inject Employee BEFORE Department for list, comparison, and
    # cross_entity queries so _canonical_entity_order keeps Employee as anchor.
    # Only true aggregation queries (headcount/budget breakdown) can safely use
    # the managed_by JOIN because they GROUP BY department anyway.
    if "Department" in entity_set and "Employee" not in entity_set:
        if q_type in ("comparison", "list", "cross_entity", "lookup"):
            # Insert Employee BEFORE Department → Employee→Department path →
            # works_in JOIN → returns all employees in the department.
            idx = expanded.index("Department")
            expanded.insert(idx, "Employee")
            entity_set.add("Employee")
            logger.info(
                "Path extension: injected Employee before Department "
                "(%s query — needs Employee→Department path for all staff)", q_type
            )
        else:
            # aggregation: use managed_by join (one manager row) + GROUP BY dept
            expanded.append("Employee")
            entity_set.add("Employee")
            logger.info(
                "Path extension: injected Employee after Department "
                "(aggregation — managed_by join for dept summary)"
            )

    return _canonical_entity_order(expanded, set(expanded), filters, original_entities=entities, q_type=q_type)


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

    The default priority list (Employee > Department > Project > Order > Product)
    is correct when the user explicitly asked about multiple entity types.

    Schema-driven anchor override: when a single entity was originally detected
    and others were injected by display expansion (e.g. Order injects Employee
    and Product; Department injects Employee for manager display), the originally-
    detected node must anchor the JOIN — not whatever the default priority puts
    first.  The condition is: candidate was the SOLE original entity AND it has
    a direct outgoing edge to all injected members, verified via the live graph.

    original_entities: the pre-expansion entity list (None = no override).
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
    # The override must NOT fire when the candidate↔injected relationship is
    # bidirectional (edges exist in BOTH directions in the schema graph).
    # Reason: when both directions exist (e.g. Employee↔Project, Employee↔Department),
    # _expand_entities_from_filters deliberately orders them to select the correct
    # JOIN direction. Overriding that ordering picks the wrong edge.
    #
    # The override SHOULD still fire for unidirectional relationships like
    # Order→Employee and Order→Product (no Employee→Order or Product→Order edges exist).
    # In those cases Order is the only valid anchor and the override is correct.
    #
    # This check is purely graph-structure-driven — reads the live NetworkX graph,
    # no entity names or filter keys hardcoded.
    if original_entities and len(set(original_entities)) == 1 and q_type != "comparison":
        candidate = original_entities[0]
        if candidate in entity_set:
            others = entity_set - {candidate}
            if others and all(_graph._graph.has_edge(candidate, other) for other in others):
                # P1: check if ANY reverse edge exists (injected → candidate)
                has_any_reverse_edge = any(
                    _graph._graph.has_edge(other, candidate) for other in others
                )
                if has_any_reverse_edge:
                    # Bidirectional relationship — injection order was deliberate.
                    # Fall through to default priority ordering which respects it.
                    logger.debug(
                        "Anchor override suppressed for %s: reverse edges exist from injected "
                        "nodes — respecting _expand_entities_from_filters ordering",
                        candidate,
                    )
                else:
                    # Unidirectional — candidate is the only valid anchor (e.g. Order).
                    ordered: List[str] = [candidate]
                    secondary = ["Employee", "Department", "Project", "Order", "Product"]
                    for node in secondary:
                        if node in entity_set and node != candidate:
                            ordered.append(node)
                    for node in entities:
                        if node not in ordered:
                            ordered.append(node)
                    logger.debug(
                        "Anchor override: %s leads (sole original, unidirectional edges only)",
                        candidate,
                    )
                    return ordered

    # Default priority for all other combinations (Employee > Dept > Project > Product)
    priority = ["Employee", "Department", "Project", "Order", "Product"]
    ordered = []
    for node in priority:
        if node in entity_set:
            ordered.append(node)

    # Append anything not in the priority list (future schema additions)
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
