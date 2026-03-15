from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

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
    _NAMED_FILTERS = {"product_name", "employee_name", "project_name", "department_name"}
    if (extraction.question_type == "aggregation"
            and _NAMED_FILTERS & set(extraction.filters.keys())):
        logger.info(
            "D5: aggregation downgraded to lookup (named filter present: %s)",
            list(_NAMED_FILTERS & set(extraction.filters.keys())),
        )
        extraction = extraction.model_copy(update={"question_type": "lookup"})

    traversal, join_chain = _run_graph_traversal(extraction)
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
        unique_entities, filters, q_type=extraction.question_type
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
) -> List[str]:
    """
    Filter-Aware Path Extension.

    If a filter references an entity that is not yet in the detected entity
    list, automatically inject that entity so the graph traversal builds the
    correct JOIN path.

    Rules (order matters — evaluated top-to-bottom):
    - `department_name` filter present but no Department entity
      → inject Department between Employee and Project (if Project present)
        or after Employee (if no Project)
    - `manager_name` filter present but no Project entity
      → inject Project (manager context only makes sense via projects)
    - `project_name` / `project_status` filter present but no Project entity
      → inject Project
    - `product_name` / `category` filter present but no Product entity
      → inject Product
    - `order_status` filter present but no Order entity
      → inject Order
    """
    expanded = list(dict.fromkeys(entities))   # deduplicate, preserve order
    entity_set = set(expanded)

    # --- Department injection ---
    if "department_name" in filters and "Department" not in entity_set:
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

    # --- Project injection from manager_name ---
    if "manager_name" in filters and "Project" not in entity_set:
        if "Employee" in entity_set:
            expanded.append("Project")
            entity_set.add("Project")
            logger.info(
                "Path extension: injected Project (manager_name filter present)"
            )

    # --- Project injection from project filters ---
    for fk in ("project_name", "project_status"):
        if fk in filters and "Project" not in entity_set:
            if "Employee" in entity_set:
                expanded.append("Project")
                entity_set.add("Project")
                logger.info(
                    "Path extension: injected Project (%s filter present)", fk
                )
            break

    # --- Product injection ---
    if "product_name" in filters and "Product" not in entity_set:
        expanded.append("Product")
        entity_set.add("Product")
        logger.info("Path extension: injected Product (product_name filter present)")

    # --- Order injection ---
    if "order_status" in filters and "Order" not in entity_set:
        expanded.append("Order")
        entity_set.add("Order")
        logger.info("Path extension: injected Order (order_status filter present)")

    # --- Order display expansion (P1-B) ---
    # When Order is the anchor entity it owns two outgoing edges in the schema:
    #   Order -[placed_by]->  Employee  (o.employee_id = e.id)
    #   Order -[contains]->   Product   (o.product_id  = p.id)
    # Without these JOINs the SELECT has no product name or ordered_by column.
    # Neighbours are derived from the live graph object — not hardcoded strings —
    # so this stays correct if edges are ever added/renamed in graph_schema.yaml.
    if "Order" in entity_set:
        schema_order_neighbours = set(_graph._graph.successors("Order"))
        for neighbour in sorted(schema_order_neighbours):  # sorted for stable order
            if neighbour not in entity_set:
                expanded.append(neighbour)
                entity_set.add(neighbour)
                logger.info(
                    "Path extension: injected %s "
                    "(Order display expansion via schema edge)", neighbour
                )

    # --- Department display expansion (P1-C / D7) ---
    # Default: inject Employee so the Department->Employee (managed_by) JOIN
    # is built, giving manager_name for display queries.
    #
    # D7 exception: for comparison queries (highest/lowest salary etc.) the
    # managed_by JOIN only returns ONE employee (the manager). We need ALL
    # employees in the department for a correct MAX/MIN.  Inject Employee
    # first in the list so _canonical_entity_order places it before Department
    # and the works_in edge (Employee→Department) is used instead.
    if "Department" in entity_set and "Employee" not in entity_set:
        if q_type == "comparison":
            # Insert Employee BEFORE Department so canonical order keeps Employee
            # as anchor → Employee→Department path → works_in JOIN → all employees.
            idx = expanded.index("Department")
            expanded.insert(idx, "Employee")
            entity_set.add("Employee")
            logger.info(
                "Path extension: injected Employee before Department "
                "(comparison query — needs all employees, not just manager)"
            )
        else:
            expanded.append("Employee")
            entity_set.add("Employee")
            logger.info(
                "Path extension: injected Employee "
                "(Department display expansion — headcount/manager)"
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
    if original_entities and len(set(original_entities)) == 1 and q_type != "comparison":
        candidate = original_entities[0]
        if candidate in entity_set:
            others = entity_set - {candidate}
            if others and all(_graph._graph.has_edge(candidate, other) for other in others):
                ordered: List[str] = [candidate]
                secondary = ["Employee", "Department", "Project", "Order", "Product"]
                for node in secondary:
                    if node in entity_set and node != candidate:
                        ordered.append(node)
                for node in entities:
                    if node not in ordered:
                        ordered.append(node)
                logger.debug(
                    "Anchor override: %s leads (sole original, reaches all injected nodes)",
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
