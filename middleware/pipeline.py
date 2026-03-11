"""
middleware/pipeline.py

The pipeline orchestrator — Phase 2 rewrite using the Semantic Graph.

The public interface (run_pipeline) is UNCHANGED. Streamlit calls
the same function and receives the same MiddlewareTrace object.

New step order:
  1. Entity Extraction    (rules + qwen2.5:1.5b)     ← replaces intent + param steps
  2. Graph Traversal      (NetworkX, pure Python)     ← new
  3. Query Building       (GraphQueryBuilder)         ← new
  4. Query Execution      (MySQL + self-healing)      ← UNCHANGED
  5. Context Formatting   (pure Python)               ← UNCHANGED
  6. Answer Synthesis     (llama3.2:3b)               ← UNCHANGED

Backward compatibility:
  trace.intent and trace.parameters are populated from the new
  extraction + traversal data so the existing Glass Box panels
  ①② render correctly without any changes to app.py.
"""

from __future__ import annotations

import time

from middleware.models import (
    MiddlewareTrace,
    IntentClassification,
    ExtractedParameters,
    GraphTraversal,
)
from middleware.entity_extractor import extract_entities
from middleware.semantic_graph import SemanticGraph, EntityNotFoundError, NoPathError
from middleware.graph_query_builder import GraphQueryBuilder
from middleware.query_executor import execute_query
from middleware.context_formatter import format_context
from middleware.answer_synthesizer import synthesize_answer


# ============================================================
# MODULE-LEVEL SINGLETONS
# Build graph and query builder once at import time.
# NetworkX graph construction is fast (~5ms) but no need to
# repeat it on every request.
# ============================================================

_graph   = SemanticGraph()
_builder = GraphQueryBuilder(_graph)


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(question: str) -> MiddlewareTrace:
    """Execute the full Semantic Graph RAG pipeline for a user question.

    Builds and returns a MiddlewareTrace containing the final answer
    and complete audit data for the Glass Box panel.

    Args:
        question: The raw user question as typed in the UI.

    Returns:
        MiddlewareTrace: Complete trace with answer and all audit fields.

    Example:
        >>> trace = run_pipeline("Which employees in Engineering work on projects managed by Sarah?")
        >>> trace.final_answer
        "3 employees in Engineering are assigned to projects managed by Sarah Connor..."
        >>> trace.graph_traversal.path_taken
        ['Employee', 'Department', 'Project']
    """
    pipeline_start = time.time()
    trace = MiddlewareTrace(user_question=question)
    trace.semantic_mode = True

    # ── Step 1: Extract Entities ─────────────────────────────
    extraction = extract_entities(question)
    trace.entity_extraction = extraction
    trace.pipeline_stage_reached = "extraction"

    # ── Step 2: Graph Traversal ───────────────────────────────
    traversal, join_chain = _run_graph_traversal(extraction)
    trace.graph_traversal = traversal
    trace.pipeline_stage_reached = "traversal"

    # ── Populate legacy trace.intent for Glass Box panel ① ───
    # intent_name stores the path string so the panel is informative
    path_str = "→".join(traversal.path_taken) if traversal.path_taken else "other"
    trace.intent = IntentClassification(
        intent_name=path_str,
        confidence="high" if extraction.extraction_method == "rules" else "medium",
        raw_llm_response=f"entities={extraction.entities}, type={extraction.question_type}",
        model_used=f"semantic_graph [{extraction.extraction_method}]",
        latency_ms=extraction.latency_ms,
    )

    # ── Populate legacy trace.parameters for Glass Box panel ②
    trace.parameters = ExtractedParameters(
        params=extraction.filters,
        missing_required=[],
        extraction_method=extraction.extraction_method,
        latency_ms=extraction.latency_ms,
    )

    # ── Step 3: Build Query ───────────────────────────────────
    # Handle "other" / no-entity case — triggers self-healing
    if not extraction.entities or extraction.question_type == "other":
        template, query_params = _build_fallback_template()
    else:
        try:
            template, query_params = _builder.build_query(
                extraction=extraction,
                traversal=traversal,
                join_chain=join_chain,
            )
        except Exception as build_err:
            template, query_params = _build_fallback_template()

    trace.pipeline_stage_reached = "query_built"

    # ── Step 4: Execute Query ─────────────────────────────────
    # Inject query_params into ExtractedParameters for executor
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

    # ── Step 5: Format Context ────────────────────────────────
    context = format_context(
        db_result=trace.db_result,
        template=template,
    )

    # ── Step 6: Synthesize Answer ─────────────────────────────
    trace.final_answer = synthesize_answer(
        user_question=question,
        context=context,
        db_result=trace.db_result,
    )
    trace.pipeline_stage_reached = "complete"

    trace.total_latency_ms = round((time.time() - pipeline_start) * 1000, 2)
    return trace


# ============================================================
# GRAPH TRAVERSAL HELPER
# ============================================================

def _run_graph_traversal(extraction) -> tuple:
    """Run graph path-finding and build GraphTraversal metadata.

    Args:
        extraction: EntityExtractionResult from entity_extractor.

    Returns:
        Tuple of (GraphTraversal, List[JoinStep]).
        Returns a single-node traversal if path-finding fails.
    """
    start = time.time()
    entities = extraction.entities

    # ── No entities → return empty traversal ─────────────────
    if not entities:
        traversal = GraphTraversal(
            path_taken=[],
            join_count=0,
            tables_involved=[],
            traversal_time_ms=0.0,
            traversal_method="single_node",
            path_description="No entities detected",
        )
        return traversal, []

    # ── Single entity → no JOIN needed ───────────────────────
    if len(set(entities)) == 1:
        entity   = entities[0]
        table    = _graph.get_table_name(entity)
        elapsed  = round((time.time() - start) * 1000, 2)
        traversal = GraphTraversal(
            path_taken=[entity],
            join_count=0,
            tables_involved=[table],
            traversal_time_ms=elapsed,
            traversal_method="single_node",
            path_description=entity,
        )
        return traversal, []

    # ── Multiple entities → find shortest connecting path ────
    unique_entities = list(dict.fromkeys(entities))   # preserve order, dedupe

    # ── CRITICAL: Normalize entity order ─────────────────────
    # The graph has two edges between Employee and Department:
    #   Employee →works_in→    Department  (e.department_id = d.id)  ← ALL employees
    #   Department →managed_by→ Employee   (d.manager_id = e.id)     ← ONE manager only
    #
    # For ANY question about employees within a department
    # (list, comparison, aggregation) we MUST anchor on Employee
    # so the works_in edge is used, not managed_by.
    # Force Employee before Department whenever both are present.
    if "Employee" in unique_entities and "Department" in unique_entities:
        # Rebuild with Employee first
        reordered = ["Employee"]
        for e in unique_entities:
            if e != "Employee":
                reordered.append(e)
        unique_entities = reordered

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
        method     = (
            "single_node" if join_count == 0
            else "two_hop" if join_count == 1
            else "multi_hop"
        )

        elapsed   = round((time.time() - start) * 1000, 2)
        path_desc = _graph.describe_path(path)

        traversal = GraphTraversal(
            path_taken=path,
            join_count=join_count,
            tables_involved=tables,
            traversal_time_ms=elapsed,
            traversal_method=method,
            path_description=path_desc,
        )
        return traversal, join_chain

    except (EntityNotFoundError, NoPathError) as e:
        # Graceful fallback: use first entity only
        entity   = unique_entities[0]
        table    = _graph.get_table_name(entity)
        elapsed  = round((time.time() - start) * 1000, 2)
        traversal = GraphTraversal(
            path_taken=[entity],
            join_count=0,
            tables_involved=[table],
            traversal_time_ms=elapsed,
            traversal_method="single_node",
            path_description=f"{entity} (path error: {str(e)[:50]})",
        )
        return traversal, []


# ============================================================
# FALLBACK TEMPLATE (triggers self-healing)
# ============================================================

def _build_fallback_template():
    """Return an empty template that triggers the self-healing loop.

    Used when: no entities detected, or query building fails.
    The executor's Guard 1 (no SQL) fires and generates a polite
    follow-up question for the user.

    Returns:
        Tuple of (QueryTemplate, empty params dict).
    """
    from middleware.models import QueryTemplate
    return QueryTemplate(
        intent_name="other",
        description="No matching database query found",
        sql_template="",
        required_params=[],
        optional_params=[],
        result_description="",
    ), {}