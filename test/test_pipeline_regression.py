"""
tests/test_pipeline_regression.py

Phase 3 full regression suite.

Covers:
  - All original demo questions (backward compat)
  - Two-hop traversal queries
  - Cross-entity / multi-hop queries
  - Self-healing cases
  - Trace structure integrity

Run with:
    pytest tests/test_pipeline_regression.py -v -s
"""

from __future__ import annotations

import pytest
from middleware.pipeline import run_pipeline
from middleware.models import MiddlewareTrace


def run(q: str) -> MiddlewareTrace:
    trace = run_pipeline(q)
    path  = trace.graph_traversal.path_taken if trace.graph_traversal else []
    mode  = trace.graph_traversal.traversal_method if trace.graph_traversal else "—"
    print(f"\n  Q: {q}")
    print(f"  A: {trace.final_answer[:100]}...")
    print(f"  Path: {path}  Mode: {mode}")
    return trace


# ============================================================
# ORIGINAL DEMO QUESTIONS — must still work exactly
# ============================================================

class TestBackwardCompatibility:

    def test_sarah_connor_salary(self) -> None:
        trace = run("What is Sarah Connor's salary?")
        assert not trace.self_healing_triggered
        assert "sarah" in trace.final_answer.lower() or "95" in trace.final_answer

    def test_engineering_budget(self) -> None:
        trace = run("What is the Engineering department budget?")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1

    def test_laptop_stock(self) -> None:
        trace = run("How many Laptop Pro 15 units are in stock?")
        assert not trace.self_healing_triggered
        assert "45" in trace.final_answer or "laptop" in trace.final_answer.lower()

    def test_list_sales_employees(self) -> None:
        trace = run("List all employees in Sales")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1
        assert any(
            name in trace.final_answer.lower()
            for name in ["michael", "jim", "dwight", "pam"]
        )

    def test_recent_orders(self) -> None:
        trace = run("Show me all recent orders")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1

    def test_pending_orders(self) -> None:
        trace = run("Show me all pending orders")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1

    def test_most_expensive_product(self) -> None:
        trace = run("What is the most expensive product?")
        assert not trace.self_healing_triggered
        assert "laptop" in trace.final_answer.lower() or "1299" in trace.final_answer


# ============================================================
# TWO-HOP TRAVERSAL
# ============================================================

class TestTwoHopTraversal:

    def test_list_engineering_employees(self) -> None:
        trace = run("List all employees in Engineering")
        assert not trace.self_healing_triggered
        assert trace.graph_traversal.traversal_method == "two_hop"
        assert trace.graph_traversal.join_count == 1
        assert trace.db_result.row_count >= 1

    def test_highest_paid_engineering(self) -> None:
        trace = run("Who is the highest paid in Engineering?")
        assert not trace.self_healing_triggered
        assert "linus" in trace.final_answer.lower() or "130" in trace.final_answer

    def test_highest_paid_marketing(self) -> None:
        trace = run("Who earns the most in Marketing?")
        assert not trace.self_healing_triggered
        assert "don" in trace.final_answer.lower() or "draper" in trace.final_answer or "98" in trace.final_answer

    def test_operations_employees(self) -> None:
        trace = run("Who works in Operations?")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1
        assert any(
            name in trace.final_answer.lower()
            for name in ["leslie", "ben", "tom"]
        )


# ============================================================
# CROSS-ENTITY / MULTI-HOP
# ============================================================

class TestCrossEntityQueries:

    def test_api_gateway_employees(self) -> None:
        trace = run("Which employees are assigned to the API Gateway Rebuild project?")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1
        assert any(
            name in trace.final_answer.lower()
            for name in ["sarah", "john", "grace", "ben"]
        )

    def test_platform_migration_employees(self) -> None:
        trace = run("Who is working on the Platform Migration project?")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1

    def test_engineering_projects_managed_by_sarah(self) -> None:
        trace = run(
            "Which employees in Engineering are working on "
            "projects managed by Sarah?"
        )
        assert isinstance(trace, MiddlewareTrace)
        assert trace.pipeline_stage_reached == "complete"
        assert trace.final_answer

    def test_don_draper_projects(self) -> None:
        trace = run("Who are the employees on projects managed by Don Draper?")
        assert isinstance(trace, MiddlewareTrace)
        assert trace.pipeline_stage_reached == "complete"
        assert trace.final_answer


# ============================================================
# SELF-HEALING
# ============================================================

class TestSelfHealing:

    def test_weather_heals(self) -> None:
        trace = run("What is the weather today?")
        assert trace.self_healing_triggered

    def test_joke_heals(self) -> None:
        trace = run("Tell me a joke")
        assert trace.self_healing_triggered

    def test_healing_produces_answer(self) -> None:
        trace = run("What is the weather today?")
        assert len(trace.final_answer) > 0


# ============================================================
# TRACE STRUCTURE — all fields populated correctly
# ============================================================

class TestTraceStructure:

    def test_graph_traversal_populated(self) -> None:
        trace = run("List all employees in Engineering")
        assert trace.graph_traversal is not None
        assert len(trace.graph_traversal.path_taken) > 0
        assert trace.graph_traversal.path_description != ""
        assert trace.graph_traversal.traversal_method in (
            "single_node", "two_hop", "multi_hop"
        )

    def test_entity_extraction_populated(self) -> None:
        trace = run("What is Sarah Connor's salary?")
        assert trace.entity_extraction is not None
        assert "Employee" in trace.entity_extraction.entities
        assert trace.entity_extraction.extraction_method in ("rules", "llm")

    def test_semantic_mode_true(self) -> None:
        trace = run("List all employees in Engineering")
        assert trace.semantic_mode is True

    def test_intent_carries_path(self) -> None:
        trace = run("List all employees in Engineering")
        assert "Employee" in trace.intent.intent_name
        assert "Department" in trace.intent.intent_name

    def test_parameters_carries_filters(self) -> None:
        trace = run("List all employees in Engineering")
        assert "department_name" in trace.parameters.params or \
               "name" in trace.parameters.params

    def test_total_latency_positive(self) -> None:
        trace = run("What is Sarah Connor's salary?")
        assert trace.total_latency_ms > 0

    def test_pipeline_reaches_complete(self) -> None:
        trace = run("What is Sarah Connor's salary?")
        assert trace.pipeline_stage_reached == "complete"

    def test_tables_involved_populated(self) -> None:
        trace = run("List all employees in Engineering")
        assert len(trace.graph_traversal.tables_involved) >= 1
        assert "employees" in trace.graph_traversal.tables_involved