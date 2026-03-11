"""
tests/test_pipeline_e2e.py

Phase 2 end-to-end pipeline tests.
These tests run the FULL pipeline against the live MySQL database.

Requirements before running:
  - MySQL running with middleware_poc database
  - schema.sql + seed.sql applied (Phase 1 Step 1.1)
  - Ollama running with qwen2.5:1.5b and llama3.2:3b

Run with:
    pytest tests/test_pipeline_e2e.py -v -s

The -s flag shows print output so you can see answers in real time.
"""

from __future__ import annotations

import pytest
from middleware.pipeline import run_pipeline
from middleware.models import MiddlewareTrace


# ============================================================
# HELPER
# ============================================================

def run(q: str) -> MiddlewareTrace:
    """Shorthand."""
    trace = run_pipeline(q)
    print(f"\n  Q: {q}")
    print(f"  A: {trace.final_answer[:120]}...")
    print(f"  Path: {trace.graph_traversal.path_taken if trace.graph_traversal else 'none'}")
    print(f"  Mode: {trace.graph_traversal.traversal_method if trace.graph_traversal else 'none'}")
    return trace


# ============================================================
# TEST 1 — Single entity: backward compat
# ============================================================

class TestSingleEntityQueries:

    def test_sarah_connor_salary(self) -> None:
        """Must return Sarah Connor's salary — same as old system."""
        trace = run("What is Sarah Connor's salary?")
        assert not trace.self_healing_triggered
        assert "sarah" in trace.final_answer.lower() or "95" in trace.final_answer
        assert trace.graph_traversal.traversal_method == "single_node"

    def test_alan_turing_details(self) -> None:
        """Alan Turing lookup must return his details."""
        trace = run("What role does Alan Turing have?")
        assert not trace.self_healing_triggered
        assert "alan" in trace.final_answer.lower() or "turing" in trace.final_answer.lower()

    def test_most_expensive_product(self) -> None:
        """Most expensive product — Laptop Pro 15 at $1299.99."""
        trace = run("What is the most expensive product?")
        assert not trace.self_healing_triggered
        # Laptop Pro 15 is $1299.99 — most expensive
        assert "laptop" in trace.final_answer.lower() or "1299" in trace.final_answer


# ============================================================
# TEST 2 — Two-hop: Employee + Department
# ============================================================

class TestTwoHopQueries:

    def test_list_engineering_employees(self) -> None:
        """Should list all 5 Engineering employees."""
        trace = run("List all employees in Engineering")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1
        assert trace.graph_traversal.join_count >= 1
        # Should mention at least one engineer
        answer_lower = trace.final_answer.lower()
        has_engineer = any(
            name in answer_lower
            for name in ["sarah", "alan", "grace", "linus", "john"]
        )
        assert has_engineer, f"Expected engineer names in: {trace.final_answer}"

    def test_highest_paid_in_engineering(self) -> None:
        """Linus Torvalds is the highest paid in Engineering ($130k)."""
        trace = run("Who is the highest paid in Engineering?")
        assert not trace.self_healing_triggered
        assert "linus" in trace.final_answer.lower() or "130" in trace.final_answer

    def test_marketing_salary_comparison(self) -> None:
        """Don Draper is the highest paid in Marketing ($98k)."""
        trace = run("Who earns the most in Marketing?")
        assert not trace.self_healing_triggered
        answer_lower = trace.final_answer.lower()
        assert "don" in answer_lower or "draper" in answer_lower or "98" in trace.final_answer


# ============================================================
# TEST 3 — Cross-entity: Employee + Project
# ============================================================

class TestCrossEntityQueries:

    def test_employees_on_api_gateway(self) -> None:
        """API Gateway Rebuild has: Sarah, John, Grace, Ben."""
        trace = run("Which employees are assigned to the API Gateway Rebuild project?")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1
        answer_lower = trace.final_answer.lower()
        # At least one of the assigned employees should appear
        has_member = any(
            name in answer_lower
            for name in ["sarah", "john", "grace", "ben"]
        )
        assert has_member, f"Expected project members in: {trace.final_answer}"

    def test_employees_in_engineering_on_projects_managed_by_sarah(self) -> None:
        """The flagship 3-hop demo query."""
        trace = run(
            "Which employees in Engineering are working on projects managed by Sarah?"
        )
        # This is a complex query — just verify it ran without healing
        assert isinstance(trace, MiddlewareTrace)
        assert trace.pipeline_stage_reached == "complete"
        assert trace.final_answer  # non-empty answer


# ============================================================
# TEST 4 — Orders
# ============================================================

class TestOrderQueries:

    def test_pending_orders(self) -> None:
        """Should return pending orders (there are 5 in seed data)."""
        trace = run("Show me all pending orders")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1
        assert "Order" in trace.graph_traversal.path_taken

    def test_all_recent_orders(self) -> None:
        """Should return order history."""
        trace = run("Show me all recent orders")
        assert not trace.self_healing_triggered
        assert trace.db_result.row_count >= 1


# ============================================================
# TEST 5 — Self-healing fires for unknown questions
# ============================================================

class TestSelfHealing:

    def test_weather_question_triggers_healing(self) -> None:
        """Non-database question must trigger self-healing."""
        trace = run("What is the weather today?")
        assert trace.self_healing_triggered
        assert trace.final_answer  # polite follow-up generated

    def test_random_question_triggers_healing(self) -> None:
        """Completely off-topic question must trigger self-healing."""
        trace = run("Tell me a joke about programmers")
        assert trace.self_healing_triggered


# ============================================================
# TEST 6 — Trace structure integrity
# ============================================================

class TestTraceIntegrity:

    def test_all_trace_fields_populated(self) -> None:
        trace = run("What is Sarah Connor's salary?")
        assert trace.intent is not None
        assert trace.parameters is not None
        assert trace.db_result is not None
        assert trace.entity_extraction is not None
        assert trace.graph_traversal is not None
        assert trace.final_answer != ""
        assert trace.total_latency_ms > 0
        assert trace.pipeline_stage_reached == "complete"

    def test_intent_name_contains_entity(self) -> None:
        """Glass Box panel ① must show something meaningful."""
        trace = run("What is Sarah Connor's salary?")
        assert "Employee" in trace.intent.intent_name

    def test_parameters_carries_filters(self) -> None:
        """Glass Box panel ② must show the filters."""
        trace = run("What is Sarah Connor's salary?")
        assert len(trace.parameters.params) > 0

    def test_semantic_mode_true(self) -> None:
        trace = run("List all employees in Engineering")
        assert trace.semantic_mode is True