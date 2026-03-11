"""
tests/test_graph_query_builder.py

Phase 2 test suite for graph_query_builder.py

Run with:
    pytest tests/test_graph_query_builder.py -v

Tests that the query builder produces correct SQL structure
from entity extraction + graph traversal inputs.
Does NOT hit the database — purely tests SQL string construction.
"""

from __future__ import annotations

import pytest

from middleware.semantic_graph import SemanticGraph
from middleware.graph_query_builder import GraphQueryBuilder
from middleware.models import EntityExtractionResult, GraphTraversal


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def graph() -> SemanticGraph:
    return SemanticGraph()


@pytest.fixture(scope="module")
def builder(graph: SemanticGraph) -> GraphQueryBuilder:
    return GraphQueryBuilder(graph)


def make_extraction(
    entities,
    filters=None,
    projections=None,
    question_type="lookup",
    method="rules",
) -> EntityExtractionResult:
    """Helper to build EntityExtractionResult for tests."""
    return EntityExtractionResult(
        entities=entities,
        filters=filters or {},
        projections=projections or [],
        question_type=question_type,
        extraction_method=method,
        latency_ms=0.0,
    )


def make_traversal(path, graph: SemanticGraph) -> tuple:
    """Helper to build GraphTraversal + join_chain from a path."""
    join_chain = graph.get_join_chain(path)
    tables = [graph.get_table_name(e) for e in path]
    join_count = len(join_chain)
    method = (
        "single_node" if join_count == 0
        else "two_hop" if join_count == 1
        else "multi_hop"
    )
    traversal = GraphTraversal(
        path_taken=path,
        join_count=join_count,
        tables_involved=tables,
        traversal_time_ms=0.0,
        traversal_method=method,
        path_description=graph.describe_path(path),
    )
    return traversal, join_chain


# ============================================================
# TEST 1 — Single node: no JOINs
# ============================================================

class TestSingleNodeQuery:

    def test_employee_name_filter(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee"],
            filters={"employee_name": "Sarah Connor"},
            question_type="lookup",
        )
        traversal, join_chain = make_traversal(["Employee"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "FROM employees e" in sql
        assert "JOIN" not in sql
        assert "employee_name" in params
        assert "Sarah Connor" in params["employee_name"]  # LIKE wrapped

    def test_single_node_has_select(self, builder, graph) -> None:
        extraction = make_extraction(entities=["Employee"], question_type="lookup")
        traversal, join_chain = make_traversal(["Employee"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert sql.strip().upper().startswith("SELECT")
        assert "FROM employees" in sql

    def test_product_single_node(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Product"],
            filters={"product_name": "Laptop Pro 15"},
            question_type="lookup",
        )
        traversal, join_chain = make_traversal(["Product"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "FROM products p" in sql
        assert "JOIN" not in sql


# ============================================================
# TEST 2 — Two-hop: Employee → Department JOIN
# ============================================================

class TestTwoHopJoin:

    def test_employee_department_join_present(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            filters={"department_name": "Engineering"},
            question_type="list",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "JOIN departments d" in sql
        assert "department_id" in sql

    def test_department_filter_in_where(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            filters={"department_name": "Marketing"},
            question_type="list",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        assert "department_name" in params
        assert "Marketing" in params["department_name"]

    def test_list_type_has_order_by(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            filters={"department_name": "Sales"},
            question_type="list",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "ORDER BY" in sql


# ============================================================
# TEST 3 — Junction table: Employee → Project
# ============================================================

class TestJunctionTableJoin:

    def test_project_assignments_present(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Project"],
            filters={"project_name": "API Gateway Rebuild"},
            question_type="cross_entity",
        )
        traversal, join_chain = make_traversal(["Employee", "Project"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "project_assignments" in sql
        assert "pa" in sql   # junction alias

    def test_project_table_joined(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Project"],
            question_type="cross_entity",
        )
        traversal, join_chain = make_traversal(["Employee", "Project"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "JOIN projects proj" in sql


# ============================================================
# TEST 4 — Comparison question type
# ============================================================

class TestComparisonQuery:

    def test_comparison_has_order_by_salary_desc(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            filters={"department_name": "Engineering"},
            question_type="comparison",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "ORDER BY e.salary DESC" in sql

    def test_product_comparison_price_desc(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Product"],
            question_type="comparison",
        )
        traversal, join_chain = make_traversal(["Product"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "ORDER BY p.price DESC" in sql


# ============================================================
# TEST 5 — Aggregation question type
# ============================================================

class TestAggregationQuery:

    def test_count_present_in_select(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            question_type="aggregation",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "COUNT(" in sql

    def test_group_by_present(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            question_type="aggregation",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        assert "GROUP BY" in sql


# ============================================================
# TEST 6 — QueryTemplate output shape
# ============================================================

class TestQueryTemplateShape:

    def test_intent_name_contains_path(self, builder, graph) -> None:
        extraction = make_extraction(
            entities=["Employee", "Department"],
            question_type="list",
        )
        traversal, join_chain = make_traversal(["Employee", "Department"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        assert "Employee" in template.intent_name
        assert "Department" in template.intent_name

    def test_description_is_not_empty(self, builder, graph) -> None:
        extraction = make_extraction(entities=["Employee"], question_type="lookup")
        traversal, join_chain = make_traversal(["Employee"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        assert len(template.description) > 0

    def test_sql_starts_with_select(self, builder, graph) -> None:
        extraction = make_extraction(entities=["Employee"], question_type="lookup")
        traversal, join_chain = make_traversal(["Employee"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        assert template.sql_template.strip().upper().startswith("SELECT")

    def test_required_params_empty(self, builder, graph) -> None:
        """required_params must be empty — params are baked in via dict."""
        extraction = make_extraction(entities=["Employee"], question_type="lookup")
        traversal, join_chain = make_traversal(["Employee"], graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        assert template.required_params == []


# ============================================================
# TEST 7 — Multi-hop: 3 entities
# ============================================================

class TestMultiHopQuery:

    def test_three_entity_has_multiple_joins(self, builder, graph) -> None:
        path = graph.find_multi_path(["Employee", "Department", "Project"])
        extraction = make_extraction(
            entities=["Employee", "Department", "Project"],
            filters={"department_name": "Engineering"},
            question_type="cross_entity",
        )
        traversal, join_chain = make_traversal(path, graph)
        template, params = builder.build_query(extraction, traversal, join_chain)

        sql = template.sql_template
        join_count = sql.upper().count("JOIN")
        assert join_count >= 2, f"Expected 2+ JOINs, got {join_count}. SQL:\n{sql}"