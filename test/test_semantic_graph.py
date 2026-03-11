"""
tests/test_semantic_graph.py

Phase 1 test suite for the SemanticGraph class.

Run with:
    pytest tests/test_semantic_graph.py -v

All 8 tests must pass before moving to Phase 2.
"""

from __future__ import annotations

import pytest

from middleware.semantic_graph import (
    SemanticGraph,
    EntityNotFoundError,
    NoPathError,
    JoinStep,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def graph() -> SemanticGraph:
    """Single shared graph instance for all tests.

    scope=module means it's built once and reused — loading
    YAML + building the NetworkX graph is fast but no need
    to repeat it for every test.
    """
    return SemanticGraph()


# ============================================================
# TEST 1 — Graph loads correctly
# ============================================================

class TestGraphLoading:

    def test_node_count(self, graph: SemanticGraph) -> None:
        """Graph should have exactly 5 entity nodes."""
        assert graph.node_count == 5, (
            f"Expected 5 nodes, got {graph.node_count}. "
            f"Nodes present: {graph.node_names}"
        )

    def test_edge_count(self, graph: SemanticGraph) -> None:
        """Graph should have exactly 7 directed edges."""
        assert graph.edge_count == 7, (
            f"Expected 7 edges, got {graph.edge_count}"
        )

    def test_all_expected_nodes_present(self, graph: SemanticGraph) -> None:
        """All 5 entity names must be registered."""
        expected = {"Employee", "Department", "Product", "Order", "Project"}
        actual   = set(graph.node_names)
        assert expected == actual, (
            f"Missing nodes: {expected - actual}"
        )

    def test_graph_repr(self, graph: SemanticGraph) -> None:
        """repr() should include node and edge counts."""
        r = repr(graph)
        assert "nodes=5" in r
        assert "edges=7" in r


# ============================================================
# TEST 2 — Node metadata
# ============================================================

class TestNodeMetadata:

    def test_employee_table_name(self, graph: SemanticGraph) -> None:
        assert graph.get_table_name("Employee") == "employees"

    def test_department_table_alias(self, graph: SemanticGraph) -> None:
        assert graph.get_table_alias("Department") == "d"

    def test_project_selectable_columns(self, graph: SemanticGraph) -> None:
        cols = graph.get_selectable_columns("Project")
        col_names = [c["column"] for c in cols]
        assert "name" in col_names
        assert "status" in col_names
        assert "budget" in col_names

    def test_employee_filterable_columns(self, graph: SemanticGraph) -> None:
        filters = graph.get_filterable_columns("Employee")
        assert "name" in filters
        assert filters["name"]["match_type"] == "like"

    def test_unknown_entity_raises(self, graph: SemanticGraph) -> None:
        with pytest.raises(EntityNotFoundError) as exc_info:
            graph.get_node_data("Spaceship")
        assert "Spaceship" in str(exc_info.value)


# ============================================================
# TEST 3 — 1-hop path: Employee → Department
# ============================================================

class TestOneHopPath:

    def test_employee_to_department_path(self, graph: SemanticGraph) -> None:
        """Shortest path Employee→Department should be 2 nodes."""
        path = graph.find_path("Employee", "Department")
        assert path == ["Employee", "Department"], (
            f"Expected ['Employee', 'Department'], got {path}"
        )

    def test_order_to_product_path(self, graph: SemanticGraph) -> None:
        """Order→Product is a direct 1-hop relationship."""
        path = graph.find_path("Order", "Product")
        assert path == ["Order", "Product"]

    def test_order_to_employee_path(self, graph: SemanticGraph) -> None:
        """Order→Employee is a direct 1-hop (placed_by)."""
        path = graph.find_path("Order", "Employee")
        assert path == ["Order", "Employee"]

    def test_same_entity_path(self, graph: SemanticGraph) -> None:
        """Path from an entity to itself should return just that entity."""
        path = graph.find_path("Employee", "Employee")
        assert path == ["Employee"]


# ============================================================
# TEST 4 — 2-hop path
# ============================================================

class TestTwoHopPath:

    def test_employee_to_project_path(self, graph: SemanticGraph) -> None:
        """Employee→Project should be a 2-node path (direct edge)."""
        path = graph.find_path("Employee", "Project")
        assert len(path) == 2
        assert path[0] == "Employee"
        assert path[-1] == "Project"

    def test_order_to_department_path(self, graph: SemanticGraph) -> None:
        """Order→Department requires going through Employee."""
        path = graph.find_path("Order", "Department")
        # Order→Employee→Department
        assert len(path) == 3
        assert path[0] == "Order"
        assert path[-1] == "Department"
        assert "Employee" in path


# ============================================================
# TEST 5 — Multi-path (3+ entities)
# ============================================================

class TestMultiPath:

    def test_three_entity_path(self, graph: SemanticGraph) -> None:
        """Employee + Department + Project should produce a connected path."""
        path = graph.find_multi_path(["Employee", "Department", "Project"])
        # Must start with Employee, contain Department and Project
        assert path[0] == "Employee"
        assert "Department" in path
        assert "Project" in path

    def test_two_entity_multi_path(self, graph: SemanticGraph) -> None:
        """Multi-path with 2 entities should behave like find_path."""
        path = graph.find_multi_path(["Employee", "Department"])
        assert path == ["Employee", "Department"]

    def test_single_entity_multi_path(self, graph: SemanticGraph) -> None:
        """Multi-path with 1 entity returns just that entity."""
        path = graph.find_multi_path(["Project"])
        assert path == ["Project"]

    def test_empty_multi_path(self, graph: SemanticGraph) -> None:
        """Multi-path with empty list returns empty list."""
        path = graph.find_multi_path([])
        assert path == []


# ============================================================
# TEST 6 — Error cases
# ============================================================

class TestErrorCases:

    def test_unknown_entity_in_find_path(self, graph: SemanticGraph) -> None:
        with pytest.raises(EntityNotFoundError):
            graph.find_path("Employee", "Unicorn")

    def test_unknown_source_in_find_path(self, graph: SemanticGraph) -> None:
        with pytest.raises(EntityNotFoundError):
            graph.find_path("Robot", "Department")

    def test_unknown_entity_in_multi_path(self, graph: SemanticGraph) -> None:
        with pytest.raises(EntityNotFoundError):
            graph.find_multi_path(["Employee", "Wormhole"])


# ============================================================
# TEST 7 — JOIN chain generation
# ============================================================

class TestJoinChain:

    def test_employee_department_join_chain(self, graph: SemanticGraph) -> None:
        """Employee→Department chain should have 1 step with correct JOIN."""
        path  = graph.find_path("Employee", "Department")
        chain = graph.get_join_chain(path)

        assert len(chain) == 1
        step = chain[0]
        assert isinstance(step, JoinStep)
        assert step.from_node == "Employee"
        assert step.to_node   == "Department"
        assert "department_id" in step.join_condition
        assert step.junction_table is None

    def test_employee_project_junction_table(self, graph: SemanticGraph) -> None:
        """Employee→Project must go via project_assignments junction table."""
        path  = graph.find_path("Employee", "Project")
        chain = graph.get_join_chain(path)

        assert len(chain) == 1
        step = chain[0]
        assert step.junction_table == "project_assignments"
        assert step.junction_alias == "pa"

    def test_order_product_chain(self, graph: SemanticGraph) -> None:
        """Order→Product chain should use product_id FK."""
        path  = graph.find_path("Order", "Product")
        chain = graph.get_join_chain(path)

        assert len(chain) == 1
        step = chain[0]
        assert "product_id" in step.join_condition

    def test_multi_hop_chain_length(self, graph: SemanticGraph) -> None:
        """Order→Department (via Employee) should have 2 join steps."""
        path  = graph.find_path("Order", "Department")
        chain = graph.get_join_chain(path)
        assert len(chain) == 2

    def test_join_step_join_type(self, graph: SemanticGraph) -> None:
        """Each step should have a valid join_type of INNER or LEFT."""
        path  = graph.find_path("Employee", "Department")
        chain = graph.get_join_chain(path)
        for step in chain:
            assert step.join_type in ("INNER", "LEFT"), (
                f"Invalid join_type: {step.join_type}"
            )

    def test_empty_path_returns_empty_chain(self, graph: SemanticGraph) -> None:
        """Single-node path produces no join steps."""
        chain = graph.get_join_chain(["Employee"])
        assert chain == []


# ============================================================
# TEST 8 — describe_path utility
# ============================================================

class TestDescribePath:

    def test_single_node_description(self, graph: SemanticGraph) -> None:
        desc = graph.describe_path(["Employee"])
        assert desc == "Employee"

    def test_two_node_description(self, graph: SemanticGraph) -> None:
        desc = graph.describe_path(["Employee", "Department"])
        assert "Employee" in desc
        assert "Department" in desc
        assert "works_in" in desc

    def test_empty_path_description(self, graph: SemanticGraph) -> None:
        desc = graph.describe_path([])
        assert desc == "(empty path)"