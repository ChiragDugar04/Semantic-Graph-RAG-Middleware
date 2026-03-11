"""
middleware/semantic_graph.py

The Semantic Knowledge Graph — the heart of the new architecture.

Represents the database as a graph of entities (nodes) and
relationships (edges). Given two or more entity names, it finds
the shortest JOIN path between them and returns the metadata
needed to construct valid SQL automatically.

This is what replaces the flat intents.yaml lookup. Instead of
a human writing a SQL template for every possible question type,
the graph dynamically figures out how to connect any two entities
the user asks about.

Usage:
    graph = SemanticGraph()
    path  = graph.find_path("Employee", "Department")
    chain = graph.get_join_chain(path)
"""

from __future__ import annotations

import yaml
import networkx as nx

from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class EntityNotFoundError(Exception):
    """Raised when a requested entity name is not in the graph."""
    pass


class NoPathError(Exception):
    """Raised when no traversal path exists between two entities."""
    pass


# ============================================================
# DATA CLASSES
# (plain dicts kept simple — Pydantic models added in Phase 2)
# ============================================================

class JoinStep:
    """One hop in a JOIN chain.

    Attributes:
        from_node     : Source entity name (e.g. "Employee")
        to_node       : Target entity name (e.g. "Department")
        relation      : Relationship label (e.g. "works_in")
        join_type     : "INNER" or "LEFT"
        join_condition: The ON clause string (e.g. "e.department_id = d.id")
        junction_table: Optional bridge table name for many-to-many
        junction_alias: Alias for the junction table
        extra_select  : Optional extra columns to include in SELECT
        manager_join  : True if this edge requires a second alias for employee
    """

    def __init__(self, edge_data: Dict[str, Any]) -> None:
        self.from_node:      str            = edge_data["from_node"]
        self.to_node:        str            = edge_data["to_node"]
        self.relation:       str            = edge_data["relation"]
        self.join_type:      str            = edge_data.get("join_type", "INNER")
        self.join_condition: str            = edge_data["join_condition"]
        self.junction_table: Optional[str]  = edge_data.get("junction_table")
        self.junction_alias: Optional[str]  = edge_data.get("junction_alias")
        self.extra_select:   List[str]      = edge_data.get("extra_select", [])
        self.manager_join:   bool           = edge_data.get("manager_join", False)

    def __repr__(self) -> str:
        return (
            f"JoinStep({self.from_node} -[{self.relation}]-> {self.to_node}, "
            f"junction={self.junction_table})"
        )


# ============================================================
# SEMANTIC GRAPH
# ============================================================

class SemanticGraph:
    """Knowledge graph of database entities and their relationships.

    Loads node and edge definitions from graph_schema.yaml and
    builds a NetworkX directed graph. Provides path-finding and
    JOIN chain generation for the query builder.

    Attributes:
        _graph    : The underlying NetworkX DiGraph
        _nodes    : Raw node config dict from YAML
        _edges    : Raw edge list from YAML
    """

    def __init__(self, schema_path: Optional[Path] = None) -> None:
        """Load graph schema and build the NetworkX graph.

        Args:
            schema_path: Path to graph_schema.yaml. Defaults to
                         config/graph_schema.yaml relative to project root.
        """
        if schema_path is None:
            schema_path = (
                Path(__file__).parent.parent / "config" / "graph_schema.yaml"
            )

        with open(schema_path, "r") as f:
            raw = yaml.safe_load(f)

        self._nodes: Dict[str, Any] = raw["nodes"]
        self._edges: List[Dict[str, Any]] = raw["edges"]
        self._graph: nx.DiGraph = nx.DiGraph()

        self._build_graph()

    # ----------------------------------------------------------
    # GRAPH CONSTRUCTION
    # ----------------------------------------------------------

    def _build_graph(self) -> None:
        """Populate the NetworkX graph from YAML definitions.

        Adds one node per entity and one directed edge per relationship.
        All YAML metadata is stored as edge attributes so it can be
        retrieved during path traversal.
        """
        # Add nodes
        for node_name, node_data in self._nodes.items():
            self._graph.add_node(node_name, **node_data)

        # Add directed edges with full metadata
        for edge in self._edges:
            self._graph.add_edge(
                edge["from_node"],
                edge["to_node"],
                **edge,
            )

    # ----------------------------------------------------------
    # GRAPH INSPECTION
    # ----------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of entity nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of relationship edges in the graph."""
        return self._graph.number_of_edges()

    @property
    def node_names(self) -> List[str]:
        """List of all entity names in the graph."""
        return list(self._graph.nodes())

    def get_node_data(self, entity: str) -> Dict[str, Any]:
        """Return the full YAML config for a node.

        Args:
            entity: Entity name (e.g. "Employee").

        Returns:
            dict: Node metadata from graph_schema.yaml.

        Raises:
            EntityNotFoundError: If entity not in graph.
        """
        if entity not in self._graph:
            raise EntityNotFoundError(
                f"Entity '{entity}' not found in graph. "
                f"Available: {self.node_names}"
            )
        return dict(self._graph.nodes[entity])

    def get_selectable_columns(self, entity: str) -> List[Dict[str, Any]]:
        """Return columns that can appear in SELECT for this entity.

        Args:
            entity: Entity name.

        Returns:
            List of column dicts with 'column', 'alias', 'label' keys.
        """
        node = self.get_node_data(entity)
        return node.get("selectable_columns", [])

    def get_filterable_columns(self, entity: str) -> Dict[str, Any]:
        """Return columns that can appear in WHERE for this entity.

        Args:
            entity: Entity name.

        Returns:
            Dict mapping filter key → filter metadata.
        """
        node = self.get_node_data(entity)
        return node.get("filterable_columns", {})

    def get_table_name(self, entity: str) -> str:
        """Return the SQL table name for an entity.

        Args:
            entity: Entity name.

        Returns:
            str: Table name (e.g. "employees").
        """
        return self.get_node_data(entity)["table"]

    def get_table_alias(self, entity: str) -> str:
        """Return the SQL alias for an entity's table.

        Args:
            entity: Entity name.

        Returns:
            str: Table alias (e.g. "e" for Employee).
        """
        return self.get_node_data(entity)["alias"]

    # ----------------------------------------------------------
    # PATH FINDING
    # ----------------------------------------------------------

    def find_path(self, from_entity: str, to_entity: str) -> List[str]:
        """Find the shortest path between two entities in the graph.

        Uses NetworkX shortest_path on the directed graph.
        Returns the node names along the path — pass this to
        get_join_chain() to get the actual JOIN instructions.

        Args:
            from_entity: Starting entity name.
            to_entity:   Target entity name.

        Returns:
            List[str]: Node names along the path, e.g.
                       ["Employee", "Department"]

        Raises:
            EntityNotFoundError: If either entity not in graph.
            NoPathError: If no directed path exists.
        """
        # Validate both entities exist
        for entity in [from_entity, to_entity]:
            if entity not in self._graph:
                raise EntityNotFoundError(
                    f"Entity '{entity}' not found. "
                    f"Available: {self.node_names}"
                )

        if from_entity == to_entity:
            return [from_entity]

        try:
            path = nx.shortest_path(self._graph, from_entity, to_entity)
            return path
        except nx.NetworkXNoPath:
            # Try undirected path as fallback
            try:
                undirected = self._graph.to_undirected()
                path = nx.shortest_path(undirected, from_entity, to_entity)
                return path
            except nx.NetworkXNoPath:
                raise NoPathError(
                    f"No path found between '{from_entity}' and '{to_entity}'"
                )

    def find_multi_path(self, entities: List[str]) -> List[str]:
        """Find a path that connects multiple entities without loops.

        For 2 entities: delegates to find_path().
        For 3+ entities: builds a Steiner-tree-style minimal path
        that visits all required entities exactly once, avoiding
        the duplicate-node problem of naive pairwise chaining.

        Strategy:
          1. Start from anchor (entities[0]).
          2. For each remaining required entity, find the shortest
             path from any node already in the path to that entity.
             This avoids re-traversing nodes already covered.
          3. If a segment introduces a node already in the path,
             skip back-tracking nodes to keep the path linear.

        Args:
            entities: List of entity names to connect (anchor first).

        Returns:
            List[str]: Linear node path with no duplicates.

        Raises:
            EntityNotFoundError: If any entity not in graph.
            NoPathError: If entities cannot be connected.
        """
        if len(entities) == 0:
            return []
        if len(entities) == 1:
            return entities

        # Validate all entities exist
        for entity in entities:
            if entity not in self._graph:
                raise EntityNotFoundError(
                    f"Entity '{entity}' not found. "
                    f"Available: {self.node_names}"
                )

        # For exactly 2 entities, simple direct path
        if len(entities) == 2:
            return self.find_path(entities[0], entities[1])

        # For 3+ entities: greedy Steiner approach
        # Try direct path from anchor through all required entities
        # in one shot first (e.g. Employee→Department→Project directly)
        required = list(dict.fromkeys(entities))  # dedupe, preserve order
        anchor = required[0]
        remaining = required[1:]

        # ── Strategy 1: try finding a direct path anchor→...→last ──
        # that naturally passes through all intermediate required nodes
        try:
            direct = self.find_path(anchor, remaining[-1])
            # Check if this path covers all required entities
            direct_set = set(direct)
            if all(r in direct_set for r in remaining):
                # Reorder path to match required entity order if needed
                return direct
        except (NoPathError, nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # ── Strategy 2: greedy expansion from current path tip ───
        # For each required entity not yet in path, find shortest
        # path from the LAST node in current path to that entity.
        full_path: List[str] = [anchor]
        path_set: set = {anchor}

        for target in remaining:
            if target in path_set:
                continue  # already covered

            # Try path from current tip to target
            tip = full_path[-1]
            try:
                segment = self.find_path(tip, target)
                # Add segment nodes, skipping any already in path
                for node in segment[1:]:
                    if node not in path_set:
                        full_path.append(node)
                        path_set.add(node)
                    elif node == target:
                        # target already in path — it's covered, continue
                        break
            except (NoPathError, nx.NetworkXNoPath, nx.NodeNotFound):
                # Try from anchor directly
                try:
                    segment = self.find_path(anchor, target)
                    for node in segment:
                        if node not in path_set:
                            full_path.append(node)
                            path_set.add(node)
                except Exception as e2:
                    raise NoPathError(
                        f"Cannot connect '{tip}' to '{target}': {e2}"
                    ) from e2

        return full_path

    # ----------------------------------------------------------
    # JOIN CHAIN GENERATION
    # ----------------------------------------------------------

    def get_join_chain(self, path: List[str]) -> List[JoinStep]:
        """Convert a node path into ordered JOIN instructions.

        Takes the output of find_path() or find_multi_path() and
        returns a list of JoinStep objects. Each JoinStep contains
        everything the query builder needs to write one JOIN clause.

        Args:
            path: List of entity names (from find_path output).

        Returns:
            List[JoinStep]: One step per hop in the path.

        Raises:
            NoPathError: If no edge exists between consecutive nodes.

        Example:
            path = ["Employee", "Department"]
            chain = graph.get_join_chain(path)
            # chain[0].join_condition == "e.department_id = d.id"
        """
        if len(path) < 2:
            return []

        chain: List[JoinStep] = []

        for i in range(len(path) - 1):
            from_node = path[i]
            to_node   = path[i + 1]

            if not self._graph.has_edge(from_node, to_node):
                # Try reverse direction
                if self._graph.has_edge(to_node, from_node):
                    edge_data = dict(self._graph[to_node][from_node])
                    # Swap for correct orientation
                    edge_data["from_node"] = from_node
                    edge_data["to_node"]   = to_node
                else:
                    raise NoPathError(
                        f"No edge between '{from_node}' and '{to_node}'"
                    )
            else:
                edge_data = dict(self._graph[from_node][to_node])

            chain.append(JoinStep(edge_data))

        return chain

    # ----------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------

    def describe_path(self, path: List[str]) -> str:
        """Return a human-readable string describing a path.

        Args:
            path: List of entity names.

        Returns:
            str: E.g. "Employee →works_in→ Department"
        """
        if not path:
            return "(empty path)"
        if len(path) == 1:
            return path[0]

        parts = []
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node   = path[i + 1]
            if self._graph.has_edge(from_node, to_node):
                relation = self._graph[from_node][to_node].get("relation", "→")
            else:
                relation = "→"
            if i == 0:
                parts.append(from_node)
            parts.append(f"→{relation}→")
            parts.append(to_node)

        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"SemanticGraph("
            f"nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"entities={self.node_names})"
        )
