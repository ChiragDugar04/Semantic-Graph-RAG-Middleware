from __future__ import annotations

import logging
import yaml
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

class EntityNotFoundError(Exception):
    pass

class NoPathError(Exception):
    pass

class SchemaValidationError(Exception):
    pass

class JoinStep:
    def __init__(self, edge_data: Dict[str, Any]) -> None:
        self.from_node: str = edge_data["from_node"]
        self.to_node: str = edge_data["to_node"]
        self.relation: str = edge_data["relation"]
        self.join_type: str = edge_data.get("join_type", "INNER")
        self.join_condition: str = edge_data["join_condition"]
        self.junction_table: Optional[str] = edge_data.get("junction_table")
        self.junction_alias: Optional[str] = edge_data.get("junction_alias")
        self.extra_select: List[str] = edge_data.get("extra_select", [])
        self.manager_join: bool = edge_data.get("manager_join", False)
        self.filter_supplements: Dict[str, Any] = edge_data.get("filter_supplements", {})

    def __repr__(self) -> str:
        return (
            f"JoinStep({self.from_node} -[{self.relation}]-> {self.to_node}, "
            f"junction={self.junction_table})"
        )


class SemanticGraph:
    def __init__(self, schema_path: Optional[Path] = None) -> None:
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "config" / "graph_schema.yaml"

        if not schema_path.exists():
            raise SchemaValidationError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            raw = yaml.safe_load(f)

        self._validate_raw_schema(raw)

        self._nodes: Dict[str, Any] = raw["nodes"]
        self._edges: List[Dict[str, Any]] = raw["edges"]
        self._graph: nx.DiGraph = nx.DiGraph()
        self._build_graph()
        self._validate_graph()

        logger.info(
            "SemanticGraph loaded: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def _validate_raw_schema(self, raw: dict) -> None:
        if "nodes" not in raw or not raw["nodes"]:
            raise SchemaValidationError("graph_schema.yaml must define at least one node")
        if "edges" not in raw or not raw["edges"]:
            raise SchemaValidationError("graph_schema.yaml must define at least one edge")

        required_node_fields = {"table", "alias", "selectable_columns", "filterable_columns", "primary_key"}
        for node_name, node_data in raw["nodes"].items():
            missing = required_node_fields - set(node_data.keys())
            if missing:
                raise SchemaValidationError(
                    f"Node '{node_name}' is missing required fields: {missing}"
                )

        required_edge_fields = {"from_node", "to_node", "relation", "join_condition"}
        node_names = set(raw["nodes"].keys())
        for i, edge in enumerate(raw["edges"]):
            missing = required_edge_fields - set(edge.keys())
            if missing:
                raise SchemaValidationError(
                    f"Edge #{i} is missing required fields: {missing}"
                )
            if edge["from_node"] not in node_names:
                raise SchemaValidationError(
                    f"Edge #{i} references unknown from_node: '{edge['from_node']}'"
                )
            if edge["to_node"] not in node_names:
                raise SchemaValidationError(
                    f"Edge #{i} references unknown to_node: '{edge['to_node']}'"
                )

    def _validate_graph(self) -> None:
        for node in self._graph.nodes():
            if self._graph.degree(node) == 0:
                logger.warning("Node '%s' has no edges — it will never be reachable in joins", node)

    def _build_graph(self) -> None:
        for node_name, node_data in self._nodes.items():
            self._graph.add_node(node_name, **node_data)

        for edge in self._edges:
            self._graph.add_edge(
                edge["from_node"],
                edge["to_node"],
                **edge,
            )

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def node_names(self) -> List[str]:
        return list(self._graph.nodes())

    @property
    def _schema_node_names(self) -> List[str]:
        # T2-A: returns nodes in YAML insertion order (Python 3.7+ dict order),
        # which is the detection priority order. Used by _expand_entities_from_filters
        # generic loop to build the entity_lower→entity prefix map.
        return list(self._nodes.keys())

    def get_node_data(self, entity: str) -> Dict[str, Any]:
        if entity not in self._graph:
            raise EntityNotFoundError(
                f"Entity '{entity}' not found. Available: {self.node_names}"
            )
        return dict(self._graph.nodes[entity])

    def get_selectable_columns(self, entity: str) -> List[Dict[str, Any]]:
        return self.get_node_data(entity).get("selectable_columns", [])

    def get_filterable_columns(self, entity: str) -> Dict[str, Any]:
        return self.get_node_data(entity).get("filterable_columns", {})

    def get_table_name(self, entity: str) -> str:
        return self.get_node_data(entity)["table"]

    def get_table_alias(self, entity: str) -> str:
        return self.get_node_data(entity)["alias"]

    def get_all_filterable_columns(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for entity in self.node_names:
            for col_key, col_meta in self.get_filterable_columns(entity).items():
                result[f"{entity}.{col_key}"] = {**col_meta, "entity": entity}
        return result

    def find_path(self, from_entity: str, to_entity: str) -> List[str]:
        for entity in [from_entity, to_entity]:
            if entity not in self._graph:
                raise EntityNotFoundError(
                    f"Entity '{entity}' not found. Available: {self.node_names}"
                )

        if from_entity == to_entity:
            return [from_entity]

        try:
            return nx.shortest_path(self._graph, from_entity, to_entity)
        except nx.NetworkXNoPath:
            try:
                return nx.shortest_path(
                    self._graph.to_undirected(), from_entity, to_entity
                )
            except nx.NetworkXNoPath:
                raise NoPathError(
                    f"No path found between '{from_entity}' and '{to_entity}'"
                )

    def find_multi_path(self, entities: List[str]) -> List[str]:
        if not entities:
            return []
        if len(entities) == 1:
            return list(entities)

        for entity in entities:
            if entity not in self._graph:
                raise EntityNotFoundError(
                    f"Entity '{entity}' not found. Available: {self.node_names}"
                )

        if len(entities) == 2:
            return self.find_path(entities[0], entities[1])

        required = list(dict.fromkeys(entities))
        anchor = required[0]

        try:
            direct = self.find_path(anchor, required[-1])
            if all(r in set(direct) for r in required):
                return direct
        except (NoPathError, nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        full_path: List[str] = [anchor]
        path_set: set = {anchor}

        for target in required[1:]:
            if target in path_set:
                continue
            tip = full_path[-1]
            try:
                segment = self.find_path(tip, target)
                for node in segment[1:]:
                    if node not in path_set:
                        full_path.append(node)
                        path_set.add(node)
            except (NoPathError, nx.NetworkXNoPath, nx.NodeNotFound):
                try:
                    segment = self.find_path(anchor, target)
                    for node in segment:
                        if node not in path_set:
                            full_path.append(node)
                            path_set.add(node)
                except Exception as exc:
                    raise NoPathError(
                        f"Cannot connect '{tip}' to '{target}': {exc}"
                    ) from exc

        return full_path

    def get_join_chain(self, path: List[str]) -> List[JoinStep]:
        if len(path) < 2:
            return []

        anchor = path[0]
        chain: List[JoinStep] = []
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]

            if self._graph.has_edge(from_node, to_node):
                edge_data = dict(self._graph[from_node][to_node])
            elif self._graph.has_edge(to_node, from_node):
                edge_data = dict(self._graph[to_node][from_node])
                edge_data["from_node"] = from_node
                edge_data["to_node"] = to_node
            elif self._graph.has_edge(anchor, to_node):
                # Fan-out topology: no direct edge from_node→to_node, but the
                # path anchor has a direct edge to to_node (e.g. Order→Product
                # when path is [Order, Employee, Product]).  Use the anchor's
                # edge so both JOINs reference the same root table.
                edge_data = dict(self._graph[anchor][to_node])
                edge_data["from_node"] = anchor
                edge_data["to_node"] = to_node
            else:
                raise NoPathError(f"No edge between '{from_node}' and '{to_node}'")

            chain.append(JoinStep(edge_data))

        return chain

    def describe_path(self, path: List[str]) -> str:
        if not path:
            return "(empty path)"
        if len(path) == 1:
            return path[0]

        parts = []
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            relation = "→"
            if self._graph.has_edge(from_node, to_node):
                relation = self._graph[from_node][to_node].get("relation", "→")
            if i == 0:
                parts.append(from_node)
            parts.append(f"→{relation}→")
            parts.append(to_node)

        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"SemanticGraph(nodes={self.node_count}, "
            f"edges={self.edge_count}, entities={self.node_names})"
        )
