from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from middleware.models import EntityExtractionResult, GraphTraversal, QueryTemplate
from middleware.semantic_graph import SemanticGraph, JoinStep

logger = logging.getLogger(__name__)


class QueryBuildError(Exception):
    pass


class GraphQueryBuilder:
    def __init__(self, graph: SemanticGraph) -> None:
        self._graph = graph

    def build_query(
        self,
        extraction: EntityExtractionResult,
        traversal: GraphTraversal,
        join_chain: List[JoinStep],
        effective_entities: Optional[List[str]] = None,
    ) -> Tuple[QueryTemplate, Dict[str, Any]]:
        entities = effective_entities if effective_entities else extraction.entities
        filters = extraction.filters
        q_type = extraction.question_type
        path = traversal.path_taken

        if not path:
            raise QueryBuildError("Cannot build query: traversal path is empty")

        anchor = path[0]
        joined_tables: Set[str] = self._get_joined_tables(path, join_chain)

        select_parts = self._build_select(entities, q_type, join_chain, extraction.projections, path)
        from_clause = self._build_from(anchor)
        join_clauses = self._build_joins(join_chain)

        if "Project" in path:
            mgr_join = "LEFT JOIN employees mgr ON mgr.id = proj.manager_id"
            if mgr_join not in join_clauses:
                join_clauses.append(mgr_join)
            mgr_col = "mgr.name AS manager_name"
            if mgr_col not in select_parts:
                select_parts = [p for p in select_parts if "manager_id" not in p]
                select_parts.append(mgr_col)

        # Honour manager_join=True on any edge (e.g. Department -[managed_by]-> Employee).
        # When this flag is set, the JOIN is on the manager FK (d.manager_id = e.id),
        # so e.name is the manager's name — expose it as manager_name, not employee_name.
        for step in join_chain:
            if step.manager_join and "Project" not in path:
                mgr_col = "e.name AS manager_name"
                if mgr_col not in select_parts:
                    # Replace employee_name with manager_name for this context
                    select_parts = [
                        mgr_col if p == "e.name AS employee_name" else p
                        for p in select_parts
                    ]
                    if mgr_col not in select_parts:
                        select_parts.append(mgr_col)

        where_clause, where_params = self._build_where(filters, path, joined_tables, join_chain)
        order_limit = self._build_order_limit(q_type, entities, filters)

        sql_parts = [f"SELECT DISTINCT {', '.join(select_parts)}", from_clause]
        sql_parts.extend(join_clauses)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
        sql_parts.append(order_limit)

        sql = "\n".join(p for p in sql_parts if p.strip())
        description = self._build_description(entities, q_type, filters, path)

        template = QueryTemplate(
            intent_name=f"graph:{'→'.join(path)}",
            description=description,
            sql_template=sql,
            required_params=[],
            optional_params=[],
            result_description=description,
        )

        logger.debug("Built query for path=%s type=%s filters=%s", path, q_type, list(filters.keys()))

        return template, where_params

    def _get_joined_tables(self, path: List[str], join_chain: List[JoinStep]) -> Set[str]:
        tables = set(path)
        for step in join_chain:
            if step.junction_table:
                tables.add(step.junction_table)
        tables.add("employees_mgr") if "Project" in path else None
        return tables

    def _build_select(
        self,
        entities: List[str],
        q_type: str,
        join_chain: List[JoinStep],
        projections: Optional[List[str]],
        path: List[str],
    ) -> List[str]:
        if q_type == "aggregation":
            return self._build_aggregation_select(entities)

        parts: List[str] = []
        seen: Set[str] = set()

        for entity in path:
            if entity in seen:
                continue
            seen.add(entity)

            node = self._graph.get_node_data(entity)
            alias = node["alias"]

            for col_def in node.get("selectable_columns", []):
                col = col_def["column"]
                col_alias = col_def["alias"]

                if col_def.get("is_hidden"):
                    continue

                if col_def.get("default_hidden"):
                    requested = projections or []
                    if col_alias not in requested and col not in requested:
                        continue

                if col_def.get("is_subquery"):
                    parts.append(f"{col} AS {col_alias}")
                else:
                    parts.append(f"{alias}.{col} AS {col_alias}")

        for step in join_chain:
            for extra in step.extra_select:
                if extra not in parts:
                    parts.append(extra)

        return parts if parts else ["*"]

    def _build_aggregation_select(self, entities: List[str]) -> List[str]:
        if "Employee" in entities and "Department" in entities:
            return [
                "d.name AS department_name",
                "COUNT(e.id) AS employee_count",
                "AVG(e.salary) AS avg_salary",
                "MAX(e.salary) AS max_salary",
                "MIN(e.salary) AS min_salary",
            ]
        if "Employee" in entities and "Project" in entities:
            return ["proj.name AS project_name", "COUNT(e.id) AS employee_count"]
        if "Order" in entities:
            return ["o.status", "COUNT(o.id) AS order_count", "SUM(o.quantity) AS total_quantity"]
        if "Employee" in entities:
            return ["COUNT(e.id) AS employee_count"]
        return ["COUNT(*) AS total_count"]

    def _build_from(self, anchor_entity: str) -> str:
        node = self._graph.get_node_data(anchor_entity)
        return f"FROM {node['table']} {node['alias']}"

    def _build_joins(self, join_chain: List[JoinStep]) -> List[str]:
        joins: List[str] = []
        joined_tables: Set[str] = set()

        for step in join_chain:
            node = self._graph.get_node_data(step.to_node)
            table = node["table"]
            alias = node["alias"]
            j_type = step.join_type

            if step.junction_table:
                jt = step.junction_table
                jalias = step.junction_alias or jt[:2]
                conditions = [c.strip() for c in step.join_condition.split(" AND ")]

                if len(conditions) == 2:
                    cond1, cond2 = conditions
                else:
                    cond1 = step.join_condition
                    cond2 = f"{jalias}.{step.to_node.lower()}_id = {alias}.id"

                if jt not in joined_tables:
                    joins.append(f"{j_type} JOIN {jt} {jalias} ON {cond1}")
                    joined_tables.add(jt)

                if table not in joined_tables:
                    joins.append(f"{j_type} JOIN {table} {alias} ON {cond2}")
                    joined_tables.add(table)
            else:
                if table not in joined_tables:
                    joins.append(f"{j_type} JOIN {table} {alias} ON {step.join_condition}")
                    joined_tables.add(table)

        return joins

    def _build_filter_map(
        self,
        path: List[str],
        join_chain: Optional[List["JoinStep"]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a dynamic filter-key → SQL mapping by reading filterable_columns
        from the schema for every node in the current path.

        Returns a dict like:
          {
            "employee_name":   {"sql": "e.name",     "match": "like"},
            "employee_role":   {"sql": "e.role",     "match": "like"},
            "department_name": {"sql": "d.name",     "match": "like"},
            "project_name":    {"sql": "proj.name",  "match": "like"},
            "project_status":  {"sql": "proj.status","match": "exact"},
            ...
          }

        The key naming convention is:  <entity_lower>_<column>
        with a special alias for "name" columns → <entity_lower>_name.

        T1-D: Cross-table supplement filters are now read from graph_schema.yaml:
          - Node-level filter_supplements: apply whenever that node is in path.
            Each entry may carry exclude_when_node_in_path to suppress itself
            when a more specific node is already present (e.g. department_name
            subquery suppressed when Department node is in path because
            Department.filterable_columns.name already generates that key).
          - Edge-level filter_supplements on JoinStep: apply when that edge
            was traversed (i.e. the edge appears in join_chain).
        """
        fmap: Dict[str, Dict[str, Any]] = {}
        if join_chain is None:
            join_chain = []

        for entity in path:
            try:
                node = self._graph.get_node_data(entity)
            except Exception:
                continue

            entity_lower = entity.lower()
            filterable = node.get("filterable_columns", {})

            for col_key, col_meta in filterable.items():
                # Canonical filter key: <entity>_<column>
                # e.g. Employee.name  → employee_name
                #      Project.status → project_status
                filter_key = f"{entity_lower}_{col_key}"
                fmap[filter_key] = {
                    "sql": col_meta["sql_column"],
                    "match": col_meta.get("match_type", "like"),
                    "param_key": filter_key,
                }

            # T1-D: node-level filter_supplements — apply when this node is in path,
            # subject to exclude_when_node_in_path suppression.
            path_set = set(path)
            for supp_key, supp_meta in node.get("filter_supplements", {}).items():
                excluded_nodes = supp_meta.get("exclude_when_node_in_path", [])
                if any(n in path_set for n in excluded_nodes):
                    continue  # more specific node is present — its filterable_column handles this key
                fmap[supp_key] = {
                    "sql": supp_meta["sql"],
                    "match": supp_meta["match"],
                    "param_key": supp_key,
                }

        # T1-D: edge-level filter_supplements — apply when that edge was traversed.
        for step in join_chain:
            for supp_key, supp_meta in step.filter_supplements.items():
                fmap[supp_key] = {
                    "sql": supp_meta["sql"],
                    "match": supp_meta["match"],
                    "param_key": supp_key,
                }

        return fmap

    def _build_where(
        self,
        filters: Dict[str, Any],
        path: List[str],
        joined_tables: Set[str],
        join_chain: Optional[List[JoinStep]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Schema-driven WHERE builder.

        Iterates extraction.filters and matches each key against the
        dynamic filter map built from graph_schema.yaml.  No hardcoded
        elif chains — adding a new filterable_column to the schema is
        sufficient to make it filterable here automatically.
        T1-D: join_chain passed through so _build_filter_map can read
        edge-level filter_supplements.
        """
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        fmap = self._build_filter_map(path, join_chain)

        for filter_key, filter_value in filters.items():
            if not filter_value:
                continue
            filter_value = str(filter_value)

            if filter_key not in fmap:
                logger.warning(
                    "Filter key '%s' not in schema filter map for path %s — skipped",
                    filter_key, path,
                )
                continue

            entry = fmap[filter_key]
            match_type = entry["match"]
            sql_col = entry["sql"]
            param_key = entry["param_key"]

            if match_type == "subquery":
                # The sql already contains the full condition including param
                conditions.append(sql_col)
                params[param_key] = f"%{filter_value}%"

            elif match_type == "like":
                conditions.append(f"{sql_col} LIKE %({param_key})s")
                params[param_key] = f"%{filter_value}%"

            elif match_type == "exact":
                conditions.append(f"{sql_col} = %({param_key})s")
                params[param_key] = filter_value

            else:
                logger.warning(
                    "Unknown match_type '%s' for filter '%s' — skipped",
                    match_type, filter_key,
                )

        return " AND ".join(conditions), params

    def _build_order_limit(
        self,
        q_type: str,
        entities: List[str],
        filters: Dict[str, Any],
    ) -> str:
        if q_type == "comparison":
            if "Employee" in entities:
                has_scope_filter = any(
                    k in filters for k in ["department_name", "project_name", "manager_name"]
                )
                return "ORDER BY e.salary DESC" if has_scope_filter else "ORDER BY e.salary DESC LIMIT 10"
            if "Product" in entities:
                return "ORDER BY p.price DESC LIMIT 10"
            return "ORDER BY 1 DESC LIMIT 10"

        if q_type == "aggregation":
            if "Employee" in entities and "Department" in entities:
                return "GROUP BY d.id, d.name ORDER BY employee_count DESC"
            if "Employee" in entities and "Project" in entities:
                return "GROUP BY proj.id, proj.name ORDER BY employee_count DESC"
            if "Order" in entities:
                return "GROUP BY o.status ORDER BY order_count DESC"
            return "GROUP BY 1"

        if q_type == "list":
            # D6b: Order anchor must take priority over Employee — expanded Order
            # paths include Employee but should still sort by order date, not name.
            if "Order" in entities:
                return "ORDER BY o.order_date DESC LIMIT 20"
            if "Employee" in entities:
                return "ORDER BY e.name ASC"
            if "Product" in entities:
                return "ORDER BY p.name ASC"
            if "Project" in entities:
                return "ORDER BY proj.name ASC"
            return "ORDER BY 1 ASC"

        if q_type == "cross_entity":
            if "Order" in entities:
                return "ORDER BY o.order_date DESC LIMIT 20"
            if "Employee" in entities:
                return "ORDER BY e.name ASC"
            return "ORDER BY 1 ASC"

        # lookup / default
        if "Order" in entities:
            return "ORDER BY o.order_date DESC LIMIT 20"
        return "LIMIT 10"

    def _build_description(
        self,
        entities: List[str],
        q_type: str,
        filters: Dict[str, Any],
        path: List[str],
    ) -> str:
        type_labels = {
            "lookup": "Lookup",
            "list": "List all",
            "comparison": "Compare/Rank",
            "aggregation": "Aggregate",
            "cross_entity": "Cross-entity join",
            "other": "Unknown",
        }

        label = type_labels.get(q_type, q_type)
        entity_str = " + ".join(entities)
        filter_str = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
        path_str = " → ".join(path)

        desc = f"{label}: {entity_str}"
        if filter_str:
            desc += f" [{filter_str}]"
        desc += f" via {path_str}"

        return desc
