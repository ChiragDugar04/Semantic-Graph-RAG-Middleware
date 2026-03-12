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

        where_clause, where_params = self._build_where(filters, path, joined_tables)
        order_limit = self._build_order_limit(q_type, entities, filters)

        sql_parts = [f"SELECT {', '.join(select_parts)}", from_clause]
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

    def _build_where(
        self,
        filters: Dict[str, Any],
        path: List[str],
        joined_tables: Set[str],
    ) -> Tuple[str, Dict[str, Any]]:
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        has_employees = "Employee" in path
        has_departments = "Department" in path
        has_projects = "Project" in path
        has_products = "Product" in path
        has_orders = "Order" in path

        for filter_key, filter_value in filters.items():
            if not filter_value:
                continue

            filter_value = str(filter_value)

            if filter_key == "employee_name" and has_employees:
                conditions.append("e.name LIKE %(employee_name)s")
                params["employee_name"] = f"%{filter_value}%"

            elif filter_key == "employee_role" and has_employees:
                conditions.append("e.role LIKE %(employee_role)s")
                params["employee_role"] = f"%{filter_value}%"

            elif filter_key == "department_name":
                if has_departments:
                    conditions.append("d.name LIKE %(department_name)s")
                    params["department_name"] = f"%{filter_value}%"
                elif has_employees:
                    conditions.append(
                        "e.department_id IN "
                        "(SELECT id FROM departments WHERE name LIKE %(department_name)s)"
                    )
                    params["department_name"] = f"%{filter_value}%"
                else:
                    logger.warning(
                        "Filter 'department_name' cannot be applied: "
                        "neither Department nor Employee in path %s", path
                    )

            elif filter_key == "project_name" and has_projects:
                conditions.append("proj.name LIKE %(project_name)s")
                params["project_name"] = f"%{filter_value}%"

            elif filter_key == "project_status" and has_projects:
                conditions.append("proj.status = %(project_status)s")
                params["project_status"] = filter_value

            elif filter_key == "project_department" and has_projects:
                conditions.append(
                    "proj.department_id IN "
                    "(SELECT id FROM departments WHERE name LIKE %(project_department)s)"
                )
                params["project_department"] = f"%{filter_value}%"

            elif filter_key == "manager_name" and has_projects:
                conditions.append(
                    "proj.manager_id IN "
                    "(SELECT id FROM employees WHERE name LIKE %(manager_name)s)"
                )
                params["manager_name"] = f"%{filter_value}%"

            elif filter_key == "product_name" and has_products:
                conditions.append("p.name LIKE %(product_name)s")
                params["product_name"] = f"%{filter_value}%"

            elif filter_key == "category" and has_products:
                conditions.append("p.category = %(category)s")
                params["category"] = filter_value

            elif filter_key == "order_status" and has_orders:
                conditions.append("o.status = %(order_status)s")
                params["order_status"] = filter_value

            else:
                logger.debug(
                    "Filter key '%s' skipped — no matching table in path %s",
                    filter_key, path
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
            if "Employee" in entities:
                return "ORDER BY e.name ASC"
            if "Product" in entities:
                return "ORDER BY p.name ASC"
            if "Order" in entities:
                return "ORDER BY o.order_date DESC LIMIT 20"
            if "Project" in entities:
                return "ORDER BY proj.name ASC"
            return "ORDER BY 1 ASC"

        if q_type == "cross_entity":
            if "Employee" in entities:
                return "ORDER BY e.name ASC"
            return "ORDER BY 1 ASC"

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
