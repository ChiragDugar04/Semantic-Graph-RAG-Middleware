"""
middleware/graph_query_builder.py

Converts a graph traversal path + entity extraction result into a
valid, parameterized SQL query — packaged as a QueryTemplate.

This is the "compiler" of the semantic graph system. It takes:
  - EntityExtractionResult  → what entities, filters, question_type
  - List[JoinStep]          → how to JOIN the tables (from semantic_graph)
  - SemanticGraph           → node metadata (table names, aliases, columns)

And produces a QueryTemplate that the existing query_executor.py
can run without any changes. The rest of the pipeline (execution,
formatting, synthesis) is completely unchanged.

SQL Construction Rules:
  - Single entity, no joins    → SELECT ... FROM table WHERE ...
  - Two-hop, one join          → SELECT ... FROM t1 JOIN t2 ON ...
  - Junction table (many-many) → SELECT ... FROM t1 JOIN junction JOIN t2
  - question_type=comparison   → ORDER BY target_column DESC
  - question_type=aggregation  → GROUP BY + COUNT/AVG
  - question_type=list         → ORDER BY name ASC
  - question_type=lookup       → LIMIT 10

Security: ALL user values go through %(param)s placeholders.
The LLM never writes SQL. This module writes SQL using graph metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from middleware.models import EntityExtractionResult, GraphTraversal, QueryTemplate
from middleware.semantic_graph import SemanticGraph, JoinStep


# ============================================================
# QUERY BUILDER
# ============================================================

class GraphQueryBuilder:
    """Builds parameterized SQL from graph traversal metadata.

    Instantiated once and reused. All state is passed per-call.
    """

    def __init__(self, graph: SemanticGraph) -> None:
        """
        Args:
            graph: The SemanticGraph instance (loaded once at startup).
        """
        self._graph = graph

    # ----------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ----------------------------------------------------------

    def build_query(
        self,
        extraction: EntityExtractionResult,
        traversal: GraphTraversal,
        join_chain: List[JoinStep],
    ) -> QueryTemplate:
        """Build a complete QueryTemplate from graph traversal data.

        Args:
            extraction : The EntityExtractionResult from entity_extractor.
            traversal  : The GraphTraversal with path and metadata.
            join_chain : Ordered list of JoinStep from semantic_graph.

        Returns:
            QueryTemplate: Ready to pass to query_executor.execute_query().
        """
        entities    = extraction.entities
        filters     = extraction.filters
        q_type      = extraction.question_type
        path        = traversal.path_taken

        # ── Determine anchor entity (primary FROM table) ──────
        # Use the traversal PATH's first node, not extraction.entities[0].
        # The pipeline normalizer may have reordered entities for correct
        # JOIN direction (e.g. Employee→Department not Department→Employee)
        # but extraction.entities still holds the original order.
        # The path is always correctly ordered by the pipeline.
        anchor = path[0] if path else (entities[0] if entities else "Employee")

        # ── Build SELECT clause ───────────────────────────────
        select_parts, params = self._build_select(
            entities, filters, q_type, join_chain
        )

        # ── Build FROM clause ─────────────────────────────────
        from_clause = self._build_from(anchor)

        # ── Build JOIN clauses ────────────────────────────────
        join_clauses = self._build_joins(join_chain)

        # ── Build WHERE clause ────────────────────────────────
        where_clause, where_params = self._build_where(filters, entities)
        params.update(where_params)

        # ── Build ORDER BY + LIMIT ────────────────────────────
        order_limit = self._build_order_limit(q_type, entities, filters)

        # ── Assemble SQL ──────────────────────────────────────
        sql_parts = [
            f"SELECT {', '.join(select_parts)}",
            from_clause,
        ]
        if join_clauses:
            sql_parts.extend(join_clauses)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
        sql_parts.append(order_limit)

        sql = "\n".join(sql_parts)

        # ── Build description ─────────────────────────────────
        description = self._build_description(entities, q_type, filters, path)

        return QueryTemplate(
            intent_name=f"graph:{'→'.join(path)}",
            description=description,
            sql_template=sql,
            required_params=[],       # params already embedded safely
            optional_params=[],
            result_description=description,
        ), params

    # ----------------------------------------------------------
    # SELECT BUILDER
    # ----------------------------------------------------------

    def _build_select(
        self,
        entities: List[str],
        filters: Dict[str, Any],
        q_type: str,
        join_chain: List[JoinStep],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Build SELECT column list.

        For aggregation queries, wraps in COUNT/AVG.
        For all others, collects meaningful columns from each entity.

        Returns:
            Tuple of (select_parts list, extra_params dict)
        """
        parts: List[str] = []
        extra_params: Dict[str, Any] = {}

        if q_type == "aggregation":
            # Count employees per department, etc.
            if "Employee" in entities and "Department" in entities:
                parts = [
                    "d.name AS department_name",
                    "COUNT(e.id) AS employee_count",
                    "AVG(e.salary) AS avg_salary",
                    "MAX(e.salary) AS max_salary",
                    "MIN(e.salary) AS min_salary",
                ]
            elif "Employee" in entities and "Project" in entities:
                parts = [
                    "proj.name AS project_name",
                    "COUNT(e.id) AS employee_count",
                ]
            elif "Employee" in entities:
                parts = ["COUNT(e.id) AS employee_count"]
            elif "Order" in entities:
                parts = [
                    "o.status",
                    "COUNT(o.id) AS order_count",
                    "SUM(o.quantity) AS total_quantity",
                ]
            else:
                parts = ["COUNT(*) AS total_count"]
            return parts, extra_params

        # ── Non-aggregation: collect columns per entity ───────
        seen_tables: set = set()

        for entity in entities:
            if entity in seen_tables:
                continue
            seen_tables.add(entity)

            node      = self._graph.get_node_data(entity)
            alias     = node["alias"]
            sel_cols  = node.get("selectable_columns", [])

            for col_def in sel_cols:
                col     = col_def["column"]
                col_alias = col_def["alias"]
                parts.append(f"{alias}.{col} AS {col_alias}")

        # ── Add junction table extras (e.g. assignment_role) ──
        for step in join_chain:
            for extra in step.extra_select:
                if extra not in parts:
                    parts.append(extra)

        # Fallback
        if not parts:
            parts = ["*"]

        return parts, extra_params

    # ----------------------------------------------------------
    # FROM BUILDER
    # ----------------------------------------------------------

    def _build_from(self, anchor_entity: str) -> str:
        """Build the FROM clause using the anchor entity's table + alias.

        Args:
            anchor_entity: The primary entity (first in path).

        Returns:
            str: e.g. "FROM employees e"
        """
        node  = self._graph.get_node_data(anchor_entity)
        table = node["table"]
        alias = node["alias"]
        return f"FROM {table} {alias}"

    # ----------------------------------------------------------
    # JOIN BUILDER
    # ----------------------------------------------------------

    def _build_joins(self, join_chain: List[JoinStep]) -> List[str]:
        """Build JOIN clauses from the ordered JoinStep list.

        Handles:
          - Simple FK joins: JOIN table alias ON condition
          - Junction table joins: JOIN junction pa ON ..., JOIN target ON ...

        Args:
            join_chain: List of JoinStep from semantic_graph.get_join_chain().

        Returns:
            List[str]: One or more JOIN clause strings.
        """
        joins: List[str] = []
        joined_tables: set = set()

        for step in join_chain:
            to_node = step.to_node
            node    = self._graph.get_node_data(to_node)
            table   = node["table"]
            alias   = node["alias"]
            j_type  = step.join_type

            if step.junction_table:
                # Many-to-many: two JOIN clauses
                jt     = step.junction_table
                jalias = step.junction_alias or jt[:2]

                # Parse the compound condition: "e.id = pa.employee_id AND pa.project_id = proj.id"
                # Split into two parts
                conditions = step.join_condition.split(" AND ")
                if len(conditions) == 2:
                    cond1 = conditions[0].strip()   # e.id = pa.employee_id
                    cond2 = conditions[1].strip()   # pa.project_id = proj.id
                else:
                    cond1 = step.join_condition
                    cond2 = f"{jalias}.{to_node.lower()}_id = {alias}.id"

                # JOIN junction table
                if jt not in joined_tables:
                    joins.append(f"{j_type} JOIN {jt} {jalias} ON {cond1}")
                    joined_tables.add(jt)

                # JOIN target table
                if table not in joined_tables:
                    joins.append(f"{j_type} JOIN {table} {alias} ON {cond2}")
                    joined_tables.add(table)

            else:
                # Simple FK join
                if table not in joined_tables:
                    joins.append(
                        f"{j_type} JOIN {table} {alias} ON {step.join_condition}"
                    )
                    joined_tables.add(table)

        return joins

    # ----------------------------------------------------------
    # WHERE BUILDER
    # ----------------------------------------------------------

    def _build_where(
        self,
        filters: Dict[str, Any],
        entities: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a parameterized WHERE clause from filters.

        Maps filter keys to SQL columns using the graph's
        filterable_columns metadata. Uses LIKE for name fields
        and = for exact match fields (status, category).

        Args:
            filters : Filter key → value dict from extraction.
            entities: Entities involved (to know which alias to use).

        Returns:
            Tuple of (where_string, params_dict).
            where_string is empty if no filters apply.
        """
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        # Map from filter key → (sql_column, match_type)
        filter_map: Dict[str, Tuple[str, str]] = {}

        for entity in entities:
            try:
                filterable = self._graph.get_filterable_columns(entity)
                for col_key, col_meta in filterable.items():
                    filter_map[col_key] = (
                        col_meta["sql_column"],
                        col_meta["match_type"],
                    )
            except Exception:
                continue

        # Also add known filter key aliases
        # IMPORTANT: The alias maps to the BASE column key (not the filter key)
        # so that query_executor._EXACT_MATCH_PARAMS {"status","category"} fires
        # correctly and does NOT wrap these values in % wildcards.
        _FILTER_KEY_ALIASES = {
            "employee_name":   "name",
            "department_name": "name",
            "product_name":    "name",
            "project_name":    "name",
            "order_status":    "status",    # resolves to exact-match "status"
            "project_status":  "status",    # resolves to exact-match "status"
            "employee_role":   "role",
        }

        # Keys that must use exact match (= not LIKE) regardless of graph metadata.
        # Mirrors query_executor._EXACT_MATCH_PARAMS so param names align.
        _FORCE_EXACT_KEYS = {"status", "category"}

        # Build a DIRECT filter_key → (sql_column, match_type) map
        # that respects which entity each filter_key belongs to.
        # This avoids the collision where both Employee.name and
        # Department.name resolve to the same base key "name".
        _DIRECT_FILTER_MAP: Dict[str, tuple] = {
            "employee_name":   ("e.name",    "like"),
            "department_name": ("d.name",    "like"),
            "product_name":    ("p.name",    "like"),
            "project_name":    ("proj.name", "like"),
            "order_status":    ("o.status",  "exact"),
            "project_status":  ("proj.status", "exact"),
            "employee_role":   ("e.role",    "like"),
            "category":        ("p.category","exact"),
            "manager_name":    ("proj.manager_id", "subquery"),  # special: subquery
        }

        for filter_key, filter_value in filters.items():
            if not filter_value:
                continue

            # Try direct map first — this is unambiguous
            if filter_key in _DIRECT_FILTER_MAP:
                sql_col, match_type = _DIRECT_FILTER_MAP[filter_key]
            else:
                # Fall back to graph filter_map via alias resolution
                resolved_key = _FILTER_KEY_ALIASES.get(filter_key, filter_key)
                sql_col    = None
                match_type = "like"

                if resolved_key in filter_map:
                    sql_col, match_type = filter_map[resolved_key]
                elif filter_key in filter_map:
                    sql_col, match_type = filter_map[filter_key]

                if sql_col is None:
                    continue

            # Force exact match for status/category fields
            resolved_key = _FILTER_KEY_ALIASES.get(filter_key, filter_key)
            if resolved_key in _FORCE_EXACT_KEYS:
                match_type = "exact"

            # Use resolved key as SQL param name so executor's
            # _EXACT_MATCH_PARAMS guard fires correctly for "status"/"category"
            safe_param = re.sub(r"[^a-z0-9_]", "_", resolved_key.lower())

            if match_type == "subquery":
                # manager_name → proj.manager_id IN (SELECT id FROM employees WHERE name LIKE ...)
                conditions.append(
                    f"proj.manager_id IN "
                    f"(SELECT id FROM employees WHERE name LIKE %({safe_param})s)"
                )
                params[safe_param] = f"%{filter_value}%"
            elif match_type == "like":
                conditions.append(f"{sql_col} LIKE %({safe_param})s")
                params[safe_param] = filter_value   # raw value — executor wraps with %
            else:
                conditions.append(f"{sql_col} = %({safe_param})s")
                params[safe_param] = filter_value

        return " AND ".join(conditions), params

    # ----------------------------------------------------------
    # ORDER BY + LIMIT BUILDER
    # ----------------------------------------------------------

    def _build_order_limit(
        self,
        q_type: str,
        entities: List[str],
        filters: Dict[str, Any],
    ) -> str:
        """Build ORDER BY and LIMIT clauses based on question type.

        Args:
            q_type  : Question type string.
            entities: Entities involved.
            filters : Applied filters (used to detect salary context).

        Returns:
            str: ORDER BY / GROUP BY / LIMIT clause(s).
        """
        if q_type == "comparison":
            # Detect what we're comparing
            if "Employee" in entities:
                if any(k in filters for k in ["department_name", "project_name"]):
                    return "ORDER BY e.salary DESC"
                return "ORDER BY e.salary DESC LIMIT 10"
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
            return "ORDER BY 1 ASC"

        if q_type == "cross_entity":
            if "Employee" in entities:
                return "ORDER BY e.name ASC"
            return "ORDER BY 1 ASC"

        # Default: lookup
        return "LIMIT 10"

    # ----------------------------------------------------------
    # DESCRIPTION BUILDER
    # ----------------------------------------------------------

    def _build_description(
        self,
        entities: List[str],
        q_type: str,
        filters: Dict[str, Any],
        path: List[str],
    ) -> str:
        """Generate a human-readable description of the query.

        Used in the Glass Box audit trail.

        Args:
            entities: Entity names involved.
            q_type  : Question type.
            filters : Applied filters.
            path    : Graph traversal path.

        Returns:
            str: Description for Glass Box display.
        """
        path_str = " → ".join(path)
        filter_str = ", ".join(
            f"{k}={v}" for k, v in filters.items() if v
        )

        type_labels = {
            "lookup":       "Lookup",
            "list":         "List all",
            "comparison":   "Compare/Rank",
            "aggregation":  "Aggregate",
            "cross_entity": "Cross-entity join",
            "other":        "Unknown",
        }

        label = type_labels.get(q_type, q_type)
        entity_str = " + ".join(entities)

        desc = f"{label}: {entity_str}"
        if filter_str:
            desc += f" [{filter_str}]"
        desc += f" via {path_str}"

        return desc