from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from middleware.models import EntityExtractionResult, GraphTraversal, QueryTemplate
from middleware.semantic_graph import SemanticGraph, JoinStep

logger = logging.getLogger(__name__)


class QueryBuildError(Exception):
    pass


# ---------------------------------------------------------------------------
# Schema loader — used by new query types to read metadata at build time.
# Intentionally separate from entity_extractor's cache — the builder runs
# in a different call context and doesn't share module-level globals.
# ---------------------------------------------------------------------------
_schema_cache: Optional[Dict[str, Any]] = None


def _load_schema() -> Dict[str, Any]:
    global _schema_cache
    if _schema_cache is None:
        schema_path = Path(__file__).parent.parent / "config" / "graph_schema.yaml"
        with open(schema_path, "r") as f:
            _schema_cache = yaml.safe_load(f)
    return _schema_cache


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

        # ── Route new question_types to dedicated builders ─────────────────
        # Each new type builds a completely self-contained SQL string and
        # returns immediately — they do not share the DISTINCT SELECT path.

        if q_type == "having_count":
            return self._build_having_count_query(entities, filters, path)

        if q_type == "temporal_filter":
            return self._build_temporal_filter_query(entities, filters, path, join_chain)

        if q_type == "computed_delta":
            return self._build_computed_delta_query(entities, filters, path, join_chain)

        # ── Project manager lookup — "who manages X project" ──────────────
        # Must be checked BEFORE the generic path so that manager-intent
        # questions never fall through to the project_assignments JOIN.
        if filters.get("project_manager_only") == "true":
            return self._build_project_manager_query(entities, filters, path)

        # ── Existing types — unchanged path ───────────────────────────────
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

        for step in join_chain:
            if step.manager_join and "Project" not in path:
                mgr_col = "e.name AS manager_name"
                if mgr_col not in select_parts:
                    select_parts = [
                        mgr_col if p == "e.name AS employee_name" else p
                        for p in select_parts
                    ]
                    if mgr_col not in select_parts:
                        select_parts.append(mgr_col)

        where_clause, where_params = self._build_where(filters, path, joined_tables, join_chain)
        order_limit = self._build_order_limit(q_type, entities, filters)

        if q_type == "group_rank":
            sql_parts = [f"SELECT {', '.join(select_parts)}", from_clause]
            sql_parts.extend(join_clauses)
            if where_clause:
                sql_parts.append(f"WHERE {where_clause}")
            sql_parts.append(self._build_group_rank_order(entities))
        else:
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

    # =========================================================================
    # NEW: having_count query builder
    # =========================================================================
    # Reads having_count_defaults from the anchor entity node in graph_schema.yaml.
    # Produces:
    #   SELECT e.name AS employee_name, COUNT(DISTINCT proj.id) AS project_count, ...
    #   FROM employees e
    #   INNER JOIN project_assignments pa ON e.id = pa.employee_id
    #   INNER JOIN projects proj ON pa.project_id = proj.id
    #   INNER JOIN departments d ON e.department_id = d.id
    #   [WHERE optional_filters]
    #   GROUP BY e.id, e.name
    #   HAVING COUNT(DISTINCT proj.id) > N
    #   ORDER BY project_count DESC
    #
    # The threshold N comes from filters["having_threshold"] (injected by the pre-filter).
    # Optional WHERE filters (e.g. department_name, project_status) are applied via
    # _build_where_for_having() which reads only the non-synthetic filter keys.
    # =========================================================================

    def _build_having_count_query(
        self,
        entities: List[str],
        filters: Dict[str, Any],
        path: List[str],
    ) -> Tuple[QueryTemplate, Dict[str, Any]]:
        anchor = entities[0] if entities else path[0]
        schema = _load_schema()
        node_data = schema["nodes"].get(anchor, {})
        hcd = node_data.get("having_count_defaults", {})

        if not hcd:
            raise QueryBuildError(
                f"having_count query requested for '{anchor}' but no "
                f"having_count_defaults declared in graph_schema.yaml"
            )

        # Extract threshold — pre-filter packed it as a string
        threshold = int(filters.get("having_threshold", hcd.get("default_threshold", 1)))

        # Build SELECT
        alias = node_data.get("alias", anchor.lower()[:1])
        label_sql   = hcd.get("label_sql",   f"{alias}.name")
        label_alias = hcd.get("label_alias", f"{anchor.lower()}_name")
        count_sql   = hcd.get("count_sql",   "COUNT(*)")
        count_alias = hcd.get("count_alias", "item_count")
        extra_select = hcd.get("extra_select", [])

        select_parts = [
            f"{label_sql} AS {label_alias}",
            f"{count_sql} AS {count_alias}",
        ] + extra_select

        # FROM + mandatory JOINs declared in the schema
        from_clause = f"FROM {node_data['table']} {alias}"
        join_sql_raw = hcd.get("join_sql", "")
        extra_joins  = hcd.get("extra_joins", [])

        join_lines = []
        if join_sql_raw:
            # join_sql may contain multiple JOIN clauses separated by spaces
            join_lines.append(join_sql_raw)
        for ej in extra_joins:
            join_lines.append(ej)

        # Optional WHERE from non-synthetic filters
        # Synthetic keys used only by this builder (not real columns):
        _SYNTHETIC_KEYS = {"having_threshold"}
        real_filters = {k: v for k, v in filters.items() if k not in _SYNTHETIC_KEYS}

        where_conditions: List[str] = []
        where_params: Dict[str, Any] = {}
        for fkey, fval in real_filters.items():
            if not fval:
                continue
            # Use the schema's fmap for all entities declared in having_count context
            # Path for WHERE purposes: anchor + joined entity
            fmap = self._build_filter_map(path, [])
            if fkey in fmap:
                entry = fmap[fkey]
                match_type = entry["match"]
                sql_col = entry["sql"]
                param_key = entry["param_key"]
                if match_type == "subquery":
                    where_conditions.append(sql_col)
                    where_params[param_key] = f"%{fval}%"
                elif match_type == "like":
                    where_conditions.append(f"{sql_col} LIKE %({param_key})s")
                    where_params[param_key] = f"%{fval}%"
                elif match_type == "exact":
                    where_conditions.append(f"{sql_col} = %({param_key})s")
                    where_params[param_key] = fval

        # GROUP BY on the anchor's primary key + label
        pk_sql = node_data.get("primary_key_sql", f"{alias}.id")
        group_by = f"GROUP BY {pk_sql}, {label_sql}"

        # HAVING
        having = f"HAVING {count_sql} > {threshold}"

        # ORDER BY
        order_by = f"ORDER BY {count_alias} DESC"

        sql_parts = [
            f"SELECT {', '.join(select_parts)}",
            from_clause,
        ] + join_lines
        if where_conditions:
            sql_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        sql_parts.extend([group_by, having, order_by])

        sql = "\n".join(p for p in sql_parts if p.strip())
        description = (
            f"Having count: {anchor} with more than {threshold} "
            f"{hcd.get('join_entity', 'items')}"
        )

        template = QueryTemplate(
            intent_name=f"graph:having_count:{anchor}",
            description=description,
            sql_template=sql,
            required_params=[],
            optional_params=[],
            result_description=description,
        )
        logger.debug("having_count query built: threshold=%d anchor=%s", threshold, anchor)
        return template, where_params

    # =========================================================================
    # NEW: temporal_filter query builder
    # =========================================================================
    # The pre-filter packed all metadata into synthetic filter keys:
    #   <param_key>               : the date value string (YYYY-MM-DD)
    #   <param_key>_direction     : "after" or "before"
    #   <param_key>_column_sql    : the SQL column expression (e.g. "proj.start_date")
    #   <param_key>_after_op      : SQL operator for "after" (e.g. ">=")
    #   <param_key>_before_op     : SQL operator for "before" (e.g. "<=")
    #
    # This builder reads those keys, constructs a WHERE clause, and emits a
    # standard SELECT DISTINCT query with appropriate ORDER BY.
    # =========================================================================

    def _build_temporal_filter_query(
        self,
        entities: List[str],
        filters: Dict[str, Any],
        path: List[str],
        join_chain: List[JoinStep],
    ) -> Tuple[QueryTemplate, Dict[str, Any]]:
        anchor = entities[0] if entities else path[0]

        # Find the synthetic temporal keys from pre-filter
        # Convention: any filter key ending in "_direction" is a temporal marker
        param_key: Optional[str] = None
        for key in filters:
            if key.endswith("_direction"):
                param_key = key[: -len("_direction")]
                break

        if param_key is None:
            raise QueryBuildError(
                "temporal_filter query has no <param_key>_direction filter — "
                "pre-filter did not inject metadata correctly"
            )

        date_val    = filters.get(param_key, "")
        direction   = filters.get(f"{param_key}_direction", "after")
        column_sql  = filters.get(f"{param_key}_column_sql", "")
        after_op    = filters.get(f"{param_key}_after_op",  ">=")
        before_op   = filters.get(f"{param_key}_before_op", "<=")

        if not date_val or not column_sql:
            raise QueryBuildError(
                f"temporal_filter query missing date value or column SQL "
                f"(param_key={param_key})"
            )

        operator = after_op if direction == "after" else before_op

        # Synthetic keys — strip from real WHERE params
        _SYNTHETIC_KEYS = {
            param_key,
            f"{param_key}_direction",
            f"{param_key}_column_sql",
            f"{param_key}_after_op",
            f"{param_key}_before_op",
        }

        # Temporal WHERE condition
        temporal_condition = f"{column_sql} {operator} %({param_key})s"
        where_params: Dict[str, Any] = {param_key: date_val}

        # Standard SELECT for the anchor entity
        schema = _load_schema()
        node_data = schema["nodes"].get(anchor, {})
        alias = node_data.get("alias", anchor.lower()[:1])

        select_parts = []
        for col_def in node_data.get("selectable_columns", []):
            if col_def.get("is_hidden"):
                continue
            if col_def.get("default_hidden"):
                continue
            col      = col_def["column"]
            col_alias = col_def["alias"]
            if col_def.get("is_subquery"):
                select_parts.append(f"{col} AS {col_alias}")
            else:
                select_parts.append(f"{alias}.{col} AS {col_alias}")

        if not select_parts:
            select_parts = ["*"]

        from_clause = f"FROM {node_data['table']} {alias}"
        join_clauses = self._build_joins(join_chain)

        # Add manager join if Project is in path
        if "Project" in path:
            mgr_join = "LEFT JOIN employees mgr ON mgr.id = proj.manager_id"
            if mgr_join not in join_clauses:
                join_clauses.append(mgr_join)
            mgr_col = "mgr.name AS manager_name"
            if mgr_col not in select_parts:
                select_parts = [p for p in select_parts if "manager_id" not in p]
                select_parts.append(mgr_col)

        # Additional real filters (not synthetic temporal keys)
        real_filters = {k: v for k, v in filters.items() if k not in _SYNTHETIC_KEYS}
        joined_tables = self._get_joined_tables(path, join_chain)
        extra_where, extra_params = self._build_where(real_filters, path, joined_tables, join_chain)
        where_params.update(extra_params)

        # Combine WHERE conditions
        all_conditions = [temporal_condition]
        if extra_where:
            all_conditions.append(extra_where)
        where_clause = " AND ".join(all_conditions)

        # ORDER BY — use the temporal column itself for natural ordering
        if direction == "after":
            order_by = f"ORDER BY {column_sql} ASC LIMIT 50"
        else:
            order_by = f"ORDER BY {column_sql} DESC LIMIT 50"

        sql_parts = [
            f"SELECT DISTINCT {', '.join(select_parts)}",
            from_clause,
        ]
        sql_parts.extend(join_clauses)
        sql_parts.append(f"WHERE {where_clause}")
        sql_parts.append(order_by)

        sql = "\n".join(p for p in sql_parts if p.strip())
        col_label = param_key.replace("_val", "").replace("_", " ")
        description = (
            f"Temporal filter: {anchor} where {col_label} "
            f"{direction} {date_val}"
        )

        template = QueryTemplate(
            intent_name=f"graph:temporal_filter:{anchor}",
            description=description,
            sql_template=sql,
            required_params=[param_key],
            optional_params=[],
            result_description=description,
        )
        logger.debug(
            "temporal_filter query built: anchor=%s col=%s op=%s date=%s",
            anchor, column_sql, operator, date_val,
        )
        return template, where_params

    # =========================================================================
    # NEW: computed_delta query builder
    # =========================================================================
    # The pre-filter packed metadata into synthetic filter keys:
    #   delta_column_sql  : e.g. "e.salary"
    #   delta_max_alias   : e.g. "highest_salary"
    #   delta_min_alias   : e.g. "lowest_salary"
    #   delta_alias       : e.g. "salary_gap"
    #   delta_is_currency : "True" or "False"
    #
    # Produces a single-row result:
    #   SELECT MAX(col) AS max_alias, MIN(col) AS min_alias,
    #          (MAX(col) - MIN(col)) AS delta_alias
    #   FROM table [alias]
    #   [JOIN …]
    #   [WHERE optional_filters]
    #
    # Optional WHERE filters (e.g. department_name for scoped salary delta)
    # are applied via _build_where() for any non-synthetic keys.
    # =========================================================================

    def _build_computed_delta_query(
        self,
        entities: List[str],
        filters: Dict[str, Any],
        path: List[str],
        join_chain: List[JoinStep],
    ) -> Tuple[QueryTemplate, Dict[str, Any]]:
        anchor = entities[0] if entities else path[0]

        # Read metadata from synthetic filter keys
        _SYNTHETIC_KEYS = {
            "delta_column_sql", "delta_max_alias", "delta_min_alias",
            "delta_alias", "delta_is_currency",
        }

        col_sql    = filters.get("delta_column_sql", "")
        max_alias  = filters.get("delta_max_alias",  "max_value")
        min_alias  = filters.get("delta_min_alias",  "min_value")
        delta_alias = filters.get("delta_alias",     "delta")

        if not col_sql:
            raise QueryBuildError(
                "computed_delta query missing delta_column_sql — "
                "pre-filter did not inject metadata correctly"
            )

        schema = _load_schema()
        node_data = schema["nodes"].get(anchor, {})
        alias = node_data.get("alias", anchor.lower()[:1])

        # SELECT: MAX, MIN, delta expression
        select_parts = [
            f"MAX({col_sql}) AS {max_alias}",
            f"MIN({col_sql}) AS {min_alias}",
            f"(MAX({col_sql}) - MIN({col_sql})) AS {delta_alias}",
        ]

        from_clause  = f"FROM {node_data['table']} {alias}"
        join_clauses = self._build_joins(join_chain)

        # Optional real WHERE filters
        real_filters = {k: v for k, v in filters.items() if k not in _SYNTHETIC_KEYS}
        joined_tables = self._get_joined_tables(path, join_chain)
        where_clause, where_params = self._build_where(
            real_filters, path, joined_tables, join_chain
        )

        sql_parts = [
            f"SELECT {', '.join(select_parts)}",
            from_clause,
        ]
        sql_parts.extend(join_clauses)
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")

        sql = "\n".join(p for p in sql_parts if p.strip())
        description = (
            f"Computed delta: {anchor} {delta_alias}"
            + (f" [{', '.join(f'{k}={v}' for k,v in real_filters.items() if v)}]"
               if real_filters else "")
        )

        template = QueryTemplate(
            intent_name=f"graph:computed_delta:{anchor}",
            description=description,
            sql_template=sql,
            required_params=[],
            optional_params=[],
            result_description=description,
        )
        logger.debug(
            "computed_delta query built: anchor=%s col=%s delta_alias=%s",
            anchor, col_sql, delta_alias,
        )
        return template, where_params

    # =========================================================================
    # NEW: project_manager_query builder
    # =========================================================================
    # Handles "who manages / who leads / who is in charge of [project]".
    #
    # Triggered when filters["project_manager_only"] == "true".
    # Bypasses project_assignments entirely — queries proj.manager_id directly.
    #
    # Produces:
    #   SELECT mgr.name AS manager_name,
    #          proj.name AS project_name,
    #          proj.status AS project_status,
    #          proj.budget AS project_budget
    #   FROM projects proj
    #   JOIN employees mgr ON mgr.id = proj.manager_id
    #   WHERE proj.name LIKE %(project_name)s
    #
    # Returns exactly 1 row: the project manager.
    # =========================================================================

    def _build_project_manager_query(
        self,
        entities: List[str],
        filters: Dict[str, Any],
        path: List[str],
    ) -> Tuple[QueryTemplate, Dict[str, Any]]:
        _SYNTHETIC_KEYS = {"project_manager_only"}

        # Real filters — only project_name is expected here
        real_filters = {k: v for k, v in filters.items() if k not in _SYNTHETIC_KEYS}

        where_conditions: List[str] = []
        where_params: Dict[str, Any] = {}

        if "project_name" in real_filters:
            where_conditions.append("proj.name LIKE %(project_name)s")
            where_params["project_name"] = f"%{real_filters['project_name']}%"

        where_clause = " AND ".join(where_conditions)

        sql_parts = [
            "SELECT mgr.name AS manager_name,",
            "       proj.name AS project_name,",
            "       proj.status AS project_status,",
            "       proj.budget AS project_budget",
            "FROM projects proj",
            "JOIN employees mgr ON mgr.id = proj.manager_id",
        ]
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")

        sql = "\n".join(sql_parts)
        project_label = real_filters.get("project_name", "project")
        description = f"Project manager lookup: who manages '{project_label}'"

        template = QueryTemplate(
            intent_name="graph:project_manager_lookup",
            description=description,
            sql_template=sql,
            required_params=[],
            optional_params=["project_name"],
            result_description=description,
        )
        logger.debug(
            "project_manager_query built: project_name=%s",
            real_filters.get("project_name"),
        )
        return template, where_params

    # =========================================================================
    # Existing methods — unchanged
    # =========================================================================

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
        if q_type == "group_rank":
            return self._build_group_rank_select(entities)

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
        schema = _load_schema()

        if "Employee" in entities and "Department" in entities:
            emp_node = schema.get("nodes", {}).get("Employee", {})
            agg = emp_node.get("aggregation_defaults", {})
            sum_cols  = agg.get("sum_columns",  [])
            avg_cols  = agg.get("avg_columns",  [])
            max_cols  = agg.get("max_columns",  [])
            min_cols  = agg.get("min_columns",  [])
            parts = ["d.name AS department_name", "COUNT(e.id) AS employee_count"]
            for col in sum_cols:
                alias = col.replace("e.", "").replace(".", "_") + "_total"
                parts.append(f"SUM({col}) AS {alias}")
            for col in avg_cols:
                alias = col.replace("e.", "").replace(".", "_") + "_avg"
                parts.append(f"AVG({col}) AS {alias}")
            for col in max_cols:
                alias = col.replace("e.", "").replace(".", "_") + "_max"
                parts.append(f"MAX({col}) AS {alias}")
            for col in min_cols:
                alias = col.replace("e.", "").replace(".", "_") + "_min"
                parts.append(f"MIN({col}) AS {alias}")
            return parts if len(parts) > 2 else [
                "d.name AS department_name",
                "COUNT(e.id) AS employee_count",
                "SUM(e.salary) AS salary_total",
                "AVG(e.salary) AS salary_avg",
                "MAX(e.salary) AS salary_max",
                "MIN(e.salary) AS salary_min",
            ]

        if "Employee" in entities and "Project" in entities:
            return ["proj.name AS project_name", "COUNT(e.id) AS employee_count"]
        if "Order" in entities:
            return ["o.status", "COUNT(o.id) AS order_count", "SUM(o.quantity) AS total_quantity"]
        if "Employee" in entities:
            return ["COUNT(e.id) AS employee_count"]
        return ["COUNT(*) AS total_count"]

    def _build_group_rank_select(self, entities: List[str]) -> List[str]:
        anchor = entities[0] if entities else None
        if not anchor:
            return ["COUNT(*) AS total_count"]
        try:
            node = self._graph.get_node_data(anchor)
        except Exception:
            return ["COUNT(*) AS total_count"]
        grd = node.get("group_rank_defaults", {})
        if not grd:
            return ["COUNT(*) AS total_count"]

        group_col   = grd.get("group_by_column", "1")
        group_alias = grd.get("group_by_alias", "group_value")
        count_col   = grd.get("count_column", "*")
        count_alias = grd.get("count_alias", "item_count")
        extra       = grd.get("extra_select", [])

        parts = [f"{group_col} AS {group_alias}"]
        parts.append(f"COUNT({count_col}) AS {count_alias}")
        for ex in extra:
            parts.append(ex)
        return parts

    def _build_group_rank_order(self, entities: List[str]) -> str:
        anchor = entities[0] if entities else None
        if not anchor:
            return "GROUP BY 1 ORDER BY item_count DESC LIMIT 10"
        try:
            node = self._graph.get_node_data(anchor)
        except Exception:
            return "GROUP BY 1 ORDER BY item_count DESC LIMIT 10"
        grd = node.get("group_rank_defaults", {})
        if not grd:
            return "GROUP BY 1 ORDER BY item_count DESC LIMIT 10"

        group_col   = grd.get("group_by_column", "1")
        count_alias = grd.get("count_alias", "item_count")
        direction   = grd.get("order_direction", "DESC")
        limit       = grd.get("limit", 10)
        return f"GROUP BY {group_col} ORDER BY {count_alias} {direction} LIMIT {limit}"

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
                filter_key = f"{entity_lower}_{col_key}"
                fmap[filter_key] = {
                    "sql": col_meta["sql_column"],
                    "match": col_meta.get("match_type", "like"),
                    "param_key": filter_key,
                }

                sqd = col_meta.get("scalar_subquery_defaults")
                if sqd:
                    sq_filter_key = f"{entity_lower}_{col_key}_avg_compare"
                    fmap[sq_filter_key] = {
                        "sql": col_meta["sql_column"],
                        "match": "scalar_subquery",
                        "param_key": sq_filter_key,
                        "scalar_sql": sqd.get("scalar_sql", ""),
                        "above_operator": sqd.get("above_operator", ">"),
                        "below_operator": sqd.get("below_operator", "<"),
                    }

            path_set = set(path)
            for supp_key, supp_meta in node.get("filter_supplements", {}).items():
                excluded_nodes = supp_meta.get("exclude_when_node_in_path", [])
                if any(n in path_set for n in excluded_nodes):
                    continue
                fmap[supp_key] = {
                    "sql": supp_meta["sql"],
                    "match": supp_meta["match"],
                    "param_key": supp_key,
                }

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
                conditions.append(sql_col)
                params[param_key] = f"%{filter_value}%"

            elif match_type == "scalar_subquery":
                scalar_sql = entry.get("scalar_sql", "")
                if filter_value == "above":
                    operator = entry.get("above_operator", ">")
                else:
                    operator = entry.get("below_operator", "<")
                if scalar_sql:
                    conditions.append(f"{sql_col} {operator} {scalar_sql}")

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
                return "GROUP BY d.id, d.name ORDER BY salary_total DESC"
            if "Employee" in entities and "Project" in entities:
                return "GROUP BY proj.id, proj.name ORDER BY employee_count DESC"
            if "Order" in entities:
                return "GROUP BY o.status ORDER BY order_count DESC"
            return "GROUP BY 1"

        if q_type == "list":
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
            "group_rank": "Group Rank",
            "having_count": "Having Count",
            "temporal_filter": "Temporal Filter",
            "computed_delta": "Computed Delta",
            "other": "Unknown",
        }

        label = type_labels.get(q_type, q_type)
        entity_str = " + ".join(entities)
        # Skip synthetic filter keys in the description
        _SYNTHETIC_PREFIXES = ("delta_", "having_threshold")
        filter_str = ", ".join(
            f"{k}={v}" for k, v in filters.items()
            if v and not any(k.startswith(p) for p in _SYNTHETIC_PREFIXES)
            and not k.endswith(("_direction", "_column_sql", "_after_op", "_before_op"))
        )
        path_str = " → ".join(path)

        desc = f"{label}: {entity_str}"
        if filter_str:
            desc += f" [{filter_str}]"
        desc += f" via {path_str}"

        return desc
