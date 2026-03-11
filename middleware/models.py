"""
middleware/models.py

Pydantic data models for the RAG Middleware pipeline.
These are the strict data contracts between every component.

Phase 2 additions (marked NEW):
  - EntityExtractionResult  replaces ExtractedParameters semantically
  - GraphTraversal          carries path + join metadata
  - MiddlewareTrace         extended with graph fields (backward-compatible)

All OLD models kept exactly as-is so the Glass Box and existing
query_executor.py require zero changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# INTENT CLASSIFICATION
# Kept exactly as Phase 1 — used to carry graph path info
# in the Glass Box (intent_name stores the path string)
# ============================================================

class IntentClassification(BaseModel):
    """Result of classifying a user question into a known intent.

    In the new semantic pipeline, intent_name carries the graph
    traversal path string (e.g. "Employee→Department") instead
    of a YAML intent key. All other fields are unchanged.
    """
    model_config = ConfigDict(protected_namespaces=())
    intent_name: str = Field(..., description="Graph path or intent key")
    confidence: str = Field(default="medium")
    raw_llm_response: str = Field(default="")
    model_used: str = Field(default="")
    latency_ms: float = Field(default=0.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        return v if v in allowed else "medium"


# ============================================================
# EXTRACTED PARAMETERS
# Kept exactly as Phase 1 — still used by query_executor.py
# In the new pipeline, params carries the filters dict
# ============================================================

class ExtractedParameters(BaseModel):
    """Named entities and values extracted from the user's question.

    In the new semantic pipeline, params carries the filters dict
    from EntityExtractionResult. The Glass Box panel ② renders this.
    """
    params: Dict[str, Any] = Field(default_factory=dict)
    missing_required: List[str] = Field(default_factory=list)
    extraction_method: str = Field(default="none")
    latency_ms: float = Field(default=0.0)


# ============================================================
# QUERY TEMPLATE
# Unchanged — query_executor.py depends on this exactly
# ============================================================

class QueryTemplate(BaseModel):
    """A pre-written or dynamically-built SQL template."""
    intent_name: str
    description: str
    sql_template: str
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)
    result_description: str = Field(default="")


# ============================================================
# DATABASE RESULT
# Unchanged — query_executor.py produces this exactly
# ============================================================

class DBResult(BaseModel):
    """Result of executing a SQL query against MySQL."""
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    execution_time_ms: float = Field(default=0.0)
    query_executed: str = Field(default="")
    params_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    self_healing_triggered: bool = Field(default=False)
    healing_reason: str = Field(default="")


# ============================================================
# NEW — ENTITY EXTRACTION RESULT
# Output of entity_extractor.py
# Replaces the intent_classifier + param_extractor conceptually
# ============================================================

class EntityExtractionResult(BaseModel):
    """Entities, filters, and question type extracted from the question.

    Attributes:
        entities        : List of entity names involved (e.g. ["Employee", "Department"])
        filters         : Dict of filter_key → value (e.g. {"employee_name": "Sarah"})
        projections     : Column names the user wants to see (e.g. ["salary", "role"])
        question_type   : One of: lookup / list / comparison / aggregation / cross_entity / other
        extraction_method: "rules" (Tier 1) or "llm" (Tier 2) or "llm_error"
        latency_ms      : Time taken for extraction
    """
    entities: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    projections: List[str] = Field(default_factory=list)
    question_type: str = Field(default="lookup")
    extraction_method: str = Field(default="rules")
    latency_ms: float = Field(default=0.0)

    @field_validator("question_type")
    @classmethod
    def validate_question_type(cls, v: str) -> str:
        allowed = {"lookup", "list", "comparison", "aggregation", "cross_entity", "other"}
        return v if v in allowed else "lookup"


# ============================================================
# NEW — GRAPH TRAVERSAL
# Output of semantic_graph path-finding
# Carries the path + metadata for the Glass Box
# ============================================================

class GraphTraversal(BaseModel):
    """Result of graph path-finding between entities.

    Attributes:
        path_taken      : Entity node names in traversal order
                          e.g. ["Employee", "Department"]
        join_count      : Number of JOIN operations required
        tables_involved : SQL table names that will be queried
        traversal_time_ms: Time for path-finding (NetworkX call)
        traversal_method: "single_node" / "two_hop" / "multi_hop"
        path_description: Human-readable e.g. "Employee →works_in→ Department"
    """
    path_taken: List[str] = Field(default_factory=list)
    join_count: int = Field(default=0)
    tables_involved: List[str] = Field(default_factory=list)
    traversal_time_ms: float = Field(default=0.0)
    traversal_method: str = Field(default="single_node")
    path_description: str = Field(default="")

    @field_validator("traversal_method")
    @classmethod
    def validate_traversal_method(cls, v: str) -> str:
        allowed = {"single_node", "two_hop", "multi_hop"}
        return v if v in allowed else "single_node"


# ============================================================
# MIDDLEWARE TRACE — EXTENDED (backward-compatible)
# The master object. Streamlit reads from this.
# New fields are Optional so old Glass Box code still works.
# ============================================================

class MiddlewareTrace(BaseModel):
    """Complete record of a single pipeline execution.

    Phase 2 additions:
      - entity_extraction : New EntityExtractionResult
      - graph_traversal   : New GraphTraversal
      - semantic_mode     : True when new graph pipeline ran

    Existing fields (intent, parameters, db_result, final_answer)
    are populated in the new pipeline to keep the Glass Box working:
      - trace.intent.intent_name  ← stores path string e.g. "Employee→Department"
      - trace.intent.model_used   ← "semantic_graph" or "rules"
      - trace.parameters.params   ← stores filters dict
      - trace.parameters.extraction_method ← "rules" or "llm"
    """

    user_question: str = Field(..., description="Original user question")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Populated step by step — OLD fields (Glass Box reads these)
    intent: Optional[IntentClassification] = Field(default=None)
    parameters: Optional[ExtractedParameters] = Field(default=None)
    db_result: Optional[DBResult] = Field(default=None)
    final_answer: str = Field(default="")

    # NEW fields (Phase 2) — Optional so Phase 1 tests still pass
    entity_extraction: Optional[EntityExtractionResult] = Field(default=None)
    graph_traversal: Optional[GraphTraversal] = Field(default=None)
    semantic_mode: bool = Field(default=False)

    # Timing and meta
    total_latency_ms: float = Field(default=0.0)
    self_healing_triggered: bool = Field(default=False)
    pipeline_stage_reached: str = Field(default="init")

    model_config = ConfigDict(arbitrary_types_allowed=True)