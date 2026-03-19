from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractionConfidence:
    HIGH = 1.0
    MEDIUM = 0.7
    LOW = 0.4
    UNCERTAIN = 0.0

    TRUST_THRESHOLD = 0.7
    ESCALATION_THRESHOLD = 0.4


class IntentClassification(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    intent_name: str = Field(..., description="Graph path or intent key")
    confidence: str = Field(default="medium")
    raw_llm_response: str = Field(default="")
    model_used: str = Field(default="")
    latency_ms: float = Field(default=0.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        return v if v in {"high", "medium", "low"} else "medium"


class ExtractedParameters(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    missing_required: List[str] = Field(default_factory=list)
    extraction_method: str = Field(default="none")
    latency_ms: float = Field(default=0.0)


class QueryTemplate(BaseModel):
    intent_name: str
    description: str
    sql_template: str
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)
    result_description: str = Field(default="")


class DBResult(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    execution_time_ms: float = Field(default=0.0)
    query_executed: str = Field(default="")
    params_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    self_healing_triggered: bool = Field(default=False)
    healing_reason: str = Field(default="")

class EntityExtractionResult(BaseModel):
    entities: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    projections: List[str] = Field(default_factory=list)
    question_type: str = Field(default="lookup")
    extraction_method: str = Field(default="pattern")
    latency_ms: float = Field(default=0.0)
    confidence_score: float = Field(default=0.0)
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    escalation_reason: str = Field(default="")

    @field_validator("question_type")
    @classmethod
    def validate_question_type(cls, v: str) -> str:
        allowed = {"lookup", "list", "comparison", "aggregation", "cross_entity", "other"}
        return v if v in allowed else "lookup"

    @property
    def is_trustworthy(self) -> bool:
        return self.confidence_score >= ExtractionConfidence.TRUST_THRESHOLD

    @property
    def requires_escalation(self) -> bool:
        return self.confidence_score < ExtractionConfidence.TRUST_THRESHOLD

    @property
    def is_genuinely_off_topic(self) -> bool:
        return (
            self.question_type == "other"
            and self.confidence_score >= ExtractionConfidence.TRUST_THRESHOLD
        )

class GraphTraversal(BaseModel):
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


class MiddlewareTrace(BaseModel):
    user_question: str = Field(..., description="Original user question")
    timestamp: datetime = Field(default_factory=datetime.now)
    intent: Optional[IntentClassification] = Field(default=None)
    parameters: Optional[ExtractedParameters] = Field(default=None)
    db_result: Optional[DBResult] = Field(default=None)
    final_answer: str = Field(default="")
    entity_extraction: Optional[EntityExtractionResult] = Field(default=None)
    graph_traversal: Optional[GraphTraversal] = Field(default=None)
    semantic_mode: bool = Field(default=False)
    total_latency_ms: float = Field(default=0.0)
    self_healing_triggered: bool = Field(default=False)
    pipeline_stage_reached: str = Field(default="init")

    model_config = ConfigDict(arbitrary_types_allowed=True)
