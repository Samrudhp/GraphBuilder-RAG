"""Shared Pydantic models and schemas."""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ==================== Enums ====================

class DocumentType(str, Enum):
    """Document source types."""
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    TEXT = "text"
    API = "api"


class TripleStatus(str, Enum):
    """Triple processing status."""
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FUSED = "fused"


class VerificationStatus(str, Enum):
    """GraphVerify verification status."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class AgentType(str, Enum):
    """Agent types."""
    REVERIFY = "reverify"
    CONFLICT_RESOLVER = "conflict_resolver"
    SCHEMA_SUGGESTOR = "schema_suggestor"


class EntityType(str, Enum):
    """Entity types."""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    DATE = "Date"
    CONCEPT = "Concept"
    PRODUCT = "Product"
    EVENT = "Event"
    OTHER = "Other"


# ==================== Document Models ====================

class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    source_url: Optional[str] = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    """Raw document schema."""
    document_id: str
    source_type: DocumentType
    source: str  # URL or file path
    content_hash: str
    file_size: int
    gridfs_id: Optional[str] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123abc",
                "source_type": "pdf",
                "source": "https://example.com/paper.pdf",
                "content_hash": "sha256:abc123...",
                "file_size": 1048576,
                "metadata": {"title": "Research Paper", "domain": "medical"}
            }
        }


class Section(BaseModel):
    """Document section."""
    section_id: str
    heading: Optional[str] = None
    level: int = Field(default=0)
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Table(BaseModel):
    """Extracted table."""
    table_id: str
    caption: Optional[str] = None
    headers: list[str]
    rows: list[list[str]]
    page_number: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    """Normalized document schema."""
    document_id: str
    raw_document_id: str
    text: str
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    language: Optional[str] = None
    word_count: int
    char_count: int
    normalized_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== Triple Models ====================

class EvidenceSpan(BaseModel):
    """Evidence span in source text."""
    document_id: str
    start_char: int
    end_char: int
    text: str
    section_id: Optional[str] = None
    table_id: Optional[str] = None


class Triple(BaseModel):
    """Knowledge triple."""
    subject: str
    predicate: str
    object: str
    subject_type: Optional[EntityType] = None
    object_type: Optional[EntityType] = None
    
    @field_validator("subject", "predicate", "object")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class CandidateTriple(BaseModel):
    """Candidate triple with extraction metadata."""
    triple_id: str
    triple: Triple
    evidence: list[EvidenceSpan]
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str  # "table" or "llm"
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Validation result for a triple."""
    rule_checks: dict[str, bool] = Field(default_factory=dict)
    external_verifications: dict[str, Optional[bool]] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    validation_errors: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


class ValidatedTriple(BaseModel):
    """Validated triple."""
    triple_id: str
    candidate_triple_id: str
    triple: Triple
    evidence: list[EvidenceSpan]
    validation: ValidationResult
    status: TripleStatus = TripleStatus.VALIDATED
    validated_at: datetime = Field(default_factory=datetime.utcnow)


# ==================== Entity Models ====================

class Entity(BaseModel):
    """Entity representation."""
    entity_id: str
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProvisionalEntity(BaseModel):
    """Provisional entity before resolution."""
    entity_id: str
    name: str
    entity_type: Optional[EntityType] = None
    source_triple_ids: list[str]
    similarity_candidates: list[tuple[str, float]] = Field(default_factory=list)
    resolution_status: str = "pending"  # pending, resolved, merged
    resolved_to: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ==================== Graph Models ====================

class GraphEdge(BaseModel):
    """Knowledge graph edge."""
    edge_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str]
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UpsertAudit(BaseModel):
    """Audit log for graph upserts."""
    audit_id: str
    edge_id: str
    operation: str  # "insert", "update", "merge"
    previous_version: Optional[int] = None
    new_version: int
    changes: dict[str, Any] = Field(default_factory=dict)
    conflict_detected: bool = False
    conflict_resolution: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Retrieval Models ====================

class ChunkMatch(BaseModel):
    """Semantic search result."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphMatch(BaseModel):
    """Graph traversal result."""
    subgraph: dict[str, Any]  # Neo4j subgraph structure
    relevance_score: float
    node_count: int
    edge_count: int


class HybridRetrievalResult(BaseModel):
    """Combined retrieval result."""
    chunks: list[ChunkMatch]
    graphs: list[GraphMatch]
    combined_score: float
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== Query Models ====================

class QueryRequest(BaseModel):
    """Query request."""
    question: str
    max_chunks: int = Field(default=10, ge=1, le=50)
    graph_depth: int = Field(default=2, ge=1, le=3)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    require_verification: bool = True


class QueryResponse(BaseModel):
    """Query response."""
    query_id: str
    question: str
    answer: str
    verification_status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]
    reasoning_trace: Optional[str] = None
    verified_edges: list[str] = Field(default_factory=list)
    contradicted_edges: list[str] = Field(default_factory=list)
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Agent Models ====================

class AgentTask(BaseModel):
    """Agent task."""
    task_id: str
    agent_type: AgentType
    status: str  # "pending", "running", "completed", "failed"
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ConflictRecord(BaseModel):
    """Conflict detection record."""
    conflict_id: str
    edge_ids: list[str]
    conflict_type: str  # "contradiction", "redundancy", "inconsistency"
    description: str
    severity: float = Field(ge=0.0, le=1.0)
    resolution_status: str = "pending"  # pending, resolved, ignored
    resolution_action: Optional[str] = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


# ==================== API Models ====================

class IngestionRequest(BaseModel):
    """Ingestion API request."""
    source_type: DocumentType
    source: str
    metadata: Optional[dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_type": "url",
                "source": "https://example.com/doc.pdf",
                "metadata": {"domain": "medical", "priority": "high"}
            }
        }


class IngestionResponse(BaseModel):
    """Ingestion API response."""
    document_id: str
    status: str
    message: str
    task_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: dict[str, str] = Field(default_factory=dict)
    version: str = "1.0.0"
