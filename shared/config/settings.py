"""
Configuration management for GraphBuilder-RAG.
Loads settings from environment variables and config files.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoDBSettings(BaseSettings):
    """MongoDB configuration."""
    uri: str = Field(default="mongodb://localhost:27017")
    database: str = Field(default="graphbuilder_rag")
    max_pool_size: int = Field(default=50)
    min_pool_size: int = Field(default=10)
    
    model_config = SettingsConfigDict(env_prefix="MONGODB_")


class Neo4jSettings(BaseSettings):
    """Neo4j configuration."""
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")
    max_connection_lifetime: int = Field(default=3600)
    max_connection_pool_size: int = Field(default=50)
    
    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class RedisSettings(BaseSettings):
    """Redis configuration."""
    uri: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=50)
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration."""
    base_url: str = Field(default="http://localhost:11434")
    extraction_model: str = Field(default="deepseek-r1:1.5b")
    reasoning_model: str = Field(default="deepseek-r1:7b")
    timeout: int = Field(default=120)
    max_retries: int = Field(default=3)
    
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")


class GroqSettings(BaseSettings):
    """Groq Cloud API configuration for fast inference."""
    api_key: str = Field(default="")
    model: str = Field(default="llama-3.3-70b-versatile")
    timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    
    model_config = SettingsConfigDict(env_prefix="GROQ_")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    model: str = Field(default="BAAI/bge-small-en-v1.5")
    dimension: int = Field(default=384)
    batch_size: int = Field(default=32)
    device: str = Field(default="cpu")
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class FAISSSettings(BaseSettings):
    """FAISS index configuration."""
    index_type: str = Field(default="IndexFlatIP")
    index_path: Path = Field(default=Path("./data/faiss_index"))
    nprobe: int = Field(default=10)
    nlist: int = Field(default=100)
    
    model_config = SettingsConfigDict(env_prefix="FAISS_")
    
    @field_validator("index_path", mode="before")
    @classmethod
    def ensure_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class ExtractionSettings(BaseSettings):
    """Extraction service configuration."""
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.1)
    min_confidence: float = Field(default=0.5)
    batch_size: int = Field(default=10)
    
    model_config = SettingsConfigDict(env_prefix="EXTRACTION_")


class ValidationSettings(BaseSettings):
    """Validation engine configuration."""
    min_confidence: float = Field(default=0.7)
    external_timeout: int = Field(default=10)
    parallel_checks: int = Field(default=5)
    
    # Bootstrap phase settings
    bootstrap_threshold: int = Field(default=1000, description="Number of triples before switching from bootstrap to mature validation")
    bootstrap_min_confidence: float = Field(default=0.8, description="Minimum confidence required during bootstrap phase")
    bootstrap_require_wikipedia: bool = Field(default=True, description="Require Wikipedia verification during bootstrap")
    bootstrap_require_wikidata: bool = Field(default=True, description="Require Wikidata verification during bootstrap")
    
    model_config = SettingsConfigDict(env_prefix="VALIDATION_")


class EntityResolutionSettings(BaseSettings):
    """Entity resolution configuration."""
    similarity_threshold: float = Field(default=0.85)
    faiss_top_k: int = Field(default=10)
    
    model_config = SettingsConfigDict(env_prefix="ENTITY_")


class FusionSettings(BaseSettings):
    """Fusion service configuration."""
    batch_size: int = Field(default=100)
    conflict_threshold: float = Field(default=0.8)
    
    model_config = SettingsConfigDict(env_prefix="FUSION_")


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""
    max_chunks: int = Field(default=10)
    graph_depth: int = Field(default=2)
    min_similarity: float = Field(default=0.5)
    graph_weight: float = Field(default=0.6)
    semantic_weight: float = Field(default=0.4)
    
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")


class QuerySettings(BaseSettings):
    """Query service configuration."""
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    timeout: int = Field(default=60)
    
    model_config = SettingsConfigDict(env_prefix="QUERY_")


class GraphVerifySettings(BaseSettings):
    """GraphVerify configuration."""
    contradiction_threshold: float = Field(default=0.7)
    support_threshold: float = Field(default=0.8)
    max_edges_check: int = Field(default=50)
    
    model_config = SettingsConfigDict(env_prefix="GRAPHVERIFY_")


class AgentSettings(BaseSettings):
    """Agent framework configuration."""
    reverify_interval_seconds: int = Field(default=3600)  # 1 hour
    conflict_resolution_interval_seconds: int = Field(default=7200)  # 2 hours
    schema_suggestion_interval_seconds: int = Field(default=86400)  # 24 hours
    reverify_batch_size: int = Field(default=100)
    conflict_batch_size: int = Field(default=50)
    min_predicate_frequency: int = Field(default=5)
    max_concurrent: int = Field(default=3)
    
    model_config = SettingsConfigDict(env_prefix="AGENT_")


class CelerySettings(BaseSettings):
    """Celery configuration."""
    broker_url: str = Field(default="redis://localhost:6379/0")
    result_backend: str = Field(default="redis://localhost:6379/1")
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: list[str] = Field(default=["json"])
    timezone: str = Field(default="UTC")
    task_track_started: bool = Field(default=True)
    task_time_limit: int = Field(default=3600)
    task_soft_time_limit: int = Field(default=3000)
    
    model_config = SettingsConfigDict(env_prefix="CELERY_")


class MonitoringSettings(BaseSettings):
    """Monitoring configuration."""
    prometheus_port: int = Field(default=9090)
    sentry_dsn: Optional[str] = Field(default=None)
    metrics_enabled: bool = Field(default=True)
    
    model_config = SettingsConfigDict(env_prefix="")


class StorageSettings(BaseSettings):
    """Storage configuration."""
    temp_dir: Path = Field(default=Path("./data/temp"))
    max_file_size: int = Field(default=104857600)  # 100MB
    allowed_extensions: list[str] = Field(default=["pdf", "html", "csv", "json", "txt"])
    
    model_config = SettingsConfigDict(env_prefix="STORAGE_")
    
    @field_validator("temp_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class RetrySettings(BaseSettings):
    """Retry configuration."""
    max_attempts: int = Field(default=3)
    backoff_factor: int = Field(default=2)
    max_delay: int = Field(default=60)
    
    model_config = SettingsConfigDict(env_prefix="RETRY_")


class APISettings(BaseSettings):
    """API server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    reload: bool = Field(default=True)
    
    model_config = SettingsConfigDict(env_prefix="API_")


class Settings(BaseSettings):
    """Master settings container."""
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)
    
    # Sub-configurations
    mongodb: MongoDBSettings = Field(default_factory=MongoDBSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    faiss: FAISSSettings = Field(default_factory=FAISSSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    entity_resolution: EntityResolutionSettings = Field(default_factory=EntityResolutionSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    graphverify: GraphVerifySettings = Field(default_factory=GraphVerifySettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    api: APISettings = Field(default_factory=APISettings)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
