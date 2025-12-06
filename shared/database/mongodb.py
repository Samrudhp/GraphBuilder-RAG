"""MongoDB database connector and utilities."""
import logging
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)


class MongoDBConnector:
    """MongoDB connection manager."""
    
    def __init__(self):
        self.settings = get_settings().mongodb
        self._client: Optional[MongoClient] = None
        self._async_client: Optional[AsyncIOMotorClient] = None
        self._gridfs: Optional[Any] = None
        
    @property
    def client(self) -> MongoClient:
        """Get synchronous MongoDB client."""
        if self._client is None:
            self._client = MongoClient(
                self.settings.uri,
                maxPoolSize=self.settings.max_pool_size,
                minPoolSize=self.settings.min_pool_size,
            )
            logger.info("MongoDB synchronous client connected")
        return self._client
    
    @property
    def async_client(self) -> AsyncIOMotorClient:
        """Get asynchronous MongoDB client."""
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(
                self.settings.uri,
                maxPoolSize=self.settings.max_pool_size,
                minPoolSize=self.settings.min_pool_size,
            )
            logger.info("MongoDB async client connected")
        return self._async_client
    
    @property
    def database(self):
        """Get database instance."""
        return self.client[self.settings.database]
    
    @property
    def async_database(self):
        """Get async database instance."""
        return self.async_client[self.settings.database]
    
    @property
    def gridfs(self) -> Any:
        """Get GridFS bucket for binary storage."""
        if self._gridfs is None:
            self._gridfs = AsyncIOMotorGridFSBucket(
                self.async_database
            )
            logger.info("GridFS bucket initialized")
        return self._gridfs
    
    def get_collection(self, collection_name: str):
        """Get collection by name."""
        return self.database[collection_name]
    
    def get_async_collection(self, collection_name: str):
        """Get async collection by name."""
        return self.async_database[collection_name]
    
    def ping(self) -> bool:
        """Test database connection."""
        try:
            self.client.admin.command("ping")
            return True
        except ConnectionFailure:
            logger.error("MongoDB connection failed")
            return False
    
    async def async_ping(self) -> bool:
        """Test async database connection."""
        try:
            await self.async_client.admin.command("ping")
            return True
        except ConnectionFailure:
            logger.error("MongoDB async connection failed")
            return False
    
    def create_indexes(self):
        """Create database indexes for optimal performance."""
        db = self.database
        
        # Raw documents
        db.raw_documents.create_index("document_id", unique=True)
        db.raw_documents.create_index("content_hash")
        db.raw_documents.create_index("ingested_at")
        db.raw_documents.create_index([("metadata.domain", 1)])
        
        # Normalized documents
        db.normalized_docs.create_index("document_id", unique=True)
        db.normalized_docs.create_index("raw_document_id")
        db.normalized_docs.create_index("normalized_at")
        db.normalized_docs.create_index("language")
        
        # Candidate triples
        db.candidate_triples.create_index("triple_id", unique=True)
        db.candidate_triples.create_index([("triple.subject", 1)])
        db.candidate_triples.create_index([("triple.predicate", 1)])
        db.candidate_triples.create_index([("triple.object", 1)])
        db.candidate_triples.create_index("confidence")
        db.candidate_triples.create_index("extraction_method")
        db.candidate_triples.create_index("extracted_at")
        
        # Validated triples
        db.validated_triples.create_index("triple_id", unique=True)
        db.validated_triples.create_index("candidate_triple_id")
        db.validated_triples.create_index("status")
        db.validated_triples.create_index([("triple.subject", 1)])
        db.validated_triples.create_index([("triple.predicate", 1)])
        db.validated_triples.create_index([("triple.object", 1)])
        db.validated_triples.create_index("validated_at")
        
        # Provisional entities
        db.provisional_entities.create_index("entity_id", unique=True)
        db.provisional_entities.create_index("name")
        db.provisional_entities.create_index("entity_type")
        db.provisional_entities.create_index("resolution_status")
        
        # Upsert audit
        db.upsert_audit.create_index("audit_id", unique=True)
        db.upsert_audit.create_index("edge_id")
        db.upsert_audit.create_index("timestamp")
        db.upsert_audit.create_index("conflict_detected")
        
        # Conflict records
        db.conflict_records.create_index("conflict_id", unique=True)
        db.conflict_records.create_index("resolution_status")
        db.conflict_records.create_index("detected_at")
        db.conflict_records.create_index("severity")
        
        # Embeddings metadata
        db.embeddings_meta.create_index("chunk_id", unique=True)
        db.embeddings_meta.create_index("document_id")
        db.embeddings_meta.create_index("faiss_id", unique=True)
        
        # Agent state
        db.agent_state.create_index("task_id", unique=True)
        db.agent_state.create_index("agent_type")
        db.agent_state.create_index("status")
        db.agent_state.create_index("created_at")
        
        logger.info("MongoDB indexes created successfully")
    
    def close(self):
        """Close database connections."""
        if self._client:
            self._client.close()
            logger.info("MongoDB synchronous client closed")
        if self._async_client:
            self._async_client.close()
            logger.info("MongoDB async client closed")


# Global connector instance
_mongodb_connector: Optional[MongoDBConnector] = None


def get_mongodb() -> MongoDBConnector:
    """Get global MongoDB connector instance."""
    global _mongodb_connector
    if _mongodb_connector is None:
        _mongodb_connector = MongoDBConnector()
    return _mongodb_connector
