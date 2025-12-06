"""
Ingestion Service - Entry point for document ingestion.

Handles:
- Fetching documents from URLs, file uploads, APIs
- Storing raw binary content in MongoDB GridFS
- Creating document metadata
- Emitting normalization tasks
"""
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.models.schemas import (
    DocumentMetadata,
    DocumentType,
    RawDocument,
)

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting documents into the system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.mongodb = get_mongodb()
        self.gridfs: AsyncIOMotorGridFSBucket = self.mongodb.gridfs
        self.raw_docs_collection = self.mongodb.get_async_collection("raw_documents")
        
    async def ingest_from_url(
        self,
        url: str,
        source_type: DocumentType,
        metadata: Optional[DocumentMetadata] = None,
    ) -> RawDocument:
        """
        Ingest document from URL.
        
        Args:
            url: Document URL
            source_type: Type of document
            metadata: Additional metadata
            
        Returns:
            RawDocument record
        """
        logger.info(f"Ingesting document from URL: {url}")
        
        # Fetch document
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            content = response.content
        
        # Generate document ID and hash
        document_id = f"doc_{uuid4().hex[:12]}"
        content_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
        
        # Check for duplicates
        existing = await self.raw_docs_collection.find_one(
            {"content_hash": content_hash}
        )
        if existing:
            logger.info(f"Document already exists: {existing['document_id']}")
            return RawDocument(**existing)
        
        # Store in GridFS
        gridfs_id = await self.gridfs.upload_from_stream(
            filename=document_id,
            source=content,
            metadata={"source_url": url, "source_type": source_type.value},
        )
        
        # Create document record
        doc_metadata = metadata or DocumentMetadata(source_url=url)
        raw_doc = RawDocument(
            document_id=document_id,
            source_type=source_type,
            source=url,
            content_hash=content_hash,
            file_size=len(content),
            gridfs_id=str(gridfs_id),
            metadata=doc_metadata,
        )
        
        # Insert into MongoDB
        await self.raw_docs_collection.insert_one(raw_doc.model_dump())
        
        logger.info(
            f"Document ingested successfully: {document_id} "
            f"(size: {len(content)} bytes, hash: {content_hash[:20]}...)"
        )
        
        # Emit normalization task
        await self._emit_normalization_task(document_id)
        
        return raw_doc
    
    async def ingest_from_file(
        self,
        file_path: Path,
        source_type: DocumentType,
        metadata: Optional[DocumentMetadata] = None,
    ) -> RawDocument:
        """
        Ingest document from local file.
        
        Args:
            file_path: Path to file
            source_type: Type of document
            metadata: Additional metadata
            
        Returns:
            RawDocument record
        """
        logger.info(f"Ingesting document from file: {file_path}")
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        max_size = self.settings.storage.max_file_size
        if file_path.stat().st_size > max_size:
            raise ValueError(
                f"File too large: {file_path.stat().st_size} bytes "
                f"(max: {max_size})"
            )
        
        # Read file
        content = file_path.read_bytes()
        
        # Generate document ID and hash
        document_id = f"doc_{uuid4().hex[:12]}"
        content_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
        
        # Check for duplicates
        existing = await self.raw_docs_collection.find_one(
            {"content_hash": content_hash}
        )
        if existing:
            logger.info(f"Document already exists: {existing['document_id']}")
            return RawDocument(**existing)
        
        # Store in GridFS
        gridfs_id = await self.gridfs.upload_from_stream(
            filename=document_id,
            source=content,
            metadata={
                "source_path": str(file_path),
                "source_type": source_type.value,
            },
        )
        
        # Create document record
        doc_metadata = metadata or DocumentMetadata()
        raw_doc = RawDocument(
            document_id=document_id,
            source_type=source_type,
            source=str(file_path),
            content_hash=content_hash,
            file_size=len(content),
            gridfs_id=str(gridfs_id),
            metadata=doc_metadata,
        )
        
        # Insert into MongoDB
        await self.raw_docs_collection.insert_one(raw_doc.model_dump())
        
        logger.info(f"Document ingested successfully: {document_id}")
        
        # Emit normalization task
        await self._emit_normalization_task(document_id)
        
        return raw_doc
    
    async def ingest_from_api(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        headers: Optional[dict] = None,
        metadata: Optional[DocumentMetadata] = None,
    ) -> RawDocument:
        """
        Ingest document from API endpoint.
        
        Args:
            api_url: API endpoint URL
            api_key: Optional API key
            headers: Optional HTTP headers
            metadata: Additional metadata
            
        Returns:
            RawDocument record
        """
        logger.info(f"Ingesting document from API: {api_url}")
        
        # Prepare headers
        request_headers = headers or {}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        
        # Fetch from API
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                api_url,
                headers=request_headers,
                follow_redirects=True,
            )
            response.raise_for_status()
            
            # Handle JSON response
            if "application/json" in response.headers.get("content-type", ""):
                import json
                content = json.dumps(response.json(), indent=2).encode()
            else:
                content = response.content
        
        # Generate document ID and hash
        document_id = f"doc_{uuid4().hex[:12]}"
        content_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
        
        # Store in GridFS
        gridfs_id = await self.gridfs.upload_from_stream(
            filename=document_id,
            source=content,
            metadata={"source_url": api_url, "source_type": "api"},
        )
        
        # Create document record
        doc_metadata = metadata or DocumentMetadata(source_url=api_url)
        raw_doc = RawDocument(
            document_id=document_id,
            source_type=DocumentType.API,
            source=api_url,
            content_hash=content_hash,
            file_size=len(content),
            gridfs_id=str(gridfs_id),
            metadata=doc_metadata,
        )
        
        # Insert into MongoDB
        await self.raw_docs_collection.insert_one(raw_doc.model_dump())
        
        logger.info(f"API document ingested successfully: {document_id}")
        
        # Emit normalization task
        await self._emit_normalization_task(document_id)
        
        return raw_doc
    
    async def get_document_content(self, document_id: str) -> bytes:
        """
        Retrieve raw document content from GridFS.
        
        Args:
            document_id: Document ID
            
        Returns:
            Raw binary content
        """
        doc = await self.raw_docs_collection.find_one(
            {"document_id": document_id}
        )
        if not doc:
            raise ValueError(f"Document not found: {document_id}")
        
        gridfs_id = doc.get("gridfs_id")
        if not gridfs_id:
            raise ValueError(f"No GridFS ID for document: {document_id}")
        
        # Download from GridFS
        from bson import ObjectId
        grid_out = await self.gridfs.open_download_stream(
            ObjectId(gridfs_id)
        )
        content = await grid_out.read()
        
        return content
    
    async def _emit_normalization_task(self, document_id: str):
        """
        Emit async task for document normalization.
        
        Args:
            document_id: Document ID to normalize
        """
        try:
            from workers.tasks import normalize_document
            
            # Enqueue Celery task
            normalize_document.delay(document_id)
            logger.info(f"Normalization task emitted for: {document_id}")
        except Exception as e:
            logger.error(f"Failed to emit normalization task: {e}")
    
    async def get_ingestion_stats(self) -> dict:
        """Get ingestion statistics."""
        pipeline = [
            {
                "$group": {
                    "_id": "$source_type",
                    "count": {"$sum": 1},
                    "total_size": {"$sum": "$file_size"},
                }
            }
        ]
        
        stats = {}
        async for result in self.raw_docs_collection.aggregate(pipeline):
            stats[result["_id"]] = {
                "count": result["count"],
                "total_size_mb": round(result["total_size"] / 1024 / 1024, 2),
            }
        
        return stats
