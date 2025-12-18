"""
Celery Worker Tasks

Async task definitions for:
- Document normalization
- Triple extraction
- Triple validation
- Triple fusion
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import asyncio
from celery import Celery

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# Initialize Celery
settings = get_settings()
celery_app = Celery(
    "graphbuilder_rag",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    task_track_started=settings.celery.task_track_started,
    task_time_limit=settings.celery.task_time_limit,
    task_soft_time_limit=settings.celery.task_soft_time_limit,
)


def get_event_loop():
    """Get or create event loop for async tasks (Windows fix)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@celery_app.task(name="normalize_document", bind=True, max_retries=3)
def normalize_document(self, document_id: str):
    """
    Normalize a raw document.
    
    Args:
        document_id: Raw document ID
    """
    try:
        logger.info(f"Task: Normalizing document {document_id}")
        
        from services.normalization.service import NormalizationService
        
        service = NormalizationService()
        loop = get_event_loop()
        result = loop.run_until_complete(service.normalize_document(document_id))
        
        logger.info(f"Document normalized: {result.document_id}")
        return {
            "status": "success",
            "normalized_document_id": result.document_id,
            "sections": len(result.sections),
            "tables": len(result.tables),
        }
        
    except Exception as e:
        logger.error(f"Normalization failed for {document_id}: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="extract_triples", bind=True, max_retries=3)
def extract_triples(self, document_id: str):
    """
    Extract triples from a normalized document.
    
    Args:
        document_id: Normalized document ID
    """
    try:
        logger.info(f"Task: Extracting triples from {document_id}")
        
        from services.extraction.service import ExtractionService
        
        service = ExtractionService()
        loop = get_event_loop()
        results = loop.run_until_complete(service.extract_from_document(document_id))
        
        logger.info(f"Extracted {len(results)} triples from {document_id}")
        return {
            "status": "success",
            "triples_extracted": len(results),
            "document_id": document_id,
        }
        
    except Exception as e:
        logger.error(f"Extraction failed for {document_id}: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="validate_triples", bind=True, max_retries=3)
def validate_triples(self, document_id: str):
    """
    Validate candidate triples from a document.
    
    Args:
        document_id: Document ID
    """
    try:
        logger.info(f"Task: Validating triples for {document_id}")
        
        from services.validation.service import ValidationEngine
        
        engine = ValidationEngine()
        loop = get_event_loop()
        results = loop.run_until_complete(engine.validate_document_triples(document_id))
        
        accepted = [v for v in results if v.status.value == "validated"]
        
        logger.info(
            f"Validation complete: {len(accepted)}/{len(results)} accepted"
        )
        return {
            "status": "success",
            "total_triples": len(results),
            "accepted": len(accepted),
            "rejected": len(results) - len(accepted),
        }
        
    except Exception as e:
        logger.error(f"Validation failed for {document_id}: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="fuse_triples", bind=True, max_retries=3)
def fuse_triples(self, document_id: str):
    """
    Fuse validated triples into Neo4j.
    
    Args:
        document_id: Document ID
    """
    try:
        logger.info(f"Task: Fusing triples for {document_id}")
        
        from services.fusion.service import FusionService
        
        service = FusionService()
        loop = get_event_loop()
        results = loop.run_until_complete(service.fuse_document_triples(document_id))
        
        logger.info(f"Fused {len(results)} edges into knowledge graph")
        
        # Emit embedding task after successful fusion
        try:
            embed_document.delay(document_id)
            logger.info(f"Embedding task emitted for: {document_id}")
        except Exception as e:
            logger.error(f"Failed to emit embedding task: {e}")
        
        return {
            "status": "success",
            "edges_created": len(results),
            "document_id": document_id,
        }
        
    except Exception as e:
        logger.error(f"Fusion failed for {document_id}: {e}")
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="embed_document", bind=True, max_retries=3)
def embed_document(self, document_id: str):
    """
    Embed a normalized document and add to FAISS.
    
    Args:
        document_id: Normalized document ID
    """
    try:
        logger.info(f"Task: Embedding document {document_id}")
        
        from services.embedding.service import EmbeddingPipelineService
        
        service = EmbeddingPipelineService()
        loop = get_event_loop()
        num_chunks = loop.run_until_complete(service.embed_document(document_id))
        
        # Save FAISS index periodically
        service.faiss_service.save_index()
        
        logger.info(f"Embedded {num_chunks} chunks from {document_id}")
        return {
            "status": "success",
            "chunks_embedded": num_chunks,
            "document_id": document_id,
        }
        
    except Exception as e:
        logger.error(f"Embedding failed for {document_id}: {e}")
        raise self.retry(exc=e, countdown=60)


# Periodic tasks
@celery_app.task(name="rebuild_faiss_index")
def rebuild_faiss_index():
    """Rebuild FAISS index from MongoDB."""
    try:
        logger.info("Task: Rebuilding FAISS index")
        
        from services.embedding.service import EmbeddingPipelineService
        import asyncio
        
        service = EmbeddingPipelineService()
        asyncio.run(service.rebuild_index())
        
        logger.info("FAISS index rebuilt successfully")
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"FAISS rebuild failed: {e}")
        raise


@celery_app.task(name="cleanup_old_audits")
def cleanup_old_audits(days_old: int = 90):
    """Clean up old audit records."""
    try:
        from datetime import datetime, timedelta
        from shared.database.mongodb import get_mongodb
        
        logger.info(f"Task: Cleaning up audits older than {days_old} days")
        
        mongodb = get_mongodb()
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        result = mongodb.get_collection("upsert_audit").delete_many({
            "timestamp": {"$lt": cutoff_date}
        })
        
        logger.info(f"Deleted {result.deleted_count} old audit records")
        return {
            "status": "success",
            "deleted_count": result.deleted_count,
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


# Celery beat schedule (for periodic tasks)
celery_app.conf.beat_schedule = {
    "rebuild-faiss-daily": {
        "task": "rebuild_faiss_index",
        "schedule": 86400.0,  # Daily
    },
    "cleanup-audits-weekly": {
        "task": "cleanup_old_audits",
        "schedule": 604800.0,  # Weekly
        "args": (90,),  # 90 days old
    },
}
