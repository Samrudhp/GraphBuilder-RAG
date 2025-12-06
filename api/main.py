"""
FastAPI Application - Main API server for GraphBuilder-RAG

Endpoints:
- POST /api/v1/ingest - Ingest documents
- POST /api/v1/query - Query with graph-augmented RAG
- GET /api/v1/documents/{id} - Get document status
- GET /api/v1/stats - Get system statistics
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from shared.models.schemas import (
    DocumentType,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    "graphbuilder_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "graphbuilder_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)
DOCUMENTS_INGESTED = Counter(
    "graphbuilder_documents_ingested_total",
    "Total documents ingested",
    ["source_type"],
)
QUERIES_PROCESSED = Counter(
    "graphbuilder_queries_processed_total",
    "Total queries processed",
    ["verification_status"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting GraphBuilder-RAG API server")
    
    settings = get_settings()
    
    # Initialize databases
    mongodb = get_mongodb()
    neo4j = get_neo4j()
    
    # Test connections
    if not mongodb.ping():
        logger.error("MongoDB connection failed")
    else:
        logger.info("MongoDB connected")
        mongodb.create_indexes()
    
    if not neo4j.ping():
        logger.error("Neo4j connection failed")
    else:
        logger.info("Neo4j connected")
        neo4j.create_constraints_and_indexes()
    
    # Initialize FAISS index
    from services.embedding.service import FAISSIndexService
    faiss_service = FAISSIndexService()
    try:
        faiss_service.load_index()
        logger.info("FAISS index loaded")
    except Exception as e:
        logger.warning(f"FAISS index not loaded: {e}")
        logger.info("Creating new FAISS index")
        faiss_service.create_index()
    
    # Verify Ollama models
    from shared.utils.ollama_client import get_ollama_client
    try:
        ollama = get_ollama_client()
        ollama.ensure_models_available()
        logger.info("Ollama models verified")
    except Exception as e:
        logger.error(f"Ollama model check failed: {e}")
    
    logger.info("API server ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server")
    mongodb.close()
    neo4j.close()


# Create FastAPI app
app = FastAPI(
    title="GraphBuilder-RAG API",
    description="Graph-Enhanced Retrieval Augmented Generation System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Track request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)
    
    return response


# ==================== API Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    mongodb = get_mongodb()
    neo4j = get_neo4j()
    
    services = {
        "mongodb": "up" if mongodb.ping() else "down",
        "neo4j": "up" if neo4j.ping() else "down",
    }
    
    all_up = all(status == "up" for status in services.values())
    
    return HealthResponse(
        status="healthy" if all_up else "degraded",
        services=services,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/api/v1/ingest", response_model=IngestionResponse)
async def ingest_document(request: IngestionRequest):
    """
    Ingest a document from URL or API.
    
    Triggers async pipeline: ingestion → normalization → extraction → validation → fusion
    """
    try:
        from services.ingestion.service import IngestionService
        
        service = IngestionService()
        
        # Route based on source type
        if request.source.startswith("http://") or request.source.startswith("https://"):
            raw_doc = await service.ingest_from_url(
                url=request.source,
                source_type=request.source_type,
                metadata=request.metadata,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Only URL sources supported via API. Use file upload endpoint for files.",
            )
        
        # Record metric
        DOCUMENTS_INGESTED.labels(source_type=request.source_type.value).inc()
        
        return IngestionResponse(
            document_id=raw_doc.document_id,
            status="processing",
            message="Document ingestion started. Processing pipeline triggered.",
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_type: DocumentType = Form(...),
):
    """
    Ingest a document from file upload.
    """
    try:
        from pathlib import Path
        from services.ingestion.service import IngestionService
        
        settings = get_settings()
        
        # Save to temp directory
        temp_dir = settings.storage.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file = temp_dir / file.filename
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest
        service = IngestionService()
        raw_doc = await service.ingest_from_file(
            file_path=temp_file,
            source_type=source_type,
        )
        
        # Clean up temp file
        temp_file.unlink()
        
        # Record metric
        DOCUMENTS_INGESTED.labels(source_type=source_type.value).inc()
        
        return IngestionResponse(
            document_id=raw_doc.document_id,
            status="processing",
            message="File uploaded and processing started.",
        )
        
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the system using graph-augmented RAG.
    
    Returns answer with GraphVerify verification.
    """
    try:
        from services.query.service import QueryService
        
        service = QueryService()
        response = await service.answer_question(request)
        
        # Record metric
        QUERIES_PROCESSED.labels(
            verification_status=response.verification_status.value
        ).inc()
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{document_id}")
async def get_document_status(document_id: str):
    """Get document processing status."""
    try:
        mongodb = get_mongodb()
        
        # Check each stage
        raw_doc = await mongodb.get_async_collection("raw_documents").find_one(
            {"document_id": document_id}
        )
        if not raw_doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        normalized = await mongodb.get_async_collection("normalized_docs").find_one(
            {"raw_document_id": document_id}
        )
        
        candidate_count = await mongodb.get_async_collection("candidate_triples").count_documents(
            {"evidence.document_id": document_id}
        )
        
        validated_count = await mongodb.get_async_collection("validated_triples").count_documents(
            {"evidence.document_id": document_id}
        )
        
        return {
            "document_id": document_id,
            "status": {
                "ingested": True,
                "normalized": normalized is not None,
                "extracted": candidate_count > 0,
                "validated": validated_count > 0,
            },
            "stats": {
                "candidate_triples": candidate_count,
                "validated_triples": validated_count,
            },
            "metadata": raw_doc.get("metadata", {}),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics."""
    try:
        from services.ingestion.service import IngestionService
        from services.embedding.service import FAISSIndexService
        
        mongodb = get_mongodb()
        neo4j = get_neo4j()
        
        # MongoDB stats
        ingestion_service = IngestionService()
        ingestion_stats = await ingestion_service.get_ingestion_stats()
        
        doc_count = await mongodb.get_async_collection("raw_documents").count_documents({})
        candidate_count = await mongodb.get_async_collection("candidate_triples").count_documents({})
        validated_count = await mongodb.get_async_collection("validated_triples").count_documents({})
        
        # Neo4j stats
        with neo4j.get_session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS count")
            entity_count = result.single()["count"]
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            relationship_count = result.single()["count"]
        
        # FAISS stats
        faiss_service = FAISSIndexService()
        faiss_service.load_index()
        faiss_stats = faiss_service.get_stats()
        
        return {
            "documents": {
                "total": doc_count,
                "by_type": ingestion_stats,
            },
            "triples": {
                "candidate": candidate_count,
                "validated": validated_count,
            },
            "graph": {
                "entities": entity_count,
                "relationships": relationship_count,
            },
            "embeddings": faiss_stats,
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers if not settings.api.reload else 1,
    )
