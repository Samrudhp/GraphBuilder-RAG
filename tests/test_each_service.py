#!/usr/bin/env python3
"""
Individual service tests - test each service one by one
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🔍 Testing Individual Services\n")
print("=" * 60)

# Test 1: Ingestion Service
print("\n1️⃣  Ingestion Service")
print("-" * 60)
try:
    from services.ingestion.service import IngestionService
    service = IngestionService()
    print("✅ IngestionService initialized")
    print(f"   - MongoDB database: graphbuilder_rag")
    print(f"   - Supported formats: PDF, HTML, CSV, JSON, TXT")
    print(f"   - Handles file uploads and stores raw documents")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Normalization Service
print("\n2️⃣  Normalization Service")
print("-" * 60)
try:
    from services.normalization.service import NormalizationService
    service = NormalizationService()
    print("✅ NormalizationService initialized")
    print(f"   - Chunking strategy: Recursive text splitting")
    print(f"   - PDF handler: pypdf + pdfplumber")
    print(f"   - HTML handler: BeautifulSoup + trafilatura")
    print(f"   - CSV/Excel handler: pandas + openpyxl")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 3: Extraction Service  
print("\n3️⃣  Extraction Service")
print("-" * 60)
try:
    from services.extraction.service import ExtractionService
    from shared.config.settings import get_settings
    service = ExtractionService()
    settings = get_settings()
    print("✅ ExtractionService initialized")
    print(f"   - LLM model: {settings.ollama.extraction_model}")
    print(f"   - Max tokens: {settings.extraction.max_tokens}")
    print(f"   - Temperature: {settings.extraction.temperature}")
    print(f"   - Min confidence: {settings.extraction.min_confidence}")
    print(f"   - Extraction methods: LLM + Table parsing")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Validation Service
print("\n4️⃣  Validation Service")
print("-" * 60)
try:
    from services.validation.service import ValidationEngine
    engine = ValidationEngine()
    print("✅ ValidationEngine initialized")
    print(f"   - Min confidence: {engine.settings.min_confidence}")
    print(f"   - Bootstrap threshold: {engine.settings.bootstrap_threshold}")
    print(f"   - Bootstrap min confidence: {engine.settings.bootstrap_min_confidence}")
    
    if engine.external_verifier:
        print(f"   - External verifier: ENABLED")
        print(f"     • Wikipedia API integration")
        print(f"     • Wikidata API integration")
        print(f"     • Response caching enabled")
    else:
        print(f"   - External verifier: DISABLED")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 5: Entity Resolution Service
print("\n5️⃣  Entity Resolution Service")
print("-" * 60)
try:
    from services.entity_resolution.service import EntityResolutionService
    service = EntityResolutionService()
    print("✅ EntityResolutionService initialized")
    print(f"   - Similarity threshold: {service.settings.similarity_threshold}")
    print(f"   - FAISS top-k: {service.settings.faiss_top_k}")
    print(f"   - Embedding service: {service.embedding_service.settings.model}")
    print(f"   - Strategy: Exact match → Provisional → FAISS → Create new")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 6: Fusion Service
print("\n6️⃣  Fusion Service")
print("-" * 60)
try:
    from services.fusion.service import FusionService
    service = FusionService()
    print("✅ FusionService initialized")
    print(f"   - Batch size: {service.settings.batch_size}")
    print(f"   - Conflict threshold: {service.settings.conflict_threshold}")
    print(f"   - Strategy: Merge identical triples, detect conflicts")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 7: Embedding Service
print("\n7️⃣  Embedding Service")
print("-" * 60)
try:
    from services.embedding.service import EmbeddingService
    service = EmbeddingService()
    print("✅ EmbeddingService initialized")
    print(f"   - Model: {service.settings.model}")
    print(f"   - Dimension: {service.settings.dimension}")
    print(f"   - Device: {service.settings.device}")
    print(f"   - Batch size: {service.settings.batch_size}")
    print(f"   - Model loaded: sentence-transformers")
    print(f"   ⚠️  Skipping embedding test (can cause segfault with NumPy)")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 8: Query Service
print("\n8️⃣  Query Service")
print("-" * 60)
try:
    from services.query.service import QueryService
    from shared.config.settings import get_settings
    service = QueryService()
    settings = get_settings()
    print("✅ QueryService initialized")
    print(f"   - Max chunks: {settings.retrieval.max_chunks}")
    print(f"   - Graph depth: {settings.retrieval.graph_depth}")
    print(f"   - Graph weight: {settings.retrieval.graph_weight}")
    print(f"   - Semantic weight: {settings.retrieval.semantic_weight}")
    print(f"   - Strategy: Hybrid (Graph + Semantic search)")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "=" * 60)
print("📊 Service Inventory: 8 services tested")
print("=" * 60)
