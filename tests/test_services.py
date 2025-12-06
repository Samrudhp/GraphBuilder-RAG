#!/usr/bin/env python3
"""
Test all service components individually
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_normalization_service():
    """Test normalization service initialization."""
    print("\n📝 Testing Normalization Service...")
    try:
        from services.normalization.service import NormalizationService
        
        service = NormalizationService()
        print("   ✅ NormalizationService initialized")
        return True
    except Exception as e:
        print(f"   ❌ NormalizationService failed: {e}")
        return False


async def test_extraction_service():
    """Test extraction service initialization."""
    print("\n🔍 Testing Extraction Service...")
    try:
        from services.extraction.service import ExtractionService
        
        service = ExtractionService()
        print("   ✅ ExtractionService initialized")
        return True
    except Exception as e:
        print(f"   ❌ ExtractionService failed: {e}")
        return False


async def test_validation_service():
    """Test validation service initialization."""
    print("\n✅ Testing Validation Service...")
    try:
        from services.validation.service import ValidationEngine
        
        engine = ValidationEngine()
        
        # Check if external verifier is initialized
        if engine.external_verifier:
            print("   ✅ ValidationEngine initialized with ExternalVerifier")
        else:
            print("   ✅ ValidationEngine initialized (no external verifier)")
        
        # Check bootstrap settings
        from shared.config.settings import get_settings
        settings = get_settings()
        print(f"   Bootstrap threshold: {settings.validation.bootstrap_threshold}")
        print(f"   Bootstrap min confidence: {settings.validation.bootstrap_min_confidence}")
        
        return True
    except Exception as e:
        print(f"   ❌ ValidationEngine failed: {e}")
        return False


async def test_entity_resolution_service():
    """Test entity resolution service initialization."""
    print("\n🔗 Testing Entity Resolution Service...")
    try:
        from services.entity_resolution.service import EntityResolutionService
        
        service = EntityResolutionService()
        print("   ✅ EntityResolutionService initialized")
        print("   Entity resolution using FAISS similarity search")
        
        return True
    except Exception as e:
        print(f"   ❌ EntityResolutionService failed: {e}")
        return False


async def test_fusion_service():
    """Test fusion service initialization."""
    print("\n🔀 Testing Fusion Service...")
    try:
        from services.fusion.service import FusionService
        
        service = FusionService()
        print("   ✅ FusionService initialized")
        return True
    except Exception as e:
        print(f"   ❌ FusionService failed: {e}")
        return False


async def test_query_service():
    """Test query service initialization."""
    print("\n🔎 Testing Query Service...")
    try:
        from services.query.service import QueryService
        
        service = QueryService()
        print("   ✅ QueryService initialized")
        return True
    except Exception as e:
        print(f"   ❌ QueryService failed: {e}")
        return False


async def test_worker_tasks():
    """Test worker tasks can be imported."""
    print("\n⚙️  Testing Worker Tasks...")
    try:
        from workers.tasks import (
            normalize_document,
            extract_triples,
            validate_triples,
            fuse_triples,
            embed_document,
        )
        
        print("   ✅ All worker tasks imported successfully")
        print(f"      - normalize_document")
        print(f"      - extract_triples")
        print(f"      - validate_triples")
        print(f"      - fuse_triples")
        print(f"      - embed_document")
        return True
    except Exception as e:
        print(f"   ❌ Worker tasks failed: {e}")
        return False


async def test_api_app():
    """Test FastAPI app initialization."""
    print("\n🌐 Testing API Application...")
    try:
        from api.main import app
        
        print("   ✅ FastAPI app initialized")
        
        # Check routes
        routes = [route.path for route in app.routes]
        important_routes = ['/health', '/api/v1/ingest', '/api/v1/query']
        
        for route in important_routes:
            if route in routes:
                print(f"   ✅ Route registered: {route}")
            else:
                print(f"   ⚠️  Route not found: {route}")
        
        return True
    except Exception as e:
        print(f"   ❌ FastAPI app failed: {e}")
        return False


async def run_all_tests():
    """Run all service tests."""
    print("🧪 Testing All Service Components")
    print("=" * 60)
    
    results = {
        "Normalization": await test_normalization_service(),
        "Extraction": await test_extraction_service(),
        "Validation": await test_validation_service(),
        "Entity Resolution": await test_entity_resolution_service(),
        "Fusion": await test_fusion_service(),
        "Query": await test_query_service(),
        "Workers": await test_worker_tasks(),
        "API": await test_api_app(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Service Test Summary:")
    print("-" * 60)
    
    all_passed = True
    for service, passed in results.items():
        status = "✅ READY" if passed else "❌ FAILED"
        print(f"   {status}: {service}")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    if all_passed:
        print("🎉 All services ready! System can start.")
    else:
        print("⚠️  Some services failed. Fix issues before continuing.")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
