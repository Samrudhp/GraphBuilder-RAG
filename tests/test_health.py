#!/usr/bin/env python3
"""
Quick service health checks - verify services can be instantiated and are ready
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main():
    print("🚀 GraphBuilder-RAG System Health Check")
    print("=" * 60)
    
    results = {}
    
    # Database connections
    print("\n💾 Database Connections:")
    try:
        from shared.database.mongodb import get_mongodb
        get_mongodb()
        print("   ✅ MongoDB connected")
        results["MongoDB"] = True
    except Exception as e:
        print(f"   ❌ MongoDB: {e}")
        results["MongoDB"] = False
    
    try:
        from shared.database.neo4j import get_neo4j
        neo4j = get_neo4j()
        with neo4j.get_session() as session:
            session.run("RETURN 1")
        print("   ✅ Neo4j connected")
        results["Neo4j"] = True
    except Exception as e:
        print(f"   ❌ Neo4j: {e}")
        results["Neo4j"] = False
    
    try:
        import redis.asyncio as redis
        from shared.config.settings import RedisSettings
        settings = RedisSettings()
        client = await redis.from_url(settings.uri)
        await client.ping()
        await client.aclose()
        print("   ✅ Redis connected")
        results["Redis"] = True
    except Exception as e:
        print(f"   ❌ Redis: {e}")
        results["Redis"] = False
    
    # Services
    print("\n🔧 Services:")
    try:
        from services.normalization.service import NormalizationService
        NormalizationService()
        print("   ✅ NormalizationService")
        results["Normalization"] = True
    except Exception as e:
        print(f"   ❌ Normalization: {e}")
        results["Normalization"] = False
    
    try:
        from services.extraction.service import ExtractionService
        ExtractionService()
        print("   ✅ ExtractionService")
        results["Extraction"] = True
    except Exception as e:
        print(f"   ❌ Extraction: {e}")
        results["Extraction"] = False
    
    try:
        from services.validation.service import ValidationEngine
        ValidationEngine()
        print("   ✅ ValidationEngine (with ExternalVerifier)")
        results["Validation"] = True
    except Exception as e:
        print(f"   ❌ Validation: {e}")
        results["Validation"] = False
    
    try:
        from services.entity_resolution.service import EntityResolutionService
        EntityResolutionService()
        print("   ✅ EntityResolutionService")
        results["Entity Resolution"] = True
    except Exception as e:
        print(f"   ❌ Entity Resolution: {e}")
        results["Entity Resolution"] = False
    
    try:
        from services.fusion.service import FusionService
        FusionService()
        print("   ✅ FusionService")
        results["Fusion"] = True
    except Exception as e:
        print(f"   ❌ Fusion: {e}")
        results["Fusion"] = False
    
    try:
        from services.query.service import QueryService
        QueryService()
        print("   ✅ QueryService")
        results["Query"] = True
    except Exception as e:
        print(f"   ❌ Query: {e}")
        results["Query"] = False
    
    # LLM
    print("\n🤖 LLM:")
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        has_1_5b = any('1.5b' in n for n in model_names)
        has_7b = any('7b' in n for n in model_names)
        if has_1_5b and has_7b:
            print("   ✅ Ollama (deepseek-r1:1.5b, deepseek-r1:7b)")
            results["Ollama"] = True
        else:
            print(f"   ⚠️  Ollama running but models missing")
            results["Ollama"] = False
    except Exception as e:
        print(f"   ❌ Ollama: {e}")
        results["Ollama"] = False
    
    # Workers
    print("\n⚙️  Workers:")
    try:
        from workers.tasks import celery_app
        print("   ✅ Celery configured")
        results["Celery"] = True
    except Exception as e:
        print(f"   ❌ Celery: {e}")
        results["Celery"] = False
    
    # API
    print("\n🌐 API:")
    try:
        from api.main import app
        routes = [r.path for r in app.routes]
        if '/health' in routes and '/api/v1/ingest' in routes:
            print("   ✅ FastAPI app ready")
            results["API"] = True
        else:
            print("   ⚠️  FastAPI app missing routes")
            results["API"] = False
    except Exception as e:
        print(f"   ❌ API: {e}")
        results["API"] = False
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"📊 Health Check: {passed}/{total} components ready")
    
    if passed == total:
        print("🎉 All systems operational!")
        return True
    elif passed >= total * 0.8:
        print("⚠️  Most systems ready, some issues detected")
        return True
    else:
        print("❌ Critical systems failing")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
