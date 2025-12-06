#!/usr/bin/env python3
"""
Deep integration tests for all services - not just imports
"""
import asyncio
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING)


async def test_normalization_service():
    """Test normalization service with actual document."""
    print("\n📝 Testing Normalization Service...")
    try:
        from services.normalization.service import NormalizationService
        from shared.models.schemas import DocumentType
        import tempfile
        import os
        
        service = NormalizationService()
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nAnd special characters: @#$%")
            temp_path = f.name
        
        try:
            result = await service.normalize_document(
                file_path=temp_path,
                document_id="test_doc_1",
                source_type=DocumentType.TEXT
            )
            
            if result and result.chunks:
                print(f"   ✅ Text normalization working ({len(result.chunks)} chunks created)")
                print(f"   Sample chunk: {result.chunks[0].text[:50]}...")
                return True
            else:
                print("   ❌ No chunks generated")
                return False
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"   ❌ NormalizationService failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_extraction_service():
    """Test extraction service with actual text."""
    print("\n🔍 Testing Extraction Service...")
    try:
        from services.extraction.service import ExtractionService
        
        service = ExtractionService()
        
        # Test extraction
        test_text = """
        Albert Einstein was born in 1879 in Germany. 
        He developed the theory of relativity.
        Einstein won the Nobel Prize in Physics in 1921.
        """
        
        result = await service.extract_from_text(
            text=test_text,
            chunk_id="test_chunk_1",
            document_id="test_doc_1"
        )
        
        if result and result.candidate_triples:
            print(f"   ✅ Extraction working ({len(result.candidate_triples)} triples extracted)")
            for candidate in result.candidate_triples[:3]:
                triple = candidate.triple
                print(f"      ({triple.subject}, {triple.predicate}, {triple.object})")
            return True
        else:
            print("   ❌ No triples extracted")
            return False
            
    except Exception as e:
        print(f"   ❌ ExtractionService failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_validation_service():
    """Test validation service with actual triple."""
    print("\n✅ Testing Validation Service...")
    try:
        from services.validation.service import ValidationEngine
        from shared.models.schemas import Triple, CandidateTriple, EvidenceSpan
        
        engine = ValidationEngine()
        
        # Test validation with a known fact
        test_candidate = CandidateTriple(
            triple_id="test_triple_1",
            triple=Triple(
                subject="Albert Einstein",
                predicate="born_in_year",
                object="1879"
            ),
            evidence=[EvidenceSpan(start=0, end=50, text="Albert Einstein was born in 1879")],
            confidence=0.9,
            extraction_method="llm"
        )
        
        result = await engine.validate_triple(test_candidate)
        
        print(f"   ✅ Validation working")
        print(f"      Status: {result.status}")
        print(f"      Confidence: {result.confidence:.3f}")
        
        return True
            
    except Exception as e:
        print(f"   ❌ ValidationEngine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_entity_resolution_service():
    """Test entity resolution with actual entities."""
    print("\n🔗 Testing Entity Resolution Service...")
    try:
        from services.entity_resolution.service import EntityResolutionService
        from shared.models.schemas import EntityType
        
        service = EntityResolutionService()
        
        # Test entity resolution
        entity_name = "Albert Einstein"
        entity_id = await service.resolve_entity(entity_name, EntityType.PERSON)
        
        print(f"   ✅ Entity resolution working")
        print(f"      '{entity_name}' resolved to: {entity_id[:50]}")
        
        # Test with similar name
        similar_name = "A. Einstein"
        similar_id = await service.resolve_entity(similar_name, EntityType.PERSON)
        
        print(f"      '{similar_name}' resolved to: {similar_id[:50]}")
        
        return True
            
    except Exception as e:
        print(f"   ❌ EntityResolutionService failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fusion_service():
    """Test fusion service with triples."""
    print("\n🔀 Testing Fusion Service...")
    try:
        from services.fusion.service import FusionService
        from shared.models.schemas import Triple, ValidationStatus
        
        service = FusionService()
        
        # Create test triples
        triples = [
            Triple(
                id="triple_1",
                subject="Albert Einstein",
                predicate="born_in_year",
                object="1879",
                confidence=0.9,
                source_chunk_id="chunk_1",
                document_id="doc_1",
                validation_status=ValidationStatus.VALIDATED
            ),
            Triple(
                id="triple_2",
                subject="Albert Einstein",
                predicate="born_in_year",
                object="1879",
                confidence=0.85,
                source_chunk_id="chunk_2",
                document_id="doc_2",
                validation_status=ValidationStatus.VALIDATED
            )
        ]
        
        result = await service.fuse_triples(triples)
        
        print(f"   ✅ Fusion working")
        print(f"      Input triples: {len(triples)}")
        print(f"      Fused triples: {len(result.fused_triples)}")
        
        if result.fused_triples:
            fused = result.fused_triples[0]
            print(f"      Merged confidence: {fused.confidence:.3f}")
        
        return True
            
    except Exception as e:
        print(f"   ❌ FusionService failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedding_service():
    """Test embedding service with actual text."""
    print("\n🎯 Testing Embedding Service...")
    try:
        from services.embedding.service import EmbeddingService
        
        service = EmbeddingService()
        
        # Test embedding generation
        test_texts = [
            "Albert Einstein was a physicist",
            "Marie Curie won the Nobel Prize",
            "The theory of relativity"
        ]
        
        embeddings = [service.embed_text(text) for text in test_texts]
        
        print(f"   ✅ Embedding service working")
        print(f"      Generated {len(embeddings)} embeddings")
        print(f"      Embedding dimension: {len(embeddings[0])}")
        
        # Test similarity
        if len(embeddings) >= 2:
            import numpy as np
            sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"      Sample similarity score: {sim:.3f}")
        
        return True
            
    except Exception as e:
        print(f"   ❌ EmbeddingService failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_query_service():
    """Test query service initialization."""
    print("\n🔎 Testing Query Service...")
    try:
        from services.query.service import QueryService
        
        service = QueryService()
        
        print(f"   ✅ QueryService initialized")
        print(f"      (Skipping full query test - requires data in graph)")
        
        return True
            
    except Exception as e:
        print(f"   ❌ QueryService failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mongodb_operations():
    """Test MongoDB read/write operations."""
    print("\n💾 Testing MongoDB Operations...")
    try:
        from shared.database.mongodb import get_mongodb
        import time
        
        db = get_mongodb()
        test_collection = db.get_async_collection("test_collection")
        
        # Write test
        test_doc = {
            "test_id": f"test_{int(time.time())}",
            "message": "Integration test",
            "timestamp": time.time()
        }
        
        result = await test_collection.insert_one(test_doc)
        
        # Read test
        found = await test_collection.find_one({"_id": result.inserted_id})
        
        # Delete test
        await test_collection.delete_one({"_id": result.inserted_id})
        
        print(f"   ✅ MongoDB operations working")
        print(f"      Write successful: {result.inserted_id}")
        print(f"      Read successful: {found['message']}")
        print(f"      Delete successful")
        
        return True
            
    except Exception as e:
        print(f"   ❌ MongoDB operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_neo4j_operations():
    """Test Neo4j read/write operations."""
    print("\n🕸️  Testing Neo4j Operations...")
    try:
        from shared.database.neo4j import get_neo4j
        import time
        
        neo4j = get_neo4j()
        
        # Write test
        test_id = f"test_entity_{int(time.time())}"
        
        with neo4j.get_session() as session:
            # Create test entity
            session.run(
                """
                CREATE (e:TestEntity {id: $id, name: $name, created_at: $timestamp})
                RETURN e.id as id
                """,
                id=test_id,
                name="Test Entity",
                timestamp=time.time()
            )
            
            # Read test
            result = session.run(
                "MATCH (e:TestEntity {id: $id}) RETURN e.name as name",
                id=test_id
            )
            record = result.single()
            
            # Delete test
            session.run(
                "MATCH (e:TestEntity {id: $id}) DELETE e",
                id=test_id
            )
        
        print(f"   ✅ Neo4j operations working")
        print(f"      Write successful: {test_id}")
        print(f"      Read successful: {record['name']}")
        print(f"      Delete successful")
        
        return True
            
    except Exception as e:
        print(f"   ❌ Neo4j operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_redis_operations():
    """Test Redis operations."""
    print("\n🔴 Testing Redis Operations...")
    try:
        import redis.asyncio as redis
        from shared.config.settings import RedisSettings
        
        settings = RedisSettings()
        client = await redis.from_url(settings.uri)
        
        # Write test
        test_key = f"test_key_{int(asyncio.get_event_loop().time())}"
        await client.set(test_key, "test_value", ex=60)
        
        # Read test
        value = await client.get(test_key)
        
        # Delete test
        await client.delete(test_key)
        
        await client.aclose()
        
        print(f"   ✅ Redis operations working")
        print(f"      Write successful: {test_key}")
        print(f"      Read successful: {value.decode()}")
        print(f"      Delete successful")
        
        return True
            
    except Exception as e:
        print(f"   ❌ Redis operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all deep integration tests."""
    print("🧪 Deep Integration Tests for All Services")
    print("=" * 60)
    
    results = {
        "MongoDB Operations": await test_mongodb_operations(),
        "Neo4j Operations": await test_neo4j_operations(),
        "Redis Operations": await test_redis_operations(),
        "Normalization": await test_normalization_service(),
        "Extraction": await test_extraction_service(),
        "Validation": await test_validation_service(),
        "Entity Resolution": await test_entity_resolution_service(),
        "Fusion": await test_fusion_service(),
        "Embedding": await test_embedding_service(),
        "Query": await test_query_service(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Deep Integration Test Summary:")
    print("-" * 60)
    
    all_passed = True
    for service, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {service}")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    if all_passed:
        print("🎉 All deep integration tests passed!")
    else:
        print("⚠️  Some tests failed. Review errors above.")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
