"""
Test all database and service connections.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio


async def test_mongodb():
    """Test MongoDB connection."""
    print("\n🔌 Testing MongoDB Connection...")
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from shared.config.settings import MongoDBSettings
        
        settings = MongoDBSettings()
        client = AsyncIOMotorClient(settings.uri)
        
        # Test connection
        await client.admin.command('ping')
        
        # List databases
        db_list = await client.list_database_names()
        
        print(f"   ✅ MongoDB connected: {settings.uri}")
        print(f"   Databases: {', '.join(db_list)}")
        
        client.close()
        return True
    except Exception as e:
        print(f"   ❌ MongoDB failed: {e}")
        return False


async def test_redis():
    """Test Redis connection."""
    print("\n🔌 Testing Redis Connection...")
    try:
        import redis.asyncio as redis
        from shared.config.settings import RedisSettings
        
        settings = RedisSettings()
        client = await redis.from_url(settings.uri)
        
        # Test connection
        await client.ping()
        
        # Get info
        info = await client.info('server')
        
        print(f"   ✅ Redis connected: {settings.uri}")
        print(f"   Version: {info['redis_version']}")
        
        await client.close()
        return True
    except Exception as e:
        print(f"   ❌ Redis failed: {e}")
        return False


async def test_neo4j():
    """Test Neo4j connection."""
    print("\n🔌 Testing Neo4j Connection...")
    try:
        from neo4j import AsyncGraphDatabase
        from shared.config.settings import Neo4jSettings
        
        settings = Neo4jSettings()
        driver = AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password)
        )
        
        # Test connection
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS num")
            record = await result.single()
            
        print(f"   ✅ Neo4j connected: {settings.uri}")
        print(f"   Database: {settings.database}")
        
        await driver.close()
        return True
    except Exception as e:
        print(f"   ❌ Neo4j failed: {e}")
        print(f"   Note: If password error, visit http://localhost:7474 and set password to 'password'")
        return False


async def test_ollama():
    """Test Ollama connection and models."""
    print("\n🔌 Testing Ollama Connection...")
    try:
        import httpx
        from shared.config.settings import OllamaSettings
        
        settings = OllamaSettings()
        
        async with httpx.AsyncClient() as client:
            # Test connection
            response = await client.get(f"{settings.base_url}/api/tags")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                print(f"   ✅ Ollama connected: {settings.base_url}")
                print(f"   Installed models: {', '.join(model_names)}")
                
                # Check required models
                required = ['deepseek-r1:1.5b', 'deepseek-r1:7b']
                for model in required:
                    if any(model in name for name in model_names):
                        print(f"   ✅ {model} available")
                    else:
                        print(f"   ⚠️  {model} not found")
                
                return True
            else:
                print(f"   ❌ Ollama returned status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"   ❌ Ollama failed: {e}")
        print(f"   Note: Run 'ollama serve' in another terminal")
        return False


async def test_embedding_model():
    """Test embedding model loading."""
    print("\n🔌 Testing Embedding Model...")
    try:
        from sentence_transformers import SentenceTransformer
        from shared.config.settings import EmbeddingSettings
        
        settings = EmbeddingSettings()
        
        print(f"   Loading model: {settings.model}...")
        model = SentenceTransformer(settings.model, device=settings.device)
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"   ✅ Embedding model loaded: {settings.model}")
        print(f"   Device: {settings.device}")
        print(f"   Embedding dimension: {len(embedding)} (expected: {settings.dimension})")
        
        return True
    except Exception as e:
        print(f"   ❌ Embedding model failed: {e}")
        print(f"   Note: This may be due to NumPy version conflicts")
        return False


async def test_all_connections():
    """Run all connection tests."""
    print("🧪 Testing All Service Connections")
    print("=" * 60)
    
    results = {}
    
    results['mongodb'] = await test_mongodb()
    results['redis'] = await test_redis()
    results['neo4j'] = await test_neo4j()
    results['ollama'] = await test_ollama()
    results['embeddings'] = await test_embedding_model()
    
    print("\n" + "=" * 60)
    print("📊 Connection Test Summary:")
    print("-" * 60)
    
    for service, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {service.upper()}: {'READY' if status else 'FAILED'}")
    
    all_passed = all(results.values())
    
    print("-" * 60)
    if all_passed:
        print("🎉 All services ready! System can start.")
        return True
    else:
        print("⚠️  Some services failed. Fix issues before continuing.")
        return False


if __name__ == "__main__":
    asyncio.run(test_all_connections())
