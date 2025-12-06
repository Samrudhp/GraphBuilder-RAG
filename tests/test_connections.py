"""
Test all database and service connections.
Run this to verify all required services are working before starting the system.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import requests


async def test_mongodb():
    """Test MongoDB connection."""
    print("\n🔌 Testing MongoDB Connection...")
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from shared.config.settings import MongoDBSettings
        
        settings = MongoDBSettings()
        client = AsyncIOMotorClient(settings.uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        await client.admin.command('ping')
        
        # List databases
        db_list = await client.list_database_names()
        
        # Check if our database exists
        db = client[settings.database]
        collections = await db.list_collection_names()
        
        print(f"   ✅ MongoDB connected: {settings.uri}")
        print(f"   Database: {settings.database}")
        print(f"   Collections: {len(collections)} ({', '.join(collections[:5])}{'...' if len(collections) > 5 else ''})")
        
        client.close()
        return True
    except Exception as e:
        print(f"   ❌ MongoDB failed: {e}")
        print(f"   Tip: Make sure MongoDB is running on port 27017")
        return False


async def test_redis():
    """Test Redis connection."""
    print("\n🔌 Testing Redis Connection...")
    try:
        import redis.asyncio as redis
        from shared.config.settings import RedisSettings
        
        settings = RedisSettings()
        client = await redis.from_url(settings.uri, socket_timeout=5)
        
        # Test connection
        pong = await client.ping()
        
        # Get info
        info = await client.info('server')
        
        # Test key operations
        await client.set('test_key', 'test_value', ex=5)
        value = await client.get('test_key')
        
        print(f"   ✅ Redis connected: {settings.uri}")
        print(f"   Version: {info['redis_version']}")
        print(f"   Uptime: {info['uptime_in_days']} days")
        
        await client.close()
        return True
    except Exception as e:
        print(f"   ❌ Redis failed: {e}")
        print(f"   Tip: Make sure Redis is running on port 6379")
        return False


async def test_neo4j():
    """Test Neo4j connection."""
    print("\n🔌 Testing Neo4j Connection...")
    try:
        from shared.database.neo4j import get_neo4j
        
        neo4j = get_neo4j()
        
        # Test connection with sync session
        with neo4j.get_session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            
            # Count entities and relationships
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]
            rel_count = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS count").single()["count"]
        
        print(f"   ✅ Neo4j connected: {neo4j.settings.uri}")
        print(f"   Database: {neo4j.settings.database}")
        print(f"   Entities: {entity_count}")
        print(f"   Relationships: {rel_count}")
        
        return True
    except Exception as e:
        print(f"   ❌ Neo4j failed: {e}")
        print(f"   Tip: Start Neo4j Desktop or run 'net start neo4j' (Windows)")
        print(f"   Visit http://localhost:7474 to check Neo4j Browser")
        return False


async def test_faiss_index():
    """Test FAISS index."""
    print("\n🔌 Testing FAISS Index...")
    try:
        from pathlib import Path
        import faiss
        import pickle
        from shared.config.settings import get_settings
        
        settings = get_settings().faiss
        index_file = settings.index_path / "index.faiss"
        map_file = settings.index_path / "chunk_map.pkl"
        
        if not index_file.exists():
            print(f"   ⚠️  FAISS index not found at {index_file}")
            print(f"   Tip: Upload documents to create FAISS index")
            return False
        
        # Load index
        index = faiss.read_index(str(index_file))
        
        # Load chunk map
        chunk_map = {}
        if map_file.exists():
            with open(map_file, "rb") as f:
                chunk_map = pickle.load(f)
        
        print(f"   ✅ FAISS index loaded: {index_file}")
        print(f"   Vectors: {index.ntotal}")
        print(f"   Chunks mapped: {len(chunk_map)}")
        print(f"   Index type: {type(index).__name__}")
        
        return True
    except Exception as e:
        print(f"   ❌ FAISS index failed: {e}")
        return False


async def test_embedding_model():
    """Test embedding model loading."""
    print("\n🔌 Testing Embedding Model...")
    try:
        from transformers import AutoTokenizer, AutoModel
        from shared.config.settings import get_settings
        import torch
        
        settings = get_settings().embedding
        
        print(f"   Loading model: {settings.model}...")
        tokenizer = AutoTokenizer.from_pretrained(settings.model)
        model = AutoModel.from_pretrained(settings.model)
        
        # Test embedding generation
        test_text = "This is a test sentence."
        encoded_input = tokenizer(test_text, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            embeddings = model_output[0].mean(dim=1)
        
        embedding_dim = embeddings.shape[1]
        
        print(f"   ✅ Embedding model loaded: {settings.model}")
        print(f"   Device: CPU (forced)")
        print(f"   Embedding dimension: {embedding_dim} (expected: {settings.dimension})")
        
        return embedding_dim == settings.dimension
    except Exception as e:
        print(f"   ❌ Embedding model failed: {e}")
        print(f"   Tip: Run 'pip install transformers torch' if missing")
        return False


def test_groq_api():
    """Test Groq API key."""
    print("\n🔌 Testing Groq API...")
    try:
        from shared.config.settings import get_settings
        
        settings = get_settings().groq
        
        if not settings.api_key:
            print(f"   ❌ Groq API key not found")
            print(f"   Tip: Add GROQ_API_KEY to .env file")
            return False
        
        # Test API call
        import groq
        client = groq.Groq(api_key=settings.api_key)
        
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=10,
            temperature=0
        )
        
        print(f"   ✅ Groq API working")
        print(f"   Model: {settings.model}")
        print(f"   Test response: {response.choices[0].message.content}")
        
        return True
    except Exception as e:
        print(f"   ❌ Groq API failed: {e}")
        print(f"   Tip: Check your GROQ_API_KEY in .env file")
        return False


def test_fastapi():
    """Test FastAPI server."""
    print("\n🔌 Testing FastAPI Server...")
    try:
        response = requests.get("http://localhost:8000/docs", timeout=3)
        
        if response.status_code == 200:
            print(f"   ✅ FastAPI server running")
            print(f"   URL: http://localhost:8000")
            print(f"   Docs: http://localhost:8000/docs")
            return True
        else:
            print(f"   ⚠️  FastAPI returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️  FastAPI server not running")
        print(f"   Tip: Start with 'python api/main.py'")
        return False
    except Exception as e:
        print(f"   ❌ FastAPI test failed: {e}")
        return False


def test_celery_worker():
    """Test Celery worker."""
    print("\n🔌 Testing Celery Worker...")
    try:
        from celery import Celery
        from shared.config.settings import get_settings
        
        settings = get_settings().redis
        
        # Connect to Celery
        app = Celery('tasks', broker=settings.uri, backend=settings.uri)
        
        # Check active workers
        inspect = app.control.inspect()
        active_workers = inspect.active()
        
        if active_workers:
            worker_count = len(active_workers)
            print(f"   ✅ Celery worker(s) running: {worker_count}")
            for worker_name in active_workers.keys():
                print(f"      - {worker_name}")
            return True
        else:
            print(f"   ⚠️  No Celery workers found")
            print(f"   Tip: Start worker with 'celery -A workers.tasks worker --loglevel=info --pool=solo'")
            return False
    except Exception as e:
        print(f"   ❌ Celery check failed: {e}")
        return False


async def test_all_connections():
    """Run all connection tests."""
    print("=" * 80)
    print("🧪 GRAPHBUILDER-RAG SYSTEM CONNECTION TEST")
    print("=" * 80)
    
    results = {}
    
    # Core databases
    print("\n📦 CORE DATABASES")
    results['mongodb'] = await test_mongodb()
    results['redis'] = await test_redis()
    results['neo4j'] = await test_neo4j()
    
    # Data & ML
    print("\n🧠 DATA & ML COMPONENTS")
    results['faiss'] = await test_faiss_index()
    results['embeddings'] = await test_embedding_model()
    results['groq'] = test_groq_api()
    
    # Services
    print("\n🚀 SERVICES")
    results['fastapi'] = test_fastapi()
    results['celery'] = test_celery_worker()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 CONNECTION TEST SUMMARY")
    print("=" * 80)
    
    # Group by category
    core = ['mongodb', 'redis', 'neo4j']
    ml = ['faiss', 'embeddings', 'groq']
    services = ['fastapi', 'celery']
    
    def print_category(name, items):
        print(f"\n{name}:")
        for item in items:
            status = results.get(item, False)
            icon = "✅" if status else "❌"
            status_text = "READY" if status else "FAILED"
            print(f"   {icon} {item.upper():<15} {status_text}")
    
    print_category("Core Databases", core)
    print_category("ML & Data", ml)
    print_category("Services", services)
    
    # Overall status
    print("\n" + "-" * 80)
    
    core_ready = all(results.get(k, False) for k in core)
    ml_ready = all(results.get(k, False) for k in ml)
    services_ready = all(results.get(k, False) for k in services)
    all_ready = core_ready and ml_ready and services_ready
    
    if all_ready:
        print("🎉 ALL SYSTEMS READY! GraphBuilder-RAG can operate at full capacity.")
    elif core_ready and ml_ready:
        print("⚡ Core systems ready. Start FastAPI and Celery to begin.")
    elif core_ready:
        print("⚠️  Databases ready but ML components need attention.")
    else:
        print("❌ Critical services missing. Fix database connections first.")
    
    print("-" * 80)
    
    # Next steps
    if not all_ready:
        print("\n📋 NEXT STEPS:")
        if not core_ready:
            if not results.get('mongodb'):
                print("   1. Start MongoDB: Ensure MongoDB is running on port 27017")
            if not results.get('redis'):
                print("   2. Start Redis: Ensure Redis is running on port 6379")
            if not results.get('neo4j'):
                print("   3. Start Neo4j: Use Neo4j Desktop or 'net start neo4j'")
        
        if not results.get('groq'):
            print("   4. Set Groq API key in .env file")
        
        if not results.get('faiss'):
            print("   5. Upload documents to create FAISS index")
        
        if not results.get('fastapi'):
            print("   6. Start API: python api/main.py")
        
        if not results.get('celery'):
            print("   7. Start Celery: celery -A workers.tasks worker --loglevel=info --pool=solo")
    
    print("=" * 80)
    
    return all_ready


if __name__ == "__main__":
    asyncio.run(test_all_connections())
