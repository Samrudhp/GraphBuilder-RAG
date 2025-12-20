"""
Script 2: Ingest 1000 samples into Neo4j + FAISS
Extracts entities/relationships and creates embeddings WITHOUT using any LLM.
"""
import json
import sys
import asyncio
from pathlib import Path
from uuid import uuid4
import re
from collections import defaultdict
from dotenv import load_dotenv
import os

# Load environment variables FIRST
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)
print(f"🔧 Loaded environment from: {env_path}")

# Add parent directory to path
sys.path.insert(0, str(project_root))

from shared.database.neo4j import get_neo4j
from shared.database.mongodb import get_mongodb
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
FAISS_INDEX_PATH = Path("data/smart_test_faiss")

# Simple entity extraction (noun phrases, proper nouns)
ENTITY_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
    r'\bNobel Prize\b',
    r'\bPacific Ocean\b',
    r'\bAtlantic Ocean\b',
    r'\bEiffel Tower\b',
    r'\bGreat Wall\b',
    r'\bUniversity of Oxford\b',
]

def extract_entities(text):
    """Extract entities using regex patterns"""
    entities = set()
    for pattern in ENTITY_PATTERNS:
        matches = re.findall(pattern, text)
        entities.update(matches)
    return list(entities)


def extract_relationships(text, entities):
    """Extract simple relationships between entities"""
    relationships = []
    
    # Patterns for common relationship types
    patterns = {
        "won": r'(\w+(?:\s+\w+)*)\s+won\s+(?:the\s+)?(\w+(?:\s+\w+)*)',
        "wrote": r'(\w+(?:\s+\w+)*)\s+wrote\s+(\w+(?:\s+\w+)*)',
        "located_in": r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?located\s+in\s+(\w+(?:\s+\w+)*)',
        "born_in": r'(\w+(?:\s+\w+)*)\s+(?:was\s+)?born\s+in\s+(\w+(?:\s+\w+)*)',
        "studied_at": r'(\w+(?:\s+\w+)*)\s+studied\s+at\s+(\w+(?:\s+\w+)*)',
        "developed": r'(\w+(?:\s+\w+)*)\s+developed\s+(?:the\s+)?(\w+(?:\s+\w+)*)',
        "capital_of": r'(\w+(?:\s+\w+)*)\s+is\s+the\s+capital\s+of\s+(\w+(?:\s+\w+)*)',
        "borders": r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?bordered\s+by\s+(?:the\s+)?(\w+(?:\s+\w+)*)',
    }
    
    for rel_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) == 2:
                relationships.append({
                    "source": match[0].strip(),
                    "target": match[1].strip(),
                    "type": rel_type
                })
    
    return relationships


async def ingest_to_neo4j(samples, dataset_name):
    """Ingest entities and relationships to Neo4j"""
    print(f"\n📊 Ingesting {dataset_name} to Neo4j...")
    
    neo4j = get_neo4j()
    
    entity_ids = {}  # canonical_name -> entity_id
    stats = {"entities": 0, "relationships": 0}
    
    with neo4j.get_session() as session:
        for sample in samples:
            # Extract entities from evidence/claim
            if dataset_name == "FEVER":
                text = f"{sample['claim']} {sample['evidence']}"
            else:  # HotpotQA
                text = f"{sample['question']} {' '.join(sample['evidence'])}"
            
            entities = extract_entities(text)
            relationships = extract_relationships(text, entities)
            
            # Insert entities
            for entity_name in entities:
                canonical = entity_name.lower().strip()
                
                if canonical not in entity_ids:
                    entity_id = f"entity_{uuid4().hex[:12]}"
                    
                    session.run("""
                        MERGE (e:Entity {entity_id: $entity_id})
                        SET e.canonical_name = $canonical_name,
                            e.display_name = $display_name,
                            e.created_at = datetime()
                    """, entity_id=entity_id, canonical_name=canonical, display_name=entity_name)
                    
                    entity_ids[canonical] = entity_id
                    stats["entities"] += 1
            
            # Insert relationships
            for rel in relationships:
                source_canonical = rel["source"].lower().strip()
                target_canonical = rel["target"].lower().strip()
                
                # Create entities if they don't exist
                for name in [source_canonical, target_canonical]:
                    if name not in entity_ids:
                        entity_id = f"entity_{uuid4().hex[:12]}"
                        session.run("""
                            MERGE (e:Entity {entity_id: $entity_id})
                            SET e.canonical_name = $canonical_name,
                                e.display_name = $display_name,
                                e.created_at = datetime()
                        """, entity_id=entity_id, canonical_name=name, display_name=name)
                        entity_ids[name] = entity_id
                        stats["entities"] += 1
                
                # Create relationship
                source_id = entity_ids[source_canonical]
                target_id = entity_ids[target_canonical]
                
                session.run("""
                    MATCH (e1:Entity {entity_id: $source_id})
                    MATCH (e2:Entity {entity_id: $target_id})
                    MERGE (e1)-[r:RELATED {semantic_type: $rel_type}]->(e2)
                    SET r.confidence = 0.9,
                        r.created_at = datetime()
                """, source_id=source_id, target_id=target_id, rel_type=rel["type"])
                
                stats["relationships"] += 1
            
            # Store extracted data back in sample
            sample["entities"] = entities
            sample["relationships"] = relationships
    
    print(f"✅ Neo4j ingestion complete:")
    print(f"   • Entities: {stats['entities']}")
    print(f"   • Relationships: {stats['relationships']}")
    
    return stats


async def ingest_to_faiss(samples, dataset_name, start_faiss_id=0):
    """Ingest text chunks and embeddings to MongoDB + FAISS"""
    print(f"\n📊 Ingesting {dataset_name} to FAISS...")
    
    # Initialize embedding model
    print("🔧 Loading embedding model (BAAI/bge-small-en-v1.5)...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Initialize FAISS index
    dimension = 384  # bge-small-en-v1.5 dimension
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # MongoDB collections
    mongodb = get_mongodb()
    embeddings_col = mongodb.get_collection("embeddings")
    embeddings_meta_col = mongodb.get_collection("embeddings_meta")
    
    chunk_ids = []
    stats = {"chunks": 0}
    current_faiss_id = start_faiss_id
    
    for sample in samples:
        # Create text chunk from evidence
        if dataset_name == "FEVER":
            chunk_text = f"Claim: {sample['claim']}\nEvidence: {sample['evidence']}"
            document_id = sample["id"]
        else:  # HotpotQA
            chunk_text = f"Question: {sample['question']}\nEvidence: {' '.join(sample['evidence'])}\nAnswer: {sample['answer']}"
            document_id = sample["id"]
        
        # Generate embedding
        embedding = model.encode(chunk_text, convert_to_numpy=True, normalize_embeddings=True)
        embedding = embedding.astype('float32')
        
        # Create chunk ID
        chunk_id = f"chunk_{uuid4().hex[:12]}"
        
        # Add to FAISS
        index.add(np.array([embedding]))
        chunk_ids.append(chunk_id)
        
        # Insert embedding to MongoDB
        embeddings_col.insert_one({
            "chunk_id": chunk_id,
            "embedding": embedding.tolist(),
            "created_at": "2024-12-20"
        })
        
        # Insert metadata to MongoDB
        embeddings_meta_col.insert_one({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "faiss_id": current_faiss_id,
            "text": chunk_text,
            "chunk_index": 0,
            "metadata": {
                "dataset": dataset_name,
                "sample_id": sample["id"]
            }
        })
        
        current_faiss_id += 1
        stats["chunks"] += 1
    
    print(f"✅ FAISS ingestion complete:")
    print(f"   • Chunks: {stats['chunks']}")
    print(f"   • Embeddings: {stats['chunks']}")
    
    return index, chunk_ids, stats


async def main():
    print("=" * 70)
    print("🚀 SMART TEST: Data Ingestion Pipeline")
    print("=" * 70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    with open(DATASETS_DIR / "fever.json") as f:
        fever_samples = json.load(f)
    with open(DATASETS_DIR / "hotpotqa.json") as f:
        hotpot_samples = json.load(f)
    
    print(f"✅ Loaded {len(fever_samples)} FEVER + {len(hotpot_samples)} HotpotQA samples")
    
    # Clear existing data
    print("\n🧹 Clearing existing data from Neo4j, MongoDB, and FAISS...")
    
    # Clear Neo4j
    try:
        neo4j = get_neo4j()
        with neo4j.get_session() as session:
            result = session.run("MATCH (n) DETACH DELETE n")
            result.consume()
            
            # Get counts
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"✅ Cleared Neo4j (was: {entity_count} entities, {rel_count} relationships)")
    except Exception as e:
        print(f"⚠️  Neo4j clear error: {e}")
        print("   Attempting to continue...")
    
    # Clear MongoDB
    mongodb = get_mongodb()
    embed_deleted = mongodb.get_collection("embeddings").delete_many({}).deleted_count
    meta_deleted = mongodb.get_collection("embeddings_meta").delete_many({}).deleted_count
    print(f"✅ Cleared MongoDB (deleted {embed_deleted} embeddings, {meta_deleted} metadata)")
    
    # Clear FAISS index
    if FAISS_INDEX_PATH.exists():
        import shutil
        shutil.rmtree(FAISS_INDEX_PATH)
        print(f"✅ Cleared FAISS index directory")
    
    print("✅ All data cleared, ready for fresh ingestion")
    
    # Ingest FEVER to Neo4j
    fever_neo4j_stats = await ingest_to_neo4j(fever_samples, "FEVER")
    
    # Ingest HotpotQA to Neo4j
    hotpot_neo4j_stats = await ingest_to_neo4j(hotpot_samples, "HotpotQA")
    
    # Ingest FEVER to FAISS
    fever_index, fever_chunk_ids, fever_faiss_stats = await ingest_to_faiss(fever_samples, "FEVER", start_faiss_id=0)
    
    # Ingest HotpotQA to FAISS (start after FEVER's 500 samples)
    hotpot_index, hotpot_chunk_ids, hotpot_faiss_stats = await ingest_to_faiss(hotpot_samples, "HotpotQA", start_faiss_id=500)
    
    # Combine FAISS indices
    print("\n🔧 Creating unified FAISS index...")
    combined_index = faiss.IndexFlatIP(384)
    combined_chunk_ids = fever_chunk_ids + hotpot_chunk_ids
    
    # Add all vectors to combined index
    mongodb = get_mongodb()
    embeddings_col = mongodb.get_collection("embeddings")
    
    all_embeddings = []
    for chunk_id in combined_chunk_ids:
        doc = embeddings_col.find_one({"chunk_id": chunk_id})
        all_embeddings.append(np.array(doc["embedding"], dtype='float32'))
    
    combined_index.add(np.array(all_embeddings))
    
    # Save FAISS index
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(combined_index, str(FAISS_INDEX_PATH / "index.faiss"))
    
    # Save chunk IDs mapping
    with open(FAISS_INDEX_PATH / "chunk_ids.json", 'w') as f:
        json.dump(combined_chunk_ids, f)
    
    print(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")
    
    # Save updated datasets with extracted entities/relationships
    with open(DATASETS_DIR / "fever.json", 'w') as f:
        json.dump(fever_samples, f, indent=2)
    with open(DATASETS_DIR / "hotpotqa.json", 'w') as f:
        json.dump(hotpot_samples, f, indent=2)
    
    # Final verification
    print("\n" + "=" * 70)
    print("✅ Ingestion Complete! Final Counts:")
    print("=" * 70)
    
    with neo4j.get_session() as session:
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rel_count = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count").single()["count"]
    
    mongo_chunks = mongodb.get_collection("embeddings_meta").count_documents({})
    mongo_embeddings = mongodb.get_collection("embeddings").count_documents({})
    
    print(f"📊 Neo4j:")
    print(f"   • Entities: {entity_count}")
    print(f"   • Relationships: {rel_count}")
    print(f"\n📊 MongoDB:")
    print(f"   • Chunks (embeddings_meta): {mongo_chunks}")
    print(f"   • Embeddings: {mongo_embeddings}")
    print(f"\n📊 FAISS:")
    print(f"   • Vectors: {combined_index.ntotal}")
    print(f"\n🔄 Next step: Run 3_retrieve_contexts.py to extract contexts for all queries")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
