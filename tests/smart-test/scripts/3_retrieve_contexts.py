"""
Script 3: Retrieve contexts for all 1000 queries
For each query, extract Neo4j subgraph + FAISS chunks and save to JSON.
"""
import json
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

sys.path.insert(0, str(project_root))

from shared.database.neo4j import get_neo4j
from shared.database.mongodb import get_mongodb

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
CONTEXTS_DIR = BASE_DIR / "retrieved_contexts"
FAISS_INDEX_PATH = Path(__file__).parent / "data/smart_test_faiss"


async def retrieve_neo4j_context(query_text, entities):
    """Retrieve Neo4j subgraph for query entities"""
    neo4j = get_neo4j()
    
    nodes = []
    edges = []
    
    # If no entities provided, extract from query text
    if not entities:
        # Use query text tokens as potential entities
        import re
        entities = [word for word in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_text)]
    
    with neo4j.get_session() as session:
        # Find entities mentioned in query
        for entity in entities[:5]:  # Limit to 5 entities max
            entity_lower = entity.lower()
            
            # Get entity and its 1-hop neighborhood
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.canonical_name) CONTAINS $entity_name
                OR toLower(e.display_name) CONTAINS $entity_name
                OPTIONAL MATCH (e)-[r:RELATED]-(neighbor:Entity)
                RETURN e, collect(DISTINCT {
                    relationship: r,
                    neighbor: neighbor
                }) as connections
                LIMIT 10
            """, entity_name=entity_lower)
            
            for record in result:
                entity_node = dict(record["e"])
                nodes.append({
                    "entity_id": entity_node.get("entity_id"),
                    "canonical_name": entity_node.get("canonical_name"),
                    "display_name": entity_node.get("display_name")
                })
                
                # Add relationships and neighbor nodes
                for conn in record["connections"]:
                    if conn["relationship"] and conn["neighbor"]:
                        rel = dict(conn["relationship"])
                        neighbor = dict(conn["neighbor"])
                        
                        edges.append({
                            "source": entity_node.get("entity_id"),
                            "target": neighbor.get("entity_id"),
                            "type": "RELATED",
                            "semantic_type": rel.get("semantic_type"),
                            "confidence": rel.get("confidence")
                        })
                        
                        # Add neighbor node
                        nodes.append({
                            "entity_id": neighbor.get("entity_id"),
                            "canonical_name": neighbor.get("canonical_name"),
                            "display_name": neighbor.get("display_name")
                        })
    
    # Remove duplicates
    unique_nodes = {n["entity_id"]: n for n in nodes if n.get("entity_id")}
    
    return {
        "nodes": list(unique_nodes.values()),
        "edges": edges
    }


async def retrieve_faiss_context(query_text, top_k=5):
    """Retrieve top-k similar chunks from FAISS"""
    # Load FAISS index
    index = faiss.read_index(str(FAISS_INDEX_PATH / "index.faiss"))
    
    with open(FAISS_INDEX_PATH / "chunk_ids.json") as f:
        chunk_ids = json.load(f)
    
    # Load embedding model
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Encode query
    query_embedding = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = query_embedding.astype('float32').reshape(1, -1)
    
    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Get chunk metadata from MongoDB
    mongodb = get_mongodb()
    embeddings_meta_col = mongodb.get_collection("embeddings_meta")
    
    chunks = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(chunk_ids):
            chunk_id = chunk_ids[idx]
            metadata = embeddings_meta_col.find_one({"chunk_id": chunk_id})
            
            if metadata:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": metadata.get("text"),
                    "score": float(distance),
                    "dataset": metadata.get("metadata", {}).get("dataset"),
                    "sample_id": metadata.get("metadata", {}).get("sample_id")
                })
    
    return chunks


async def process_fever_samples(fever_samples):
    """Process all FEVER samples"""
    print("\n📊 Processing FEVER samples...")
    
    for i, sample in enumerate(fever_samples, 1):
        query_id = sample["id"]
        query_text = sample["claim"]
        entities = sample.get("entities", [])
        
        # Retrieve contexts
        neo4j_context = await retrieve_neo4j_context(query_text, entities)
        faiss_context = await retrieve_faiss_context(query_text, top_k=5)
        
        # Save to JSON
        output = {
            "query_id": query_id,
            "dataset": "FEVER",
            "query_text": query_text,
            "ground_truth": sample["label"],
            "evidence": sample["evidence"],
            "neo4j_context": neo4j_context,
            "faiss_context": faiss_context
        }
        
        output_file = CONTEXTS_DIR / f"{query_id}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        if i % 50 == 0:
            print(f"   Processed {i}/{len(fever_samples)} FEVER samples...")
    
    print(f"✅ Completed {len(fever_samples)} FEVER samples")


async def process_hotpotqa_samples(hotpot_samples):
    """Process all HotpotQA samples"""
    print("\n📊 Processing HotpotQA samples...")
    
    for i, sample in enumerate(hotpot_samples, 1):
        query_id = sample["id"]
        query_text = sample["question"]
        entities = sample.get("entities", [])
        
        # Retrieve contexts
        neo4j_context = await retrieve_neo4j_context(query_text, entities)
        faiss_context = await retrieve_faiss_context(query_text, top_k=5)
        
        # Save to JSON
        output = {
            "query_id": query_id,
            "dataset": "HotpotQA",
            "query_text": query_text,
            "ground_truth": sample["answer"],
            "question_type": sample["type"],
            "evidence": sample["evidence"],
            "neo4j_context": neo4j_context,
            "faiss_context": faiss_context
        }
        
        output_file = CONTEXTS_DIR / f"{query_id}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        if i % 50 == 0:
            print(f"   Processed {i}/{len(hotpot_samples)} HotpotQA samples...")
    
    print(f"✅ Completed {len(hotpot_samples)} HotpotQA samples")


async def main():
    print("=" * 70)
    print("🚀 SMART TEST: Context Retrieval")
    print("=" * 70)
    
    # Create output directory
    CONTEXTS_DIR.mkdir(exist_ok=True)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    with open(DATASETS_DIR / "fever.json") as f:
        fever_samples = json.load(f)
    with open(DATASETS_DIR / "hotpotqa.json") as f:
        hotpot_samples = json.load(f)
    
    print(f"✅ Loaded {len(fever_samples)} FEVER + {len(hotpot_samples)} HotpotQA samples")
    
    # Process samples
    await process_fever_samples(fever_samples)
    await process_hotpotqa_samples(hotpot_samples)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Context Retrieval Complete!")
    print("=" * 70)
    print(f"📊 Retrieved contexts for {len(fever_samples) + len(hotpot_samples)} queries")
    print(f"📁 Saved to: {CONTEXTS_DIR}")
    print(f"\n🔄 Next step: Run 4_claude_evaluation.py for me to evaluate all contexts")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
