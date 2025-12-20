"""
Ingest test data for benchmark evaluation WITHOUT using LLMs.

This script creates entities, relationships, and text chunks
directly in the databases without calling Groq/Ollama.
"""
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from services.embedding.service import EmbeddingService, FAISSIndexService
import numpy as np


# Define entities with types
ENTITIES = {
    # Scientists
    "Albert Einstein": ("Person", ["Einstein"]),
    "Marie Curie": ("Person", ["Curie"]),
    "Isaac Newton": ("Person", ["Newton"]),
    "Stephen Hawking": ("Person", ["Hawking"]),
    "Charles Darwin": ("Person", ["Darwin"]),
    "Heinrich Hertz": ("Person", ["Hertz"]),
    "Dmitri Mendeleev": ("Person", ["Mendeleev"]),
    
    # Tech people
    "Guido van Rossum": ("Person", ["van Rossum"]),
    "Linus Torvalds": ("Person", ["Torvalds"]),
    "Bill Gates": ("Person", ["Gates"]),
    "Tim Berners-Lee": ("Person", ["Berners-Lee"]),
    "Jeff Bezos": ("Person", ["Bezos"]),
    
    # Places
    "Warsaw": ("Location", ["Warsaw, Poland"]),
    "Poland": ("Location", ["Republic of Poland"]),
    "France": ("Location", ["French Republic"]),
    "Finland": ("Location", []),
    "Pakistan": ("Location", []),
    "South America": ("Location", []),
    "Australia": ("Location", []),
    
    # Mountains & Geography
    "Mount Everest": ("Location", ["Everest", "Sagarmatha"]),
    "K2": ("Location", ["Mount Godwin-Austen"]),
    "Mariana Trench": ("Location", ["Marianas Trench"]),
    "Pacific Ocean": ("Location", ["Pacific"]),
    "Dead Sea": ("Location", []),
    "Amazon River": ("Location", ["Amazon"]),
    
    # Planets & Space
    "Jupiter": ("Concept", ["planet Jupiter"]),
    "Mars": ("Concept", ["Red Planet"]),
    "Saturn": ("Concept", ["planet Saturn"]),
    "Ganymede": ("Concept", ["Jupiter's moon"]),
    
    # Companies
    "Microsoft": ("Organization", ["Microsoft Corporation"]),
    "Apple Inc.": ("Organization", ["Apple"]),
    "Amazon": ("Organization", ["Amazon.com"]),
    "Harvard University": ("Organization", ["Harvard"]),
    
    # Books & Concepts
    "A Brief History of Time": ("Concept", []),
    "Principia Mathematica": ("Concept", []),
    "On the Origin of Species": ("Concept", []),
    "Python": ("Concept", ["Python programming language"]),
    "Linux": ("Concept", ["Linux operating system"]),
    "World Wide Web": ("Concept", ["WWW", "Web"]),
    "Monty Python": ("Organization", ["Monty Python's Flying Circus"]),
    
    # Scientific concepts
    "Nobel Prize": ("Concept", []),
    "Physics": ("Concept", []),
    "Chemistry": ("Concept", []),
    "Mendelevium": ("Concept", ["element 101"]),
    "Gold": ("Concept", ["Au", "element 79"]),
    
    # Dates/Years
    "1700": ("Date", []),
    "1859": ("Date", []),
    "1901": ("Date", []),
    "1921": ("Date", []),
    "1945": ("Date", []),
    "1950s": ("Date", ["1950-1959"]),
    "1956": ("Date", []),
    "1960": ("Date", []),
    "1966": ("Date", []),
    "1969": ("Date", []),
    "1975": ("Date", []),
    "1976": ("Date", []),
    "1997": ("Date", []),
    
    # Misc
    "World War II": ("Concept", ["WWII", "Second World War"]),
    "hertz": ("Concept", ["Hz", "frequency unit"]),
}


# Define relationships (source, relationship_type, target)
RELATIONSHIPS = [
    # Einstein
    ("Albert Einstein", "won_prize", "Nobel Prize"),
    ("Albert Einstein", "won_in_year", "1921"),
    ("Albert Einstein", "prize_category", "Physics"),
    ("Nobel Prize", "awarded_to", "Albert Einstein"),
    
    # Marie Curie
    ("Marie Curie", "born_in", "Warsaw"),
    ("Marie Curie", "born_in_country", "Poland"),
    ("Warsaw", "capital_of", "Poland"),
    ("Marie Curie", "won_prize", "Nobel Prize"),
    
    # Isaac Newton
    ("Isaac Newton", "wrote", "Principia Mathematica"),
    ("Principia Mathematica", "written_by", "Isaac Newton"),
    
    # Stephen Hawking
    ("Stephen Hawking", "wrote", "A Brief History of Time"),
    ("A Brief History of Time", "written_by", "Stephen Hawking"),
    ("Stephen Hawking", "received_phd_in", "1966"),
    
    # Charles Darwin
    ("Charles Darwin", "wrote", "On the Origin of Species"),
    ("On the Origin of Species", "published_in", "1859"),
    
    # Python
    ("Python", "created_by", "Guido van Rossum"),
    ("Guido van Rossum", "created", "Python"),
    ("Guido van Rossum", "born_in", "1956"),
    ("Python", "named_after", "Monty Python"),
    
    # Linux
    ("Linus Torvalds", "created", "Linux"),
    ("Linus Torvalds", "born_in", "Finland"),
    ("Linus Torvalds", "born_in_year", "1969"),
    
    # Microsoft
    ("Bill Gates", "founded", "Microsoft"),
    ("Microsoft", "founded_in", "1975"),
    ("Bill Gates", "attended", "Harvard University"),
    
    # Apple
    ("Apple Inc.", "founded_in", "1976"),
    
    # Amazon
    ("Jeff Bezos", "founded", "Amazon"),
    ("Amazon", "went_public_in", "1997"),
    
    # Mountains
    ("Mount Everest", "height", "8849 meters"),
    ("K2", "located_in", "Pakistan"),
    
    # Space
    ("Jupiter", "largest_planet_in", "Solar System"),
    ("Mars", "known_as", "Red Planet"),
    ("Saturn", "has_most_moons", "Solar System"),
    ("Ganymede", "moon_of", "Jupiter"),
    
    # Geography
    ("Mariana Trench", "located_in", "Pacific Ocean"),
    ("Amazon River", "located_in", "South America"),
    
    # Nobel Prize
    ("Nobel Prize", "first_awarded_in", "1901"),
    
    # Elements
    ("Mendelevium", "named_after", "Dmitri Mendeleev"),
    ("Gold", "chemical_symbol", "Au"),
    
    # Physics
    ("Heinrich Hertz", "unit_named_after", "hertz"),
    
    # World War II
    ("World War II", "ended_in", "1945"),
    
    # Web
    ("Tim Berners-Lee", "created", "World Wide Web"),
    ("Tim Berners-Lee", "born_in_decade", "1950s"),
]


# Text chunks for FAISS (factual statements)
TEXT_CHUNKS = [
    "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.",
    "Marie Curie was born in Warsaw, Poland in 1867. She was a pioneering physicist and chemist.",
    "Isaac Newton published Principia Mathematica in 1687, laying the foundations of classical mechanics.",
    "Stephen Hawking wrote A Brief History of Time, a popular science book published in 1988.",
    "Charles Darwin published On the Origin of Species in 1859, introducing the theory of evolution.",
    "Python programming language was created by Guido van Rossum and first released in 1991.",
    "Guido van Rossum was born in 1956 in the Netherlands and created the Python programming language.",
    "Mount Everest is the highest mountain on Earth, standing at 8,849 meters (29,032 feet) tall.",
    "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined.",
    "Linus Torvalds was born in Finland in 1969 and created the Linux operating system kernel.",
    "Bill Gates founded Microsoft in 1975 and attended Harvard University before dropping out.",
    "K2, also known as Mount Godwin-Austen, is located in Pakistan and is the second highest mountain in the world.",
    "The Mariana Trench in the Pacific Ocean contains the deepest point on Earth's surface.",
    "Ganymede is the largest moon of Jupiter and the largest moon in the solar system.",
    "Tim Berners-Lee, born in the 1950s, invented the World Wide Web in 1989.",
    "Saturn is known for its prominent ring system and has the most moons of any planet in our solar system.",
    "Jeff Bezos founded Amazon, which went public in 1997.",
    "The Nobel Prize was first awarded in 1901 in accordance with Alfred Nobel's will.",
    "Mendelevium is a chemical element named after Dmitri Mendeleev, the creator of the periodic table.",
    "The Amazon River flows through South America and is one of the longest rivers in the world.",
    "Heinrich Hertz was a German physicist who has the unit of frequency (hertz) named after him.",
    "Australia is the smallest continent and also a country.",
    "Gold is a chemical element with the symbol Au from the Latin word aurum.",
    "Mars is known as the Red Planet due to its reddish appearance from iron oxide on its surface.",
    "World War II ended in 1945 with the surrender of Germany and Japan.",
    "The Dead Sea is the saltiest body of water on Earth with a salinity of about 34%.",
    "Python's name was inspired by the British comedy group Monty Python.",
    "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    "Warsaw is the capital and largest city of Poland.",
    "Stephen Hawking received his PhD in 1966 from the University of Cambridge.",
]


def ingest_data():
    """Ingest all test data into databases."""
    print("=" * 80)
    print("INGESTING TEST DATA (No LLM calls)")
    print("=" * 80)
    
    # Initialize connections
    mongodb = get_mongodb()
    neo4j = get_neo4j()
    embedding_svc = EmbeddingService()
    faiss_svc = FAISSIndexService()
    
    # Step 1: Insert entities into Neo4j
    print(f"\n1. Inserting {len(ENTITIES)} entities into Neo4j...")
    for canonical_name, (entity_type, aliases) in ENTITIES.items():
        entity_id = f"entity_{uuid4().hex[:12]}"
        
        neo4j.upsert_entity(
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases,
            attributes={"created_at": datetime.utcnow().isoformat()},
        )
    
    print(f"   ✓ Inserted {len(ENTITIES)} entities")
    
    # Step 2: Get entity IDs for relationships
    print(f"\n2. Fetching entity IDs...")
    entity_id_map = {}
    with neo4j.get_session() as session:
        result = session.run(
            "MATCH (e:Entity) RETURN e.entity_id as id, e.canonical_name as name"
        )
        for record in result:
            entity_id_map[record["name"]] = record["id"]
    
    print(f"   ✓ Fetched {len(entity_id_map)} entity IDs")
    
    # Step 3: Insert relationships into Neo4j (using RELATED type for NL2Cypher compatibility)
    print(f"\n3. Inserting {len(RELATIONSHIPS)} relationships into Neo4j...")
    inserted_rels = 0
    for source_name, rel_type, target_name in RELATIONSHIPS:
        if source_name not in entity_id_map:
            print(f"   ⚠ Skipping: {source_name} not found")
            continue
        if target_name not in entity_id_map:
            print(f"   ⚠ Skipping: {target_name} not found")
            continue
        
        source_id = entity_id_map[source_name]
        target_id = entity_id_map[target_name]
        edge_id = f"edge_{uuid4().hex[:12]}"
        
        neo4j.upsert_relationship(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type="RELATED",  # Use RELATED type for NL2Cypher queries
            confidence=0.95,  # High confidence for manually created facts
            evidence_ids=[],
            properties={
                "created_at": datetime.utcnow().isoformat(),
                "semantic_type": rel_type,  # Store original type as property
            },
        )
        inserted_rels += 1
    
    print(f"   ✓ Inserted {inserted_rels} relationships")
    
    # Step 4: Insert text chunks into MongoDB and FAISS
    print(f"\n4. Inserting {len(TEXT_CHUNKS)} text chunks into MongoDB + FAISS...")
    
    # Create a fake document
    doc_id = f"doc_{uuid4().hex[:12]}"
    
    normalized_docs = mongodb.get_collection("normalized_documents")
    embeddings_col = mongodb.get_collection("embeddings")
    embeddings_meta = mongodb.get_collection("embeddings_meta")  # Collection used by search
    
    all_embeddings = []
    all_chunk_ids = []
    
    for i, text in enumerate(TEXT_CHUNKS):
        chunk_id = f"chunk_{uuid4().hex[:12]}"
        faiss_id = i  # FAISS index position
        
        # Insert into MongoDB normalized_documents (for backwards compatibility)
        normalized_docs.insert_one({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "text": text,
            "chunk_index": i,
            "metadata": {
                "source": "benchmark_test_data",
                "created_at": datetime.utcnow().isoformat(),
            }
        })
        
        # ALSO insert into embeddings_meta (what search_similar_chunks uses!)
        embeddings_meta.insert_one({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "faiss_id": faiss_id,  # Required field with unique index
            "text": text,
            "chunk_index": i,
            "metadata": {
                "source": "benchmark_test_data",
                "created_at": datetime.utcnow().isoformat(),
            }
        })
        
        # Create embedding
        embedding = embedding_svc.embed_text(text)
        
        # Store embedding in MongoDB
        embeddings_col.insert_one({
            "chunk_id": chunk_id,
            "embedding": embedding.tolist(),
            "metadata": {
                "document_id": doc_id,
                "text": text[:100],  # First 100 chars
                "created_at": datetime.utcnow().isoformat(),
            }
        })
        
        # Collect for FAISS batch insert
        all_embeddings.append(embedding)
        all_chunk_ids.append(chunk_id)
    
    # Add all embeddings to FAISS in batch
    embeddings_array = np.array(all_embeddings)
    faiss_svc.add_embeddings(embeddings_array, all_chunk_ids)
    faiss_svc.save_index()
    
    print(f"   ✓ Inserted {len(TEXT_CHUNKS)} chunks")
    print(f"   ✓ Added {len(all_embeddings)} vectors to FAISS")
    
    # Step 5: Verify
    print(f"\n5. Verifying data...")
    with neo4j.get_session() as session:
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rel_count = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) as count").single()["count"]
    
    mongo_chunks = normalized_docs.count_documents({})
    mongo_embeddings = embeddings_col.count_documents({})
    faiss_vectors = faiss_svc.index.ntotal if faiss_svc.index else 0
    
    print(f"   Neo4j Entities: {entity_count}")
    print(f"   Neo4j Relationships: {rel_count}")
    print(f"   MongoDB Chunks: {mongo_chunks}")
    print(f"   MongoDB Embeddings: {mongo_embeddings}")
    print(f"   FAISS Vectors: {faiss_vectors}")
    
    print("\n" + "=" * 80)
    print("✅ TEST DATA INGESTION COMPLETE!")
    print("=" * 80)
    print("\nYou can now run: python run_evaluations.py --test fever_full_system")


if __name__ == "__main__":
    ingest_data()
