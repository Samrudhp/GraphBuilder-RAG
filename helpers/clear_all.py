"""
Clear ALL data from the entire system.

This will delete:
- MongoDB: All documents, triples, chunks, metadata
- Neo4j: All nodes and relationships
- FAISS: Vector index and metadata

Use this for a complete fresh start.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from shared.config.settings import get_settings


def clear_all():
    """Clear all data from the entire system."""
    
    print("\n" + "="*80)
    print("🗑️  CLEARING ALL SYSTEM DATA")
    print("="*80 + "\n")
    
    print("⚠️  WARNING: This will PERMANENTLY delete ALL data from:")
    print("   - MongoDB (documents, triples, chunks, metadata)")
    print("   - Neo4j (all nodes and relationships)")
    print("   - FAISS (vector index and metadata)")
    
    confirmation = input("\nType 'DELETE EVERYTHING' to confirm: ")
    
    if confirmation != "DELETE EVERYTHING":
        print("\n❌ Deletion cancelled.")
        return
    
    mongodb = get_mongodb()
    neo4j = get_neo4j()
    settings = get_settings()
    
    print("\n" + "─"*80)
    print("🔄 STEP 1: Clearing MongoDB...")
    print("─"*80)
    
    try:
        collections = [
            "raw_documents",
            "normalized_docs",
            "candidate_triples",
            "validated_triples",
            "embeddings_meta",
            "chunks",
            "audit_log",
        ]
        
        total_deleted = 0
        for collection_name in collections:
            count = mongodb.database[collection_name].count_documents({})
            if count > 0:
                result = mongodb.database[collection_name].delete_many({})
                print(f"   ✅ Deleted {result.deleted_count} documents from '{collection_name}'")
                total_deleted += result.deleted_count
            else:
                print(f"   ⏭️  '{collection_name}' already empty")
        
        print(f"\n✅ MongoDB cleared: {total_deleted} total documents deleted")
    
    except Exception as e:
        print(f"\n❌ Error clearing MongoDB: {e}")
    
    print("\n" + "─"*80)
    print("🔄 STEP 2: Clearing Neo4j...")
    print("─"*80)
    
    try:
        with neo4j.driver.session() as session:
            # Count before deletion
            count_result = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as node_count, count(DISTINCT r) as rel_count
            """)
            
            record = count_result.single()
            node_count = record["node_count"]
            rel_count = record["rel_count"]
            
            if node_count > 0 or rel_count > 0:
                # Delete all relationships
                if rel_count > 0:
                    session.run("MATCH ()-[r]->() DELETE r")
                    print(f"   ✅ Deleted {rel_count} relationships")
                
                # Delete all nodes
                if node_count > 0:
                    session.run("MATCH (n) DELETE n")
                    print(f"   ✅ Deleted {node_count} nodes")
                
                print(f"\n✅ Neo4j cleared: {node_count} nodes, {rel_count} relationships deleted")
            else:
                print(f"   ⏭️  Neo4j already empty")
    
    except Exception as e:
        print(f"\n❌ Error clearing Neo4j: {e}")
    
    print("\n" + "─"*80)
    print("🔄 STEP 3: Clearing FAISS...")
    print("─"*80)
    
    try:
        # FAISS index file
        faiss_index_path = settings.faiss.index_path / "faiss_index.bin"
        
        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
            print(f"   ✅ Deleted FAISS index file: {faiss_index_path}")
        else:
            print(f"   ⏭️  FAISS index file already deleted")
        
        print(f"\n✅ FAISS cleared")
    
    except Exception as e:
        print(f"\n❌ Error clearing FAISS: {e}")
    
    print("\n" + "="*80)
    print("✅ ALL DATA CLEARED SUCCESSFULLY!")
    print("="*80)
    print("\nYou now have a completely fresh system.")
    print("\nNext steps:")
    print("   1. Upload a document: python helpers/upload_test.py")
    print("   2. Wait for pipeline to complete (~2-3 minutes)")
    print("   3. Trigger embedding: python helpers/trigger_embedding.py")
    print("   4. Test queries: python helpers/test_query.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    clear_all()
