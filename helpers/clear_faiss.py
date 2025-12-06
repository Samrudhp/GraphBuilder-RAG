"""
Clear FAISS vector index and embedding metadata.

This will delete:
- FAISS index file (faiss_index.bin)
- Chunk metadata (MongoDB chunks collection)
- Embedding metadata (MongoDB embeddings_meta collection)

Use this when you want to rebuild the vector index from scratch.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.database.mongodb import get_mongodb
from shared.config.settings import get_settings


def clear_faiss():
    """Clear FAISS index and related metadata."""
    
    print("\n" + "="*80)
    print("🗑️  CLEARING FAISS INDEX & METADATA")
    print("="*80 + "\n")
    
    mongodb = get_mongodb()
    settings = get_settings()
    
    try:
        # Count chunks and embeddings before deletion
        chunks_count = mongodb.database.chunks.count_documents({})
        embeddings_count = mongodb.database.embeddings_meta.count_documents({})
        
        print(f"📊 Current state:")
        print(f"   - Chunks in MongoDB: {chunks_count}")
        print(f"   - Embedding metadata: {embeddings_count}")
        
        # Check for FAISS index file
        faiss_index_path = settings.faiss.index_path / "faiss_index.bin"
        faiss_exists = os.path.exists(faiss_index_path)
        
        if faiss_exists:
            print(f"   - FAISS index file: {faiss_index_path}")
        else:
            print(f"   - FAISS index file: Not found")
        
        if chunks_count == 0 and embeddings_count == 0 and not faiss_exists:
            print("\n✅ FAISS and metadata are already empty!")
            return
        
        # Ask for confirmation
        print(f"\n⚠️  WARNING: This will permanently delete:")
        print(f"   - {chunks_count} chunks from MongoDB")
        print(f"   - {embeddings_count} embedding records from MongoDB")
        if faiss_exists:
            print(f"   - FAISS index file")
        
        confirmation = input("\nType 'YES' to confirm deletion: ")
        
        if confirmation != "YES":
            print("\n❌ Deletion cancelled.")
            return
        
        print(f"\n🔄 Deleting FAISS data...")
        
        # Delete MongoDB chunks
        if chunks_count > 0:
            result = mongodb.database.chunks.delete_many({})
            print(f"   ✅ Deleted {result.deleted_count} chunks from MongoDB")
        
        # Delete MongoDB embedding metadata
        if embeddings_count > 0:
            result = mongodb.database.embeddings_meta.delete_many({})
            print(f"   ✅ Deleted {result.deleted_count} embedding metadata records")
        
        # Delete FAISS index file
        if faiss_exists:
            try:
                os.remove(faiss_index_path)
                print(f"   ✅ Deleted FAISS index file: {faiss_index_path}")
            except Exception as e:
                print(f"   ⚠️  Could not delete FAISS file: {e}")
        
        # Verify deletion
        remaining_chunks = mongodb.database.chunks.count_documents({})
        remaining_embeddings = mongodb.database.embeddings_meta.count_documents({})
        remaining_faiss = os.path.exists(faiss_index_path)
        
        if remaining_chunks == 0 and remaining_embeddings == 0 and not remaining_faiss:
            print(f"\n✅ FAISS successfully cleared!")
            print(f"   - Chunks: 0")
            print(f"   - Embedding metadata: 0")
            print(f"   - FAISS index: deleted")
        else:
            print(f"\n⚠️  Warning: Some data remains:")
            print(f"   - Chunks: {remaining_chunks}")
            print(f"   - Embedding metadata: {remaining_embeddings}")
            print(f"   - FAISS index: {'exists' if remaining_faiss else 'deleted'}")
    
    except Exception as e:
        print(f"\n❌ Error clearing FAISS: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    clear_faiss()
