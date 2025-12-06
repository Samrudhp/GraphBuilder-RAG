"""
Clear all data from Neo4j database.

This will delete:
- All Entity nodes
- All RELATED relationships
- All graph data

Use this when you want to start fresh with Neo4j.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.database.neo4j import get_neo4j


def clear_neo4j():
    """Clear all data from Neo4j."""
    
    print("\n" + "="*80)
    print("🗑️  CLEARING NEO4J DATABASE")
    print("="*80 + "\n")
    
    neo4j = get_neo4j()
    
    try:
        with neo4j.driver.session() as session:
            # Count nodes and relationships before deletion
            count_result = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as node_count, count(DISTINCT r) as rel_count
            """)
            
            record = count_result.single()
            node_count = record["node_count"]
            rel_count = record["rel_count"]
            
            print(f"📊 Current state:")
            print(f"   - Nodes: {node_count}")
            print(f"   - Relationships: {rel_count}")
            
            if node_count == 0 and rel_count == 0:
                print("\n✅ Neo4j is already empty!")
                return
            
            # Ask for confirmation
            print(f"\n⚠️  WARNING: This will permanently delete all data from Neo4j!")
            confirmation = input("\nType 'YES' to confirm deletion: ")
            
            if confirmation != "YES":
                print("\n❌ Deletion cancelled.")
                return
            
            print(f"\n🔄 Deleting all data...")
            
            # Delete all relationships first
            if rel_count > 0:
                session.run("MATCH ()-[r]->() DELETE r")
                print(f"   ✅ Deleted {rel_count} relationships")
            
            # Delete all nodes
            if node_count > 0:
                session.run("MATCH (n) DELETE n")
                print(f"   ✅ Deleted {node_count} nodes")
            
            # Verify deletion
            verify_result = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as node_count, count(DISTINCT r) as rel_count
            """)
            
            verify_record = verify_result.single()
            remaining_nodes = verify_record["node_count"]
            remaining_rels = verify_record["rel_count"]
            
            if remaining_nodes == 0 and remaining_rels == 0:
                print(f"\n✅ Neo4j successfully cleared!")
                print(f"   - Nodes: 0")
                print(f"   - Relationships: 0")
            else:
                print(f"\n⚠️  Warning: Some data remains:")
                print(f"   - Nodes: {remaining_nodes}")
                print(f"   - Relationships: {remaining_rels}")
    
    except Exception as e:
        print(f"\n❌ Error clearing Neo4j: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    clear_neo4j()
