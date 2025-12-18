"""
Quick test to verify Neo4j entity creation works
"""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.database.neo4j import get_neo4j

def test_entity_creation():
    """Test creating a simple entity in Neo4j"""
    neo4j = get_neo4j()
    
    print("Testing Neo4j entity creation...")
    print(f"Connected to Neo4j")
    
    # Test 1: Create a test entity
    test_entity_id = "test_entity_123"
    test_name = "Test Entity"
    
    try:
        result = neo4j.upsert_entity(
            entity_id=test_entity_id,
            canonical_name=test_name,
            entity_type="Person",
            aliases=["Test", "TestEntity"],
            attributes={"test": True}
        )
        print(f"✓ Created entity: {result}")
    except Exception as e:
        print(f"✗ Failed to create entity: {e}")
        return False
    
    # Test 2: Verify it exists
    try:
        with neo4j.get_session() as session:
            result = session.run(
                "MATCH (e:Entity {entity_id: $id}) RETURN e",
                id=test_entity_id
            )
            record = result.single()
            if record:
                print(f"✓ Entity verified: {dict(record['e'])}")
            else:
                print(f"✗ Entity not found after creation!")
                return False
    except Exception as e:
        print(f"✗ Failed to verify entity: {e}")
        return False
    
    # Test 3: Clean up
    try:
        with neo4j.get_session() as session:
            session.run(
                "MATCH (e:Entity {entity_id: $id}) DELETE e",
                id=test_entity_id
            )
        print(f"✓ Cleaned up test entity")
    except Exception as e:
        print(f"⚠ Failed to clean up: {e}")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_entity_creation()
