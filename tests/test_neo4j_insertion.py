#!/usr/bin/env python3
"""
Test Neo4j relationship insertion - the most critical part
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import asyncio
import sys
from pathlib import Path

import torch
torch.set_default_device('cpu')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🔬 Testing Neo4j Relationship Insertion\n")
print("=" * 70)


async def test_neo4j_connection():
    """Test basic Neo4j connection."""
    print("\n1️⃣  Testing Neo4j Connection")
    print("-" * 70)
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        with connector.get_session() as session:
            result = session.run("RETURN 1 as num")
            record = result.single()
            
        print(f"✅ Neo4j connected")
        print(f"   Test query result: {record['num']}")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


async def test_create_entities():
    """Test creating entities in Neo4j."""
    print("\n2️⃣  Testing Entity Creation")
    print("-" * 70)
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        # Clear test data first
        with connector.get_session() as session:
            session.run("""
                MATCH (n)
                WHERE n.test_id = 'test_run_entities'
                DETACH DELETE n
            """)
        
        # Create test entities
        entities = [
            {"id": "entity_einstein", "name": "Albert Einstein", "type": "Person"},
            {"id": "entity_germany", "name": "Germany", "type": "Location"},
            {"id": "entity_physics", "name": "Physics", "type": "Concept"}
        ]
        
        with connector.get_session() as session:
            for entity in entities:
                session.run(f"""
                    CREATE (e:{entity['type']} {{
                        entity_id: $id,
                        canonical_name: $name,
                        entity_type: $type,
                        test_id: 'test_run_entities'
                    }})
                """, id=entity['id'], name=entity['name'], type=entity['type'])
        
        # Verify entities created
        with connector.get_session() as session:
            result = session.run("""
                MATCH (n {test_id: 'test_run_entities'})
                RETURN count(n) as count
            """)
            record = result.single()
            count = record['count']
        
        print(f"✅ Entity creation working")
        print(f"   Created {count} entities")
        for entity in entities:
            print(f"   • {entity['name']} ({entity['type']})")
        
        return count == len(entities)
        
    except Exception as e:
        print(f"❌ Entity creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_create_relationships():
    """Test creating relationships between entities."""
    print("\n3️⃣  Testing Relationship Creation")
    print("-" * 70)
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        # Create relationships
        relationships = [
            {
                "source": "entity_einstein",
                "target": "entity_germany",
                "type": "BORN_IN",
                "properties": {"year": 1879, "confidence": 0.95}
            },
            {
                "source": "entity_einstein",
                "target": "entity_physics",
                "type": "STUDIED",
                "properties": {"field": "theoretical", "confidence": 0.98}
            }
        ]
        
        with connector.get_session() as session:
            for rel in relationships:
                session.run(f"""
                    MATCH (source {{entity_id: $source_id, test_id: 'test_run_entities'}})
                    MATCH (target {{entity_id: $target_id, test_id: 'test_run_entities'}})
                    CREATE (source)-[r:{rel['type']} $props]->(target)
                    SET r.test_id = 'test_run_entities'
                """, 
                source_id=rel['source'], 
                target_id=rel['target'],
                props=rel['properties'])
        
        # Verify relationships created
        with connector.get_session() as session:
            result = session.run("""
                MATCH ()-[r {test_id: 'test_run_entities'}]->()
                RETURN count(r) as count
            """)
            record = result.single()
            count = record['count']
        
        print(f"✅ Relationship creation working")
        print(f"   Created {count} relationships")
        for rel in relationships:
            print(f"   • {rel['type']}: confidence={rel['properties']['confidence']}")
        
        return count == len(relationships)
        
    except Exception as e:
        print(f"❌ Relationship creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_query_relationships():
    """Test querying relationships from Neo4j."""
    print("\n4️⃣  Testing Relationship Queries")
    print("-" * 70)
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        # Query relationships
        with connector.get_session() as session:
            result = session.run("""
                MATCH (source {test_id: 'test_run_entities'})-[r]->(target {test_id: 'test_run_entities'})
                RETURN 
                    source.canonical_name as source_name,
                    type(r) as relationship,
                    target.canonical_name as target_name,
                    r.confidence as confidence
                ORDER BY source_name, relationship
            """)
            
            records = result.data()
        
        print(f"✅ Relationship queries working")
        print(f"   Found {len(records)} relationships:")
        for record in records:
            print(f"   • ({record['source_name']}) -[{record['relationship']}]-> ({record['target_name']})")
            print(f"     Confidence: {record['confidence']}")
        
        return len(records) > 0
        
    except Exception as e:
        print(f"❌ Relationship query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graph_traversal():
    """Test graph traversal (2-hop query)."""
    print("\n5️⃣  Testing Graph Traversal (2-hop)")
    print("-" * 70)
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        # Create additional entity for 2-hop test
        with connector.get_session() as session:
            session.run("""
                CREATE (n:Event {
                    entity_id: 'entity_nobel',
                    canonical_name: 'Nobel Prize in Physics',
                    entity_type: 'Event',
                    test_id: 'test_run_entities'
                })
            """)
            
            # Connect Einstein to Nobel Prize
            session.run("""
                MATCH (source {entity_id: 'entity_einstein', test_id: 'test_run_entities'})
                MATCH (target {entity_id: 'entity_nobel', test_id: 'test_run_entities'})
                CREATE (source)-[r:WON {year: 1921, confidence: 1.0, test_id: 'test_run_entities'}]->(target)
            """)
        
        # Perform 2-hop traversal from Germany
        with connector.get_session() as session:
            result = session.run("""
                MATCH path = (start {entity_id: 'entity_germany', test_id: 'test_run_entities'})-[*1..2]-(end)
                WHERE end.test_id = 'test_run_entities'
                RETURN 
                    [node in nodes(path) | node.canonical_name] as path_nodes,
                    [rel in relationships(path) | type(rel)] as path_rels,
                    length(path) as hops
                ORDER BY hops DESC
                LIMIT 5
            """)
            
            paths = result.data()
        
        print(f"✅ Graph traversal working")
        print(f"   Found {len(paths)} paths from Germany:")
        for i, path in enumerate(paths, 1):
            nodes = " -> ".join(path['path_nodes'])
            rels = ", ".join(path['path_rels'])
            print(f"   {i}. [{path['hops']} hops] {nodes}")
            print(f"      Relations: {rels}")
        
        return len(paths) > 0
        
    except Exception as e:
        print(f"❌ Graph traversal failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_pipeline():
    """Test full pipeline: Triple -> Entities -> Relationships -> Query."""
    print("\n6️⃣  Testing Full Pipeline (Triple to Graph)")
    print("-" * 70)
    try:
        from shared.database.neo4j import get_neo4j
        from shared.models.schemas import Triple, EntityType
        
        connector = get_neo4j()
        
        # Simulate a validated triple
        triple = Triple(
            subject="Marie Curie",
            predicate="won_award",
            object="Nobel Prize in Chemistry",
            subject_type=EntityType.PERSON,
            object_type=EntityType.EVENT
        )
        
        print(f"📋 Input triple: ({triple.subject}, {triple.predicate}, {triple.object})")
        
        # Clear previous test data
        with connector.get_session() as session:
            session.run("""
                MATCH (n)
                WHERE n.test_id = 'test_run_pipeline'
                DETACH DELETE n
            """)
        
        # Step 1: Create subject entity
        with connector.get_session() as session:
            session.run(f"""
                MERGE (e:{triple.subject_type.value} {{
                    entity_id: $entity_id,
                    canonical_name: $name,
                    entity_type: $type,
                    test_id: 'test_run_pipeline'
                }})
            """, entity_id=f"entity_{hash(triple.subject)}", name=triple.subject, type=triple.subject_type.value)
        
        # Step 2: Create object entity
        with connector.get_session() as session:
            session.run(f"""
                MERGE (e:{triple.object_type.value} {{
                    entity_id: $entity_id,
                    canonical_name: $name,
                    entity_type: $type,
                    test_id: 'test_run_pipeline'
                }})
            """, entity_id=f"entity_{hash(triple.object)}", name=triple.object, type=triple.object_type.value)
        
        # Step 3: Create relationship
        relationship_type = triple.predicate.upper().replace(" ", "_")
        with connector.get_session() as session:
            session.run(f"""
                MATCH (subject {{entity_id: $subject_id, test_id: 'test_run_pipeline'}})
                MATCH (object {{entity_id: $object_id, test_id: 'test_run_pipeline'}})
                MERGE (subject)-[r:{relationship_type} {{
                    predicate: $predicate,
                    confidence: $confidence,
                    test_id: 'test_run_pipeline'
                }}]->(object)
            """, 
            subject_id=f"entity_{hash(triple.subject)}", 
            object_id=f"entity_{hash(triple.object)}",
            predicate=triple.predicate,
            confidence=0.92)
        
        # Step 4: Query back the graph
        with connector.get_session() as session:
            result = session.run("""
                MATCH (s {test_id: 'test_run_pipeline'})-[r]->(o {test_id: 'test_run_pipeline'})
                RETURN 
                    s.canonical_name as subject,
                    type(r) as predicate,
                    o.canonical_name as object,
                    r.confidence as confidence,
                    s.entity_type as subject_type,
                    o.entity_type as object_type
            """)
            
            records = result.data()
        
        print(f"✅ Full pipeline working")
        print(f"   Pipeline steps:")
        print(f"   1. Created subject entity: {triple.subject} ({triple.subject_type.value})")
        print(f"   2. Created object entity: {triple.object} ({triple.object_type.value})")
        print(f"   3. Created relationship: {relationship_type}")
        print(f"   4. Retrieved from graph:")
        
        for record in records:
            print(f"      ({record['subject']}) -[{record['predicate']}]-> ({record['object']})")
            print(f"      Confidence: {record['confidence']}")
            print(f"      Types: {record['subject_type']} -> {record['object_type']}")
        
        return len(records) > 0
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_test_data():
    """Clean up all test data."""
    print("\n🧹 Cleaning up test data...")
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        with connector.get_session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.test_id IN ['test_run_entities', 'test_run_pipeline']
                DETACH DELETE n
                RETURN count(n) as deleted
            """)
            record = result.single()
            
        print(f"✅ Cleaned up {record['deleted']} test nodes")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


async def run_all_tests():
    """Run all Neo4j tests."""
    
    results = {}
    
    results["Neo4j Connection"] = await test_neo4j_connection()
    results["Entity Creation"] = await test_create_entities()
    results["Relationship Creation"] = await test_create_relationships()
    results["Relationship Queries"] = await test_query_relationships()
    results["Graph Traversal"] = await test_graph_traversal()
    results["Full Pipeline"] = await test_full_pipeline()
    
    # Cleanup
    await cleanup_test_data()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Neo4j Relationship Test Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print("-" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Neo4j relationship tests passed!")
        print("\n✅ CONFIRMED: Relationships are correctly formed and inserted into Neo4j")
        return True
    else:
        print("❌ Some Neo4j tests failed")
        return False


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
