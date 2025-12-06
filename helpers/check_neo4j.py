"""Check what's in Neo4j graph database"""
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

def check_graph():
    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (n) RETURN count(n) as count")
        node_count = result.single()["count"]
        print(f"Total nodes: {node_count}")
        
        # Count relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        print(f"Total relationships: {rel_count}")
        
        # Show all entities
        print("\n=== Entities (Nodes) ===")
        result = session.run("""
            MATCH (n:Entity)
            RETURN n.canonical_name as name, n.entity_type as type
            LIMIT 20
        """)
        for record in result:
            print(f"  - {record['name']} ({record['type']})")
        
        # Show all relationships
        print("\n=== Relationships (Edges) ===")
        result = session.run("""
            MATCH (a:Entity)-[r:RELATED]->(b:Entity)
            RETURN a.canonical_name as subject, 
                   type(r) as relation,
                   b.canonical_name as object,
                   r.edge_id as edge_id,
                   r.confidence as confidence
            LIMIT 20
        """)
        for record in result:
            print(f"  {record['subject']} → {record['object']}")
            print(f"    Edge ID: {record['edge_id']}, Confidence: {record['confidence']:.2f}")

if __name__ == "__main__":
    check_graph()
    driver.close()
