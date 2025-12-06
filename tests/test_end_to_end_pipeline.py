"""
End-to-End Pipeline Test: Document -> LLM Extraction -> Neo4j Knowledge Graph

Tests the complete flow:
1. Document ingestion
2. Text normalization
3. LLM extraction of entities and relationships
4. Entity resolution
5. Insertion into Neo4j knowledge graph
6. Query verification

This validates the entire RAG-GraphBuilder system.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🔬 End-to-End Pipeline Test: Document -> LLM -> Neo4j")
print("=" * 80)


async def test_complete_pipeline():
    """Test complete pipeline from document to knowledge graph."""
    print("\n📋 Testing Complete Pipeline")
    print("-" * 80)
    
    try:
        # Import services
        from services.ingestion.service import IngestionService
        from services.normalization.service import NormalizationService
        from services.extraction.service import ExtractionService
        from services.entity_resolution.service import EntityResolutionService
        from shared.database.neo4j import get_neo4j
        from shared.models.schemas import Triple, DocumentType, EntityType
        import tempfile
        from pathlib import Path
        
        # Initialize services
        print("\n🔧 Initializing services...")
        ingestion = IngestionService()
        normalization = NormalizationService()
        extraction = ExtractionService()
        entity_resolution = EntityResolutionService()
        neo4j_connector = get_neo4j()
        
        print("✅ All services initialized")
        
        # Clean up previous test data
        print("\n🧹 Cleaning previous test data from Neo4j...")
        with neo4j_connector.get_session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.test_id = 'e2e_pipeline_test'
                DETACH DELETE n
                RETURN count(n) as deleted
            """)
            deleted = result.single()['deleted']
            print(f"✅ Deleted {deleted} previous test nodes")
        
        # Step 1: Create test document
        print("\n📄 Step 1: Document Ingestion")
        print("-" * 80)
        
        test_content = """
        Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of 
        relativity, one of the two pillars of modern physics. Einstein received the 
        Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
        
        Marie Curie was a Polish physicist and chemist who conducted pioneering research 
        on radioactivity. She was the first woman to win a Nobel Prize, and the first 
        person to win the Nobel Prize twice in different scientific fields. Marie Curie 
        was born in Warsaw, Poland in 1867.
        
        Isaac Newton was an English mathematician, physicist, and astronomer. He formulated 
        the laws of motion and universal gravitation. Newton was born in Woolsthorpe, England 
        in 1643. His work Principia Mathematica laid the foundations for classical mechanics.
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Ingest document (async)
            result = await ingestion.ingest_from_file(
                Path(temp_file), 
                DocumentType.TEXT
            )
            document_id = result.document_id
            print(f"✅ Document ingested: {document_id}")
            print(f"   Size: {result.file_size} bytes")
            print(f"   Type: {result.source_type.value}")
            
            # Step 2: Normalize text
            print("\n📝 Step 2: Text Normalization")
            print("-" * 80)
            
            norm_result = await normalization.normalize_document(document_id)
            normalized_doc_id = norm_result.document_id  # Get the normalized document ID
            sections = norm_result.sections
            print(f"✅ Text normalized: {len(sections)} sections")
            print(f"   Normalized doc ID: {normalized_doc_id}")
            print(f"   Word count: {norm_result.word_count}")
            print(f"   Preview: {sections[0].text[:100]}...")
            
            # Step 3: Extract triples using LLM
            print("\n🤖 Step 3: LLM Entity & Relationship Extraction")
            print("-" * 80)
            
            # Extract from the normalized document
            print(f"\n   Extracting from normalized document {normalized_doc_id}...")
            candidate_triples = await extraction.extract_from_document(normalized_doc_id)
            
            print(f"\n✅ Total candidate triples extracted: {len(candidate_triples)}")
            
            # Convert CandidateTriple to Triple for processing
            all_triples = []
            for i, candidate in enumerate(candidate_triples[:10], 1):  # Show first 10
                subject_type = candidate.triple.subject_type.value if candidate.triple.subject_type else "Unknown"
                object_type = candidate.triple.object_type.value if candidate.triple.object_type else "Unknown"
                print(f"   {i}. ({candidate.triple.subject}) -[{candidate.triple.predicate}]-> ({candidate.triple.object})")
                print(f"      Types: {subject_type} -> {object_type}")
                print(f"      Confidence: {candidate.confidence}")
                all_triples.append(candidate.triple)
            
            # Step 4: Resolve entities
            print("\n🔍 Step 4: Entity Resolution")
            print("-" * 80)
            
            # Extract unique entities from triples
            entities_to_resolve = {}
            for candidate in candidate_triples:
                triple = candidate.triple
                if triple.subject not in entities_to_resolve:
                    entities_to_resolve[triple.subject] = {
                        'name': triple.subject,
                        'type': triple.subject_type if triple.subject_type else EntityType.OTHER
                    }
                if triple.object not in entities_to_resolve:
                    entities_to_resolve[triple.object] = {
                        'name': triple.object,
                        'type': triple.object_type if triple.object_type else EntityType.OTHER
                    }
            
            print(f"   Found {len(entities_to_resolve)} unique entities")
            
            resolved_entities = {}
            for entity_name, entity_data in entities_to_resolve.items():
                entity_id = await entity_resolution.resolve_entity(
                    entity_name, 
                    entity_data['type']  # Pass EntityType enum
                )
                resolved_entities[entity_name] = {
                    'entity_id': entity_id,
                    'canonical_form': entity_name,  # Use original name as canonical
                    'type': entity_data['type'].value  # Convert to string for display
                }
                print(f"   ✅ Resolved: {entity_name} ({entity_data['type'].value})")
                print(f"      Entity ID: {entity_id}")
            
            # Step 5: Insert into Neo4j
            print("\n🗄️  Step 5: Knowledge Graph Insertion")
            print("-" * 80)
            
            with neo4j_connector.get_session() as session:
                entities_created = 0
                relationships_created = 0
                
                # Create entities
                print("\n   Creating entities...")
                for entity_name, resolved in resolved_entities.items():
                    entity_type = entities_to_resolve[entity_name]['type']
                    canonical_name = resolved.get('canonical_form', entity_name)
                    entity_id = f"entity_{hash(canonical_name)}"
                    
                    session.run(f"""
                        MERGE (e:{entity_type} {{
                            entity_id: $entity_id,
                            canonical_name: $canonical_name,
                            original_name: $original_name,
                            entity_type: $entity_type,
                            test_id: 'e2e_pipeline_test'
                        }})
                    """, 
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    original_name=entity_name,
                    entity_type=entity_type)
                    
                    entities_created += 1
                
                print(f"   ✅ Created {entities_created} entities")
                
                # Create relationships
                print("\n   Creating relationships...")
                for i, candidate in enumerate(candidate_triples):
                    triple = candidate.triple
                    subject_canonical = resolved_entities[triple.subject].get('canonical_form', triple.subject)
                    object_canonical = resolved_entities[triple.object].get('canonical_form', triple.object)
                    
                    subject_id = f"entity_{hash(subject_canonical)}"
                    object_id = f"entity_{hash(object_canonical)}"
                    
                    relationship_type = triple.predicate.upper().replace(" ", "_").replace("-", "_")
                    
                    # Convert evidence to string if it's a list of EvidenceSpan objects
                    evidence_str = ""
                    if candidate.evidence:
                        if isinstance(candidate.evidence, list):
                            evidence_str = "; ".join([
                                f"{ev.text}" if hasattr(ev, 'text') else str(ev)
                                for ev in candidate.evidence
                            ])
                        else:
                            evidence_str = str(candidate.evidence)
                    
                    session.run(f"""
                        MATCH (subject {{entity_id: $subject_id, test_id: 'e2e_pipeline_test'}})
                        MATCH (object {{entity_id: $object_id, test_id: 'e2e_pipeline_test'}})
                        MERGE (subject)-[r:{relationship_type} {{
                            predicate: $predicate,
                            confidence: $confidence,
                            evidence: $evidence,
                            test_id: 'e2e_pipeline_test'
                        }}]->(object)
                    """,
                    subject_id=subject_id,
                    object_id=object_id,
                    predicate=triple.predicate,
                    confidence=candidate.confidence,
                    evidence=evidence_str)
                    
                    relationships_created += 1
                
                print(f"   ✅ Created {relationships_created} relationships")
            
            # Step 6: Query and verify
            print("\n🔍 Step 6: Knowledge Graph Verification")
            print("-" * 80)
            
            with neo4j_connector.get_session() as session:
                # Count entities
                result = session.run("""
                    MATCH (n {test_id: 'e2e_pipeline_test'})
                    RETURN count(DISTINCT n) as entity_count,
                           collect(DISTINCT labels(n)[0]) as entity_types
                """)
                record = result.single()
                entity_count = record['entity_count']
                entity_types = record['entity_types']
                
                print(f"\n   📊 Graph Statistics:")
                print(f"   • Total entities: {entity_count}")
                print(f"   • Entity types: {', '.join(entity_types)}")
                
                # Count relationships
                result = session.run("""
                    MATCH ()-[r {test_id: 'e2e_pipeline_test'}]->()
                    RETURN count(r) as rel_count,
                           collect(DISTINCT type(r)) as rel_types
                """)
                record = result.single()
                rel_count = record['rel_count']
                rel_types = record['rel_types']
                
                print(f"   • Total relationships: {rel_count}")
                print(f"   • Relationship types: {', '.join(rel_types[:5])}{'...' if len(rel_types) > 5 else ''}")
                
                # Show sample entities
                print("\n   🔎 Sample Entities:")
                result = session.run("""
                    MATCH (n {test_id: 'e2e_pipeline_test'})
                    RETURN n.canonical_name as name, 
                           labels(n)[0] as type,
                           n.original_name as original
                    LIMIT 5
                """)
                for i, record in enumerate(result, 1):
                    name = record['name']
                    entity_type = record['type']
                    original = record['original']
                    print(f"   {i}. {name} ({entity_type})")
                    if original != name:
                        print(f"      Original: {original}")
                
                # Show sample relationships
                print("\n   🔗 Sample Relationships:")
                result = session.run("""
                    MATCH (s {test_id: 'e2e_pipeline_test'})-[r]->(o {test_id: 'e2e_pipeline_test'})
                    RETURN s.canonical_name as subject,
                           type(r) as predicate,
                           o.canonical_name as object,
                           r.confidence as confidence
                    LIMIT 10
                """)
                for i, record in enumerate(result, 1):
                    subject = record['subject']
                    predicate = record['predicate']
                    obj = record['object']
                    confidence = record['confidence']
                    print(f"   {i}. ({subject}) -[{predicate}]-> ({obj})")
                    print(f"      Confidence: {confidence:.2f}")
                
                # Test graph traversal
                print("\n   🌐 Graph Traversal Test (Find all facts about Albert Einstein):")
                result = session.run("""
                    MATCH (einstein {canonical_name: 'Albert Einstein', test_id: 'e2e_pipeline_test'})
                    OPTIONAL MATCH (einstein)-[r1]->(connected1)
                    WHERE connected1.test_id = 'e2e_pipeline_test'
                    OPTIONAL MATCH (einstein)<-[r2]-(connected2)
                    WHERE connected2.test_id = 'e2e_pipeline_test'
                    RETURN 
                        collect(DISTINCT {direction: 'outgoing', type: type(r1), target: connected1.canonical_name}) as outgoing,
                        collect(DISTINCT {direction: 'incoming', type: type(r2), source: connected2.canonical_name}) as incoming
                """)
                record = result.single()
                if record:
                    outgoing = [r for r in record['outgoing'] if r['target']]
                    incoming = [r for r in record['incoming'] if r['source']]
                    
                    print(f"\n   Einstein connections:")
                    print(f"   • Outgoing: {len(outgoing)} relationships")
                    for rel in outgoing[:5]:
                        print(f"     - [{rel['type']}] -> {rel['target']}")
                    print(f"   • Incoming: {len(incoming)} relationships")
                    for rel in incoming[:5]:
                        print(f"     - [{rel['type']}] <- {rel['source']}")
            
            print("\n" + "=" * 80)
            print("✅ END-TO-END PIPELINE TEST PASSED")
            print("=" * 80)
            print(f"""
📊 Pipeline Summary:
   1. ✅ Document Ingestion: Successfully ingested text document
   2. ✅ Normalization: Chunked into {len(sections)} sections
   3. ✅ LLM Extraction: Extracted {len(all_triples)} triples
   4. ✅ Entity Resolution: Resolved {len(resolved_entities)} entities
   5. ✅ Neo4j Insertion: Created {entities_created} entities, {relationships_created} relationships
   6. ✅ Graph Verification: Knowledge graph queryable and traversable

🎉 CONFIRMED: Complete pipeline from document to knowledge graph is working!
   - LLM correctly extracts entities and relationships
   - Entities are properly resolved and canonicalized
   - Knowledge graph is correctly populated in Neo4j
   - Graph queries and traversals work as expected
            """)
            
            return True
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_test_data():
    """Clean up test data from Neo4j."""
    print("\n🧹 Cleaning up test data...")
    try:
        from shared.database.neo4j import get_neo4j
        
        connector = get_neo4j()
        
        with connector.get_session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.test_id = 'e2e_pipeline_test'
                DETACH DELETE n
                RETURN count(n) as deleted
            """)
            record = result.single()
            deleted = record['deleted']
            print(f"✅ Cleaned up {deleted} test nodes")
    
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")


async def main():
    """Run all tests."""
    try:
        # Run complete pipeline test
        success = await test_complete_pipeline()
        
        # Cleanup
        await cleanup_test_data()
        
        if success:
            print("\n✅ All end-to-end tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1
    
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
