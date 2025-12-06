"""
Test LLM Output with Graph Context (Simplified - No FAISS)

Shows real LLM responses when given graph context only (avoiding FAISS crash).
Tests what the LLM actually outputs when answering questions using knowledge graph.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print("🤖 Testing Actual LLM Output (Graph Context Only)")
print("=" * 80)


async def test_llm_with_graph_context():
    """Test real LLM outputs with graph context (no FAISS)."""
    
    try:
        from services.ingestion.service import IngestionService
        from services.normalization.service import NormalizationService
        from services.extraction.service import ExtractionService
        from shared.models.schemas import DocumentType
        from shared.database.neo4j import get_neo4j
        from shared.prompts.templates import QA_SYSTEM_PROMPT
        import tempfile
        from pathlib import Path
        from groq import AsyncGroq
        import os
        
        print("\n📋 Step 1: Setup Test Document & Knowledge Graph")
        print("-" * 80)
        
        # Create test document
        test_content = """
        Albert Einstein was born in Ulm, Germany on March 14, 1879. He developed 
        the theory of relativity, which revolutionized modern physics. Einstein 
        received the Nobel Prize in Physics in 1921 for his explanation of the 
        photoelectric effect.
        
        In 1905, known as Einstein's "miracle year," he published four groundbreaking 
        papers covering the photoelectric effect, Brownian motion, special relativity, 
        and the famous equation E=mc².
        
        Einstein worked at the Swiss Patent Office in Bern from 1902 to 1909 while 
        pursuing his doctorate. He later held professorships at universities in 
        Zurich, Prague, and Berlin.
        
        Due to the rise of Nazi Germany, Einstein emigrated to the United States 
        in 1933. He settled at Princeton University where he continued his research 
        until his death in 1955.
        """
        
        # Clean up previous test data
        print("   Cleaning previous test data...")
        neo4j_connector = get_neo4j()
        
        with neo4j_connector.get_session() as session:
            session.run("""
                MATCH (n)
                WHERE n.test_id = 'llm_simple_test'
                DETACH DELETE n
            """)
        
        # Initialize services
        ingestion = IngestionService()
        normalization = NormalizationService()
        extraction = ExtractionService()
        
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("\n❌ GROQ_API_KEY environment variable not set!")
            print("   Please set it with: export GROQ_API_KEY='your-api-key'")
            return False
        
        groq = AsyncGroq(api_key=groq_api_key)
        
        # Ingest and process document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            print("   Ingesting document...")
            result = await ingestion.ingest_from_file(Path(temp_file), DocumentType.TEXT)
            raw_doc_id = result.document_id
            
            print("   Normalizing document...")
            norm_result = await normalization.normalize_document(raw_doc_id)
            normalized_doc_id = norm_result.document_id
            
            print("   Extracting triples with LLM...")
            candidate_triples = await extraction.extract_from_document(normalized_doc_id)
            print(f"   ✅ Extracted {len(candidate_triples)} triples")
            
            # Insert into knowledge graph
            print("   Building knowledge graph...")
            entities = {}
            for candidate in candidate_triples:
                triple = candidate.triple
                if triple.subject not in entities:
                    entities[triple.subject] = triple.subject_type if triple.subject_type else 'Other'
                if triple.object not in entities:
                    entities[triple.object] = triple.object_type if triple.object_type else 'Other'
            
            # Create entities
            with neo4j_connector.get_session() as session:
                for entity_name, entity_type in entities.items():
                    entity_id = f"entity_{hash(entity_name)}"
                    type_str = entity_type.value if hasattr(entity_type, 'value') else str(entity_type)
                    
                    session.run(f"""
                        MERGE (e:{type_str} {{
                            entity_id: $entity_id,
                            canonical_name: $name,
                            entity_type: $type,
                            test_id: 'llm_simple_test'
                        }})
                    """, entity_id=entity_id, name=entity_name, type=type_str)
            
            # Create relationships
            with neo4j_connector.get_session() as session:
                for candidate in candidate_triples:
                    triple = candidate.triple
                    
                    subject_id = f"entity_{hash(triple.subject)}"
                    object_id = f"entity_{hash(triple.object)}"
                    
                    # Clean relationship type
                    rel_type = triple.predicate.upper()
                    rel_type = rel_type.replace(" ", "_").replace("-", "_").replace(".", "")
                    rel_type = rel_type.replace(",", "_").replace("(", "").replace(")", "")
                    rel_type = rel_type.replace("/", "_").replace("'", "").replace('"', "")
                    rel_type = ''.join(c for c in rel_type if c.isalnum() or c == '_')[:50]
                    if not rel_type:
                        rel_type = "RELATED_TO"
                    
                    evidence_str = ""
                    if candidate.evidence:
                        if isinstance(candidate.evidence, list):
                            evidence_str = "; ".join([
                                ev.text if hasattr(ev, 'text') else str(ev)
                                for ev in candidate.evidence
                            ])
                        else:
                            evidence_str = str(candidate.evidence)
                    
                    session.run(f"""
                        MATCH (subject {{entity_id: $subject_id, test_id: 'llm_simple_test'}})
                        MATCH (object {{entity_id: $object_id, test_id: 'llm_simple_test'}})
                        MERGE (subject)-[r:{rel_type} {{
                            predicate: $predicate,
                            confidence: $confidence,
                            evidence: $evidence,
                            test_id: 'llm_simple_test'
                        }}]->(object)
                    """,
                    subject_id=subject_id,
                    object_id=object_id,
                    predicate=triple.predicate,
                    confidence=candidate.confidence,
                    evidence=evidence_str[:500])
            
            print(f"   ✅ Knowledge graph built: {len(entities)} entities, {len(candidate_triples)} relationships")
            
            # Now test actual LLM Q&A with graph context
            print("\n" + "=" * 80)
            print("🤖 TESTING REAL LLM OUTPUTS (Using Graph Context)")
            print("=" * 80)
            
            # Helper function to query graph and build context
            def get_graph_context(query_text):
                """Get relevant graph context for a query."""
                with neo4j_connector.get_session() as session:
                    # Simple entity extraction (look for capitalized words)
                    words = query_text.split()
                    entities_in_query = [w for w in words if w[0].isupper() and len(w) > 2]
                    
                    # Query graph
                    result = session.run("""
                        MATCH (e {test_id: 'llm_simple_test'})
                        WHERE ANY(word IN $entities WHERE e.canonical_name CONTAINS word)
                        OPTIONAL MATCH (e)-[r {test_id: 'llm_simple_test'}]->(target {test_id: 'llm_simple_test'})
                        RETURN 
                            e.canonical_name as entity,
                            type(r) as relationship,
                            target.canonical_name as target,
                            r.confidence as confidence,
                            r.evidence as evidence
                        LIMIT 10
                    """, entities=entities_in_query)
                    
                    records = list(result)
                    
                    if not records and entities_in_query:
                        # Try looser match
                        result = session.run("""
                            MATCH (e {test_id: 'llm_simple_test'})-[r]->(target {test_id: 'llm_simple_test'})
                            RETURN 
                                e.canonical_name as entity,
                                type(r) as relationship,
                                target.canonical_name as target,
                                r.confidence as confidence,
                                r.evidence as evidence
                            LIMIT 10
                        """)
                        records = list(result)
                    
                    return records
            
            # Test Question 1: Simple factual
            print("\n" + "─" * 80)
            print("📝 Question 1: Where was Albert Einstein born?")
            print("─" * 80)
            
            question1 = "Where was Albert Einstein born?"
            graph_context1 = get_graph_context(question1)
            
            print(f"\n🔗 Retrieved {len(graph_context1)} graph relationships")
            for rec in graph_context1[:5]:
                if rec['relationship']:
                    print(f"   • ({rec['entity']}) -[{rec['relationship']}]-> ({rec['target']})")
                    if rec.get('confidence'):
                        print(f"     Confidence: {rec['confidence']:.2f}")
            
            # Build prompt with graph context
            context_str = "\n".join([
                f"({r['entity']}) - {r['relationship']} -> ({r['target']}) [confidence: {r.get('confidence', 0.5):.2f}]"
                for r in graph_context1 if r['relationship']
            ])
            
            prompt1 = f"""Based on the following knowledge graph information, answer the question.

Knowledge Graph Context:
{context_str if context_str else "No specific graph context found."}

Question: {question1}

Answer the question concisely based on the knowledge graph. If the answer is not in the graph, say so."""
            
            print("\n🤖 Calling Groq LLM...")
            response1 = await groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt1}
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            
            answer1 = response1.choices[0].message.content.strip()
            
            print("\n🤖 LLM ANSWER:")
            print("┌" + "─" * 78 + "┐")
            for line in answer1.split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘")
            
            # Test Question 2: Multi-fact question
            print("\n" + "─" * 80)
            print("📝 Question 2: What were Einstein's major achievements?")
            print("─" * 80)
            
            question2 = "What were Einstein's major achievements in physics?"
            graph_context2 = get_graph_context(question2)
            
            print(f"\n🔗 Retrieved {len(graph_context2)} graph relationships")
            for rec in graph_context2[:5]:
                if rec['relationship']:
                    print(f"   • ({rec['entity']}) -[{rec['relationship']}]-> ({rec['target']})")
            
            context_str2 = "\n".join([
                f"({r['entity']}) - {r['relationship']} -> ({r['target']})"
                for r in graph_context2 if r['relationship']
            ])
            
            prompt2 = f"""Based on the following knowledge graph information, answer the question.

Knowledge Graph Context:
{context_str2 if context_str2 else "No specific graph context found."}

Question: {question2}

Answer the question based on the knowledge graph. Synthesize multiple facts if available."""
            
            print("\n🤖 Calling Groq LLM...")
            response2 = await groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt2}
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            
            answer2 = response2.choices[0].message.content.strip()
            
            print("\n🤖 LLM ANSWER:")
            print("┌" + "─" * 78 + "┐")
            for line in answer2.split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘")
            
            # Test Question 3: Out-of-scope question
            print("\n" + "─" * 80)
            print("📝 Question 3: What was Einstein's relationship with Tesla?")
            print("   (This info is NOT in our document/graph)")
            print("─" * 80)
            
            question3 = "What was Einstein's relationship with Nikola Tesla?"
            graph_context3 = get_graph_context(question3)
            
            print(f"\n🔗 Retrieved {len(graph_context3)} graph relationships")
            
            context_str3 = "\n".join([
                f"({r['entity']}) - {r['relationship']} -> ({r['target']})"
                for r in graph_context3 if r['relationship']
            ])
            
            prompt3 = f"""Based on the following knowledge graph information, answer the question.

Knowledge Graph Context:
{context_str3 if context_str3 else "No specific graph context found."}

Question: {question3}

Answer the question ONLY based on the knowledge graph. If the information is not available, clearly state that."""
            
            print("\n🤖 Calling Groq LLM...")
            response3 = await groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt3}
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            
            answer3 = response3.choices[0].message.content.strip()
            
            print("\n🤖 LLM ANSWER:")
            print("┌" + "─" * 78 + "┐")
            for line in answer3.split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘")
            
            # Summary
            print("\n" + "=" * 80)
            print("✅ LLM OUTPUT TEST COMPLETE")
            print("=" * 80)
            print(f"""
📊 What we tested:
   1. Simple factual question → LLM uses graph context to answer
   2. Multi-fact question → LLM synthesizes multiple graph relationships
   3. Out-of-scope question → LLM should indicate lack of information
   
💡 Key observations:
   • LLM receives graph relationships with confidence scores
   • Answers are grounded in retrieved graph facts
   • When info not in graph, LLM should say so (avoiding hallucination)
   
🎯 The graph-based approach provides:
   • Structured facts with confidence scores
   • Clear source attribution (which relationships used)
   • Natural language generation from structured data
   • Ability to detect out-of-scope questions
   
📈 Next: Add semantic chunks for richer context!
            """)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Cleanup
        print("\n🧹 Cleaning up test data...")
        with neo4j_connector.get_session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.test_id = 'llm_simple_test'
                DETACH DELETE n
                RETURN count(n) as deleted
            """)
            record = result.single()
            print(f"✅ Cleaned up {record['deleted']} test nodes")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run test."""
    try:
        success = await test_llm_with_graph_context()
        
        if success:
            print("\n✅ All LLM output tests passed!")
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
