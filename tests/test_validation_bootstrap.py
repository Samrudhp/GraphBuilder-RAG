"""
Test bootstrap validation with Wikipedia and Wikidata integration.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.validation.service import ExternalVerifier, ValidationEngine
from shared.models.schemas import CandidateTriple, Triple, EvidenceSpan
from shared.config.settings import ValidationSettings


@pytest.mark.asyncio
async def test_external_verifier_imports():
    """Test that ExternalVerifier can be instantiated."""
    try:
        verifier = ExternalVerifier()
        assert verifier is not None
        assert hasattr(verifier, 'verify')
        assert hasattr(verifier, '_verify_wikipedia')
        assert hasattr(verifier, '_verify_wikidata')
        print("✅ ExternalVerifier instantiated successfully")
    except Exception as e:
        pytest.fail(f"Failed to instantiate ExternalVerifier: {e}")


@pytest.mark.asyncio
async def test_wikipedia_verification():
    """Test Wikipedia API verification."""
    verifier = ExternalVerifier()
    
    # Create a test triple about Albert Einstein
    triple = Triple(
        subject="Albert Einstein",
        predicate="born_in",
        object="1879"
    )
    
    candidate = CandidateTriple(
        triple_id="test_001",
        triple=triple,
        evidence=[EvidenceSpan(
            document_id="doc_001",
            start_char=0,
            end_char=45,
            text="Albert Einstein was born in 1879 in Germany."
        )],
        confidence=0.9,
        extraction_method="llm"
    )
    
    # Test verification
    result = await verifier.verify(candidate, graph_size=0)
    
    print(f"\n📊 Wikipedia Verification Result:")
    print(f"   Subject: {triple.subject}")
    print(f"   Predicate: {triple.predicate}")
    print(f"   Object: {triple.object}")
    print(f"   Found: {result.get('wikipedia', {}).get('found', False)}")
    print(f"   Confidence: {result.get('wikipedia', {}).get('confidence', 0.0):.2f}")
    print(f"   Snippet: {result.get('wikipedia', {}).get('snippet', 'N/A')[:100]}...")
    
    assert 'wikipedia' in result
    assert 'wikidata' in result
    # Should find Einstein in Wikipedia
    assert result['wikipedia']['found'] == True
    assert result['wikipedia']['confidence'] > 0.5
    

@pytest.mark.asyncio
async def test_wikidata_verification():
    """Test Wikidata API verification."""
    verifier = ExternalVerifier()
    
    # Create a test triple about Marie Curie
    triple = Triple(
        subject="Marie Curie",
        predicate="won",
        object="Nobel Prize"
    )
    
    candidate = CandidateTriple(
        triple_id="test_002",
        triple=triple,
        evidence=[EvidenceSpan(
            document_id="doc_002",
            start_char=0,
            end_char=59,
            text="Marie Curie won the Nobel Prize in Physics and Chemistry."
        )],
        confidence=0.85,
        extraction_method="llm"
    )
    
    result = await verifier.verify(candidate, graph_size=0)
    
    print(f"\n📊 Wikidata Verification Result:")
    print(f"   Subject: {triple.subject}")
    print(f"   Predicate: {triple.predicate}")
    print(f"   Object: {triple.object}")
    print(f"   Found: {result.get('wikidata', {}).get('found', False)}")
    print(f"   Confidence: {result.get('wikidata', {}).get('confidence', 0.0):.2f}")
    if result.get('wikidata', {}).get('entity_id'):
        print(f"   Entity ID: {result['wikidata']['entity_id']}")
    
    assert 'wikidata' in result


@pytest.mark.asyncio
async def test_caching_mechanism():
    """Test that verification results are cached."""
    verifier = ExternalVerifier()
    
    triple = Triple(
        subject="Isaac Newton",
        predicate="discovered",
        object="gravity"
    )
    
    candidate = CandidateTriple(
        triple_id="test_003",
        triple=triple,
        evidence=[EvidenceSpan(
            document_id="doc_003",
            start_char=0,
            end_char=33,
            text="Isaac Newton discovered gravity."
        )],
        confidence=0.9,
        extraction_method="llm"
    )
    
    # First call - should query APIs
    result1 = await verifier.verify(candidate, graph_size=0)
    
    # Second call - should use cache
    result2 = await verifier.verify(candidate, graph_size=0)
    
    # Results should be identical (from cache)
    assert result1 == result2
    
    # Check cache was used
    cache_key = f"{triple.subject}|{triple.predicate}|{triple.object}"
    assert cache_key in verifier.cache  # Note: cache not _cache
    
    print(f"\n✅ Caching verified - {len(verifier.cache)} items cached")


@pytest.mark.asyncio
async def test_hallucination_detection():
    """Test detection of hallucinated facts."""
    verifier = ExternalVerifier()
    
    # Create a FALSE triple (hallucination)
    triple = Triple(
        subject="Albert Einstein",
        predicate="born_in",
        object="1920"  # WRONG - he was born in 1879
    )
    
    candidate = CandidateTriple(
        triple_id="test_004",
        triple=triple,
        evidence=[EvidenceSpan(
            document_id="doc_004",
            start_char=0,
            end_char=35,
            text="Albert Einstein was born in 1920."
        )],
        confidence=0.95,  # LLM is confident but wrong
        extraction_method="llm"
    )
    
    result = await verifier.verify(candidate, graph_size=0)
    
    print(f"\n🚨 Hallucination Detection Test:")
    print(f"   LLM says: Einstein born in {triple.object}")
    print(f"   Wikipedia confidence: {result.get('wikipedia', {}).get('confidence', 0.0):.2f}")
    print(f"   Expected: Low confidence (should not confirm false date)")
    
    # Wikipedia should give low confidence for wrong date
    # or snippet should contain the correct date (1879)
    wikipedia_result = result.get('wikipedia', {})
    snippet = wikipedia_result.get('snippet', '').lower()
    
    # Either low confidence, or snippet contains correct year
    is_detected = (
        wikipedia_result.get('confidence', 0) < 0.7 or
        '1879' in snippet
    )
    
    print(f"   Hallucination detected: {is_detected}")
    assert is_detected, "Failed to detect hallucination"


if __name__ == "__main__":
    print("🧪 Running Bootstrap Validation Tests\n")
    print("=" * 60)
    
    import asyncio
    
    async def run_tests():
        print("\n1️⃣ Testing ExternalVerifier Instantiation...")
        await test_external_verifier_imports()
        
        print("\n2️⃣ Testing Wikipedia API Integration...")
        await test_wikipedia_verification()
        
        print("\n3️⃣ Testing Wikidata API Integration...")
        await test_wikidata_verification()
        
        print("\n4️⃣ Testing Caching Mechanism...")
        await test_caching_mechanism()
        
        print("\n5️⃣ Testing Hallucination Detection...")
        await test_hallucination_detection()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
    
    asyncio.run(run_tests())
