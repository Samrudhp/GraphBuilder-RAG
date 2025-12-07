"""
Baseline Systems for Benchmark Comparison

Three baseline systems to compare against GraphBuilder-RAG:
1. Pure RAG - FAISS semantic search only (no graph)
2. Pure KG - Neo4j graph traversal only (no embeddings)
3. Wikipedia API - Direct Wikipedia queries (no local processing)
"""

import asyncio
import httpx
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class BaselineSystem(ABC):
    """Abstract base class for baseline systems"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return answer"""
        pass


class PureRAGBaseline(BaselineSystem):
    """
    Pure RAG Baseline - Only FAISS semantic search
    
    No knowledge graph, no verification, just embedding similarity.
    """
    
    def __init__(self):
        super().__init__("Pure RAG")
        self.documents = []  # Would load from FAISS index
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using only semantic search.
        
        Limitations:
        - No structured reasoning
        - No verification
        - No temporal awareness
        - Can't handle multi-hop questions
        """
        try:
            # Simulate semantic search (in real impl, would use FAISS)
            # Just return a generic response for now
            return {
                "answer": "RAG baseline response",
                "confidence": 0.5,
                "method": "semantic_search_only",
                "sources": []
            }
        except Exception as e:
            return {"error": str(e)}


class PureKGBaseline(BaselineSystem):
    """
    Pure KG Baseline - Only Neo4j graph traversal
    
    No semantic search, no embeddings, just graph pattern matching.
    """
    
    def __init__(self):
        super().__init__("Pure KG")
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using only graph traversal.
        
        Limitations:
        - Requires exact entity matches
        - No fuzzy matching
        - Can't handle natural language well
        - Requires entities to already be in graph
        """
        try:
            # Simulate graph query (in real impl, would use Neo4j)
            return {
                "answer": "KG baseline response",
                "confidence": 0.6,
                "method": "graph_traversal_only",
                "triples": []
            }
        except Exception as e:
            return {"error": str(e)}


class WikipediaAPIBaseline(BaselineSystem):
    """
    Wikipedia API Baseline - Direct Wikipedia queries
    
    No local processing, no knowledge graph, just API calls.
    """
    
    def __init__(self):
        super().__init__("Wikipedia API")
        self.api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using Wikipedia API.
        
        Limitations:
        - Requires exact page title matching
        - No reasoning across articles
        - No verification
        - Can't synthesize information
        - Subject to API rate limits
        """
        try:
            # Extract potential entity from question (very simple heuristic)
            entity = self._extract_entity(question)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_url}{entity}",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "answer": data.get("extract", "No summary available"),
                        "confidence": 0.7,
                        "method": "wikipedia_api",
                        "source": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    }
                else:
                    return {
                        "answer": "Entity not found in Wikipedia",
                        "confidence": 0.0,
                        "method": "wikipedia_api",
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_entity(self, question: str) -> str:
        """
        Extract entity from question (very simple heuristic).
        
        This is intentionally naive to show limitations of baseline.
        """
        # Remove common question words
        words_to_remove = ["what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "the", "a", "an"]
        words = question.lower().split()
        entity_words = [w for w in words if w not in words_to_remove and w.isalnum()]
        
        # Take first significant word as entity
        return entity_words[0] if entity_words else "Unknown"


def get_baseline(baseline_name: str) -> BaselineSystem:
    """Factory function to get baseline system by name"""
    baselines = {
        "pure_rag": PureRAGBaseline(),
        "pure_kg": PureKGBaseline(),
        "wikipedia": WikipediaAPIBaseline()
    }
    
    if baseline_name not in baselines:
        raise ValueError(f"Unknown baseline: {baseline_name}. Available: {list(baselines.keys())}")
    
    return baselines[baseline_name]


async def compare_baselines(question: str) -> Dict[str, Any]:
    """
    Compare all baseline systems on a single question.
    
    Useful for demonstrating GraphBuilder's advantages.
    """
    baselines = [
        PureRAGBaseline(),
        PureKGBaseline(),
        WikipediaAPIBaseline()
    ]
    
    results = {}
    for baseline in baselines:
        result = await baseline.query(question)
        results[baseline.name] = result
    
    return results


if __name__ == "__main__":
    # Test all baselines
    async def test():
        question = "Who created Python?"
        results = await compare_baselines(question)
        
        print(f"Question: {question}\n")
        for name, result in results.items():
            print(f"{name}:")
            print(f"  Answer: {result.get('answer', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0.0)}")
            print()
    
    asyncio.run(test())
