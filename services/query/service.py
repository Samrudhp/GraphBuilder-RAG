"""
Query Service with GraphVerify

Handles:
- Hybrid retrieval (FAISS + Neo4j)
- Prompt building with graph context
- DeepSeek reasoning for QA
- GraphVerify for hallucination detection
"""
import json
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from shared.models.schemas import (
    ChunkMatch,
    GraphMatch,
    HybridRetrievalResult,
    QueryRequest,
    QueryResponse,
    VerificationStatus,
)
from shared.prompts.templates import (
    QA_SYSTEM_PROMPT,
    GRAPHVERIFY_SYSTEM_PROMPT,
    format_qa_prompt,
    format_graphverify_prompt,
)
from shared.utils.ollama_client import get_ollama_client
from shared.utils.groq_client import get_groq_client

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """Hybrid retrieval combining FAISS and Neo4j."""
    
    def __init__(self):
        self.settings = get_settings().retrieval
        self.mongodb = get_mongodb()
        self.neo4j = get_neo4j()
        
        from services.embedding.service import EmbeddingPipelineService
        self.embedding_pipeline = EmbeddingPipelineService()
        
    async def retrieve(
        self,
        query: str,
        max_chunks: int = 10,
        graph_depth: int = 2,
    ) -> HybridRetrievalResult:
        """
        Retrieve relevant context using hybrid approach.
        
        Steps:
        1. FAISS semantic search for text chunks
        2. Extract entities from query
        3. Neo4j subgraph extraction
        4. Score and combine results
        
        Args:
            query: User query
            max_chunks: Max text chunks to retrieve
            graph_depth: Neo4j traversal depth
            
        Returns:
            HybridRetrievalResult with chunks and graphs
        """
        logger.info(f"Hybrid retrieval for query: {query[:100]}...")
        
        # Step 1: FAISS semantic search
        chunk_results = await self.embedding_pipeline.search_similar_chunks(
            query=query,
            top_k=max_chunks,
            min_score=self.settings.min_similarity,
        )
        
        chunks = [
            ChunkMatch(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in chunk_results
        ]
        
        logger.debug(f"Retrieved {len(chunks)} semantic chunks")
        
        # Step 2: Extract entities from query (simplified NER)
        query_entities = self._extract_entities_simple(query)
        
        # Step 3: Neo4j subgraph extraction
        graphs = []
        
        if query_entities:
            # Find entity IDs in Neo4j
            entity_ids = self._find_entity_ids(query_entities)
            
            if entity_ids:
                subgraph = self.neo4j.get_subgraph(
                    entity_ids=entity_ids,
                    depth=graph_depth,
                    min_confidence=self.settings.min_similarity,
                )
                
                if subgraph["nodes"] or subgraph["relationships"]:
                    graph_match = GraphMatch(
                        subgraph=subgraph,
                        relevance_score=0.8,  # Simplified scoring
                        node_count=len(subgraph["nodes"]),
                        edge_count=len(subgraph["relationships"]),
                    )
                    graphs.append(graph_match)
                    
                    logger.debug(
                        f"Retrieved graph: {graph_match.node_count} nodes, "
                        f"{graph_match.edge_count} edges"
                    )
        
        # Step 4: Compute combined score
        semantic_weight = self.settings.semantic_weight
        graph_weight = self.settings.graph_weight
        
        avg_chunk_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0
        avg_graph_score = sum(g.relevance_score for g in graphs) / len(graphs) if graphs else 0
        
        combined_score = (
            semantic_weight * avg_chunk_score +
            graph_weight * avg_graph_score
        )
        
        result = HybridRetrievalResult(
            chunks=chunks,
            graphs=graphs,
            combined_score=combined_score,
            retrieval_metadata={
                "query_entities": query_entities,
                "graph_depth": graph_depth,
            },
        )
        
        return result
    
    def _extract_entities_simple(self, query: str) -> list[str]:
        """
        Extract potential entities from query (simplified).
        
        In production, use NER model.
        """
        # Simple capitalized word extraction
        import re
        
        # Find capitalized phrases
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        
        # Deduplicate
        entities = list(set(entities))
        
        logger.debug(f"Extracted entities: {entities}")
        return entities
    
    def _find_entity_ids(self, entity_names: list[str]) -> list[str]:
        """Find entity IDs in Neo4j by names."""
        entity_ids = []
        
        with self.neo4j.get_session() as session:
            for name in entity_names:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.canonical_name = $name OR $name IN e.aliases
                    RETURN e.entity_id AS entity_id
                    LIMIT 1
                    """,
                    name=name,
                )
                record = result.single()
                if record:
                    entity_ids.append(record["entity_id"])
        
        return entity_ids


class PromptBuilder:
    """Build prompts with graph and text context."""
    
    def build_qa_prompt(
        self,
        question: str,
        retrieval_result: HybridRetrievalResult,
    ) -> str:
        """
        Build QA prompt with graph and text context.
        
        Args:
            question: User question
            retrieval_result: Retrieval results
            
        Returns:
            Formatted prompt
        """
        # Format graph context
        graph_context_parts = []
        
        for graph_match in retrieval_result.graphs:
            subgraph = graph_match.subgraph
            
            # Format nodes
            for node in subgraph.get("nodes", []):
                graph_context_parts.append(
                    f"Entity: {node.get('canonical_name')} "
                    f"(ID: {node.get('entity_id')}, Type: {node.get('entity_type')})"
                )
            
            # Format relationships
            for rel in subgraph.get("relationships", []):
                # Extract relationship details (Neo4j format)
                edge_id = rel.get("edge_id", "unknown")
                rel_type = type(rel).__name__
                confidence = rel.get("confidence", 0.0)
                
                graph_context_parts.append(
                    f"[Edge:{edge_id}] {rel_type} "
                    f"(confidence: {confidence:.2f})"
                )
        
        graph_context = "\n".join(graph_context_parts) if graph_context_parts else "No graph context available."
        
        # Format text chunks
        text_chunks_parts = []
        
        for i, chunk in enumerate(retrieval_result.chunks, 1):
            text_chunks_parts.append(
                f"[Doc:{chunk.document_id}, Chunk:{i}] (score: {chunk.score:.2f})\n{chunk.text}\n"
            )
        
        text_chunks = "\n".join(text_chunks_parts) if text_chunks_parts else "No text chunks available."
        
        # Build final prompt
        prompt = format_qa_prompt(
            question=question,
            graph_context=graph_context,
            text_chunks=text_chunks,
        )
        
        return prompt


class GraphVerify:
    """GraphVerify - Hallucination detection via graph verification."""
    
    def __init__(self):
        self.settings = get_settings().graphverify
        self.neo4j = get_neo4j()
        self.ollama = get_ollama_client()
        
    async def verify(
        self,
        claims: list[dict],
        retrieval_result: HybridRetrievalResult,
    ) -> dict:
        """
        Verify claims against knowledge graph.
        
        Args:
            claims: List of claims to verify
            retrieval_result: Original retrieval context
            
        Returns:
            Verification results
        """
        logger.info(f"Verifying {len(claims)} claims")
        
        # Extract graph edges from retrieval result
        graph_edges = []
        
        for graph_match in retrieval_result.graphs:
            for rel in graph_match.subgraph.get("relationships", []):
                graph_edges.append({
                    "edge_id": rel.get("edge_id"),
                    "type": type(rel).__name__,
                    "confidence": rel.get("confidence", 0.0),
                    "source": "graph",  # Simplified
                    "target": "graph",
                })
        
        # Call LLM for verification
        prompt = format_graphverify_prompt(claims, graph_edges)
        
        response = await self.ollama.generate_reasoning(
            system_prompt=GRAPHVERIFY_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=2048,
        )
        
        # Parse verification results
        verifications = response.get("verifications", [])
        overall_status = response.get("overall_status", "UNKNOWN")
        verification_score = response.get("verification_score", 0.5)
        
        return {
            "verifications": verifications,
            "overall_status": overall_status,
            "verification_score": verification_score,
        }


class QueryService:
    """Main query service coordinating QA with GraphVerify."""
    
    def __init__(self):
        self.settings = get_settings().query
        self.mongodb = get_mongodb()
        
        self.retrieval = HybridRetrievalService()
        self.prompt_builder = PromptBuilder()
        self.graphverify = GraphVerify()
        self.ollama = get_ollama_client()
        self.groq = get_groq_client()  # Fast inference for QA
        
    async def answer_question(
        self,
        request: QueryRequest,
    ) -> QueryResponse:
        """
        Answer question using graph-augmented RAG.
        
        Pipeline:
        1. Hybrid retrieval
        2. Build prompt with context
        3. Generate answer with DeepSeek
        4. Verify with GraphVerify
        5. Return response with verification
        
        Args:
            request: Query request
            
        Returns:
            Query response with answer and verification
        """
        start_time = datetime.utcnow()
        query_id = f"query_{uuid4().hex[:12]}"
        
        logger.info(f"Processing query: {query_id} - {request.question[:100]}...")
        
        try:
            # Step 1: Retrieval
            retrieval_result = await self.retrieval.retrieve(
                query=request.question,
                max_chunks=request.max_chunks,
                graph_depth=request.graph_depth,
            )
            
            # Step 2: Build prompt
            qa_prompt = self.prompt_builder.build_qa_prompt(
                question=request.question,
                retrieval_result=retrieval_result,
            )
            
            # Step 3: Generate answer with Groq (fast inference)
            logger.debug("Generating answer with Groq...")
            groq_response = await self.groq.generate(
                system_prompt=QA_SYSTEM_PROMPT,
                user_prompt=qa_prompt,
                temperature=request.temperature,
                max_tokens=self.settings.max_tokens,
            )
            
            answer = groq_response.get("response", "")
            
            # Extract claims from answer (simple sentence splitting for now)
            claims = [s.strip() for s in answer.split('.') if s.strip() and len(s.strip()) > 10]
            
            # Build sources from retrieval
            sources = []
            if retrieval_result.chunks:
                sources.extend([f"Chunk:{c.chunk_id}" for c in retrieval_result.chunks[:3]])
            if retrieval_result.graphs:
                for graph in retrieval_result.graphs[:2]:
                    sources.append(f"Graph:{graph.entity_id}")
            
            reasoning_trace = f"Groq {groq_response.get('model', '')} - {groq_response.get('usage', {}).get('total_tokens', 0)} tokens"
            
            # Step 4: Verify with GraphVerify (if requested)
            verification_status = VerificationStatus.UNKNOWN
            verification_result = None
            verified_edges = []
            contradicted_edges = []
            
            if request.require_verification and claims:
                verification_result = await self.graphverify.verify(
                    claims=claims,
                    retrieval_result=retrieval_result,
                )
                
                overall_status = verification_result.get("overall_status", "UNKNOWN")
                
                if overall_status == "SUPPORTED":
                    verification_status = VerificationStatus.SUPPORTED
                elif overall_status == "CONTRADICTED":
                    verification_status = VerificationStatus.CONTRADICTED
                elif overall_status == "UNSUPPORTED":
                    verification_status = VerificationStatus.UNSUPPORTED
                else:
                    verification_status = VerificationStatus.UNKNOWN
                
                # Extract edge IDs
                for verif in verification_result.get("verifications", []):
                    if verif.get("status") == "SUPPORTED":
                        verified_edges.extend(verif.get("supporting_edges", []))
                    elif verif.get("status") == "CONTRADICTED":
                        contradicted_edges.extend(verif.get("contradicting_edges", []))
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Build response
            response = QueryResponse(
                query_id=query_id,
                question=request.question,
                answer=answer,
                verification_status=verification_status,
                confidence=verification_result.get("verification_score", 0.5) if verification_result else 0.5,
                sources=sources,
                reasoning_trace=reasoning_trace,
                verified_edges=verified_edges,
                contradicted_edges=contradicted_edges,
                processing_time_ms=processing_time_ms,
            )
            
            logger.info(
                f"Query complete: {query_id} "
                f"(status: {verification_status}, time: {processing_time_ms}ms)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            
            # Return error response
            end_time = datetime.utcnow()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query_id=query_id,
                question=request.question,
                answer=f"An error occurred while processing your question: {str(e)}",
                verification_status=VerificationStatus.UNKNOWN,
                confidence=0.0,
                sources=[],
                processing_time_ms=processing_time_ms,
            )
