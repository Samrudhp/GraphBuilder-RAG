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
    NL2CYPHER_SYSTEM_PROMPT,
    format_qa_prompt,
    format_graphverify_prompt,
    format_nl2cypher_prompt,
)
from shared.utils.groq_client import get_groq_client

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """Hybrid retrieval combining FAISS and Neo4j with NL2Cypher."""
    
    def __init__(self):
        self.settings = get_settings().retrieval
        self.mongodb = get_mongodb()
        self.neo4j = get_neo4j()
        self.groq = get_groq_client()
        
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
        
        # Step 2: Neo4j graph retrieval with NL2Cypher
        # This is the CORE implementation for:
        # "Querying property graphs with natural language interfaces powered by LLMs"
        graphs = []
        nl2cypher_result = None
        nl2cypher_success = False
        
        try:
            # Generate Cypher query from natural language using LLM
            logger.info(f"Attempting NL2Cypher for query: {query}")
            nl2cypher_result = await self._generate_cypher_query(
                question=query,
                domain="general",
            )
            
            if nl2cypher_result and nl2cypher_result.get("cypher"):
                logger.info(f"NL2Cypher generated: {nl2cypher_result.get('cypher')}")
                logger.info(f"NL2Cypher parameters: {nl2cypher_result.get('parameters', {})}")
                # Execute LLM-generated Cypher query
                subgraph = await self._execute_cypher_query(
                    cypher=nl2cypher_result["cypher"],
                    parameters=nl2cypher_result.get("parameters", {}),
                )
                
                if subgraph and (subgraph.get("nodes") or subgraph.get("relationships")):
                    graph_match = GraphMatch(
                        subgraph=subgraph,
                        relevance_score=0.9,  # High score for LLM-generated queries
                        node_count=len(subgraph.get("nodes", [])),
                        edge_count=len(subgraph.get("relationships", [])),
                    )
                    graphs.append(graph_match)
                    nl2cypher_success = True
                    
                    logger.info(
                        f"NL2Cypher retrieved: {graph_match.node_count} nodes, "
                        f"{graph_match.edge_count} edges - {nl2cypher_result.get('explanation')}"
                    )
                else:
                    logger.warning("NL2Cypher returned empty subgraph, will try entity extraction fallback")
            else:
                logger.warning("NL2Cypher did not generate a valid Cypher query, will try entity extraction fallback")
        except Exception as e:
            logger.warning(f"NL2Cypher failed with exception, will try entity extraction fallback: {e}", exc_info=True)
        
        # Try entity extraction fallback if NL2Cypher didn't return results
        if not nl2cypher_success:
            logger.info("Attempting entity extraction fallback")
            query_entities = self._extract_entities_simple(query)
            
            if query_entities:
                entity_ids = self._find_entity_ids(query_entities)
                
                if entity_ids:
                    logger.info(f"Fetching subgraph for entity IDs: {entity_ids} with depth={graph_depth}, min_confidence={self.settings.min_similarity}")
                    subgraph = self.neo4j.get_subgraph(
                        entity_ids=entity_ids,
                        depth=graph_depth,
                        min_confidence=self.settings.min_similarity,
                    )
                    
                    logger.info(f"Subgraph retrieved: {len(subgraph.get('nodes', []))} nodes, {len(subgraph.get('relationships', []))} relationships")
                    
                    if subgraph["nodes"] or subgraph["relationships"]:
                        graph_match = GraphMatch(
                            subgraph=subgraph,
                            relevance_score=0.7,  # Lower score for fallback
                            node_count=len(subgraph["nodes"]),
                            edge_count=len(subgraph["relationships"]),
                        )
                        graphs.append(graph_match)
                        
                        logger.info(
                            f"Fallback entity retrieval: {graph_match.node_count} nodes, "
                            f"{graph_match.edge_count} edges"
                        )
                    else:
                        logger.warning("Entity extraction fallback returned empty subgraph")
                else:
                    logger.warning(f"No entity IDs found for entities: {query_entities}")
            else:
                logger.warning(f"No entities extracted from query: {query}")
        
        # Step 3: Compute combined score
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
                "nl2cypher_used": bool(nl2cypher_result),
                "graph_depth": graph_depth,
            },
        )
        
        return result
    
    async def _generate_cypher_query(
        self,
        question: str,
        domain: str = "general",
    ) -> Optional[dict]:
        """
        Generate Cypher query from natural language using LLM.
        
        This implements NL2Cypher for the conference paper:
        "Querying property graphs with natural language interfaces powered by LLMs"
        
        Args:
            question: Natural language question
            domain: Domain context
            
        Returns:
            Dict with cypher, parameters, explanation
        """
        try:
            # Extract potential entities for context
            entities = self._extract_entities_simple(question)
            
            # Format NL2Cypher prompt
            user_prompt = format_nl2cypher_prompt(
                question=question,
                domain=domain,
                entities=entities,
            )
            
            # Call LLM to generate Cypher
            result = await self.groq.generate_cypher(
                question=question,
                system_prompt=NL2CYPHER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,  # Low for precise queries
                max_tokens=1024,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}", exc_info=True)
            return None
    
    async def _execute_cypher_query(
        self,
        cypher: str,
        parameters: dict = None,
    ) -> Optional[dict]:
        """
        Execute LLM-generated Cypher query on Neo4j.
        
        Args:
            cypher: Cypher query string
            parameters: Query parameters
            
        Returns:
            Subgraph dict with nodes and relationships
        """
        try:
            # Execute query using Neo4j driver
            with self.neo4j.driver.session() as session:
                result = session.run(cypher, parameters or {})
                
                # Parse results into nodes and relationships
                nodes = []
                relationships = []
                seen_nodes = set()
                seen_rels = set()
                
                for record in result:
                    # Extract nodes and relationships from the record
                    for key, value in record.items():
                        if hasattr(value, 'labels'):  # It's a node
                            node_id = value.element_id
                            if node_id not in seen_nodes:
                                nodes.append({
                                    "entity_id": value.get("entity_id", node_id),
                                    "canonical_name": value.get("canonical_name", ""),
                                    "entity_type": value.get("entity_type", "Entity"),
                                    "aliases": value.get("aliases", []),
                                })
                                seen_nodes.add(node_id)
                        
                        elif hasattr(value, 'type'):  # It's a relationship
                            rel_id = value.element_id
                            if rel_id not in seen_rels:
                                relationships.append({
                                    "edge_id": value.get("edge_id", rel_id),
                                    "source_id": value.start_node.get("entity_id"),
                                    "target_id": value.end_node.get("entity_id"),
                                    "relationship_type": value.type,
                                    "confidence": value.get("confidence", 0.5),
                                    "evidence_ids": value.get("evidence_ids", []),
                                })
                                seen_rels.add(rel_id)
                
                return {
                    "nodes": nodes,
                    "relationships": relationships,
                }
                
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}", exc_info=True)
            return None
    
    def _extract_entities_simple(self, query: str) -> list[str]:
        """
        Extract potential entities from query (simplified).
        
        In production, use NER model.
        """
        import re
        
        entities = []
        
        # Remove possessive 's to avoid "Einstein's" being different from "Einstein"
        clean_query = re.sub(r"'s\b", "", query)
        
        # Find capitalized phrases (e.g., "Albert Einstein")
        # Limit to 2-3 words to avoid capturing full sentences
        capitalized_patterns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", clean_query)
        entities.extend(capitalized_patterns)
        
        # Also find common patterns even if lowercase (for queries like "who was isaac newton")
        # Match exactly 2 or 3 consecutive words that might be names
        lowercase_query = clean_query.lower()
        
        # Extract 2-word patterns
        two_word_patterns = re.findall(r"\b([a-z]+\s+[a-z]+)\b", lowercase_query)
        
        # Extract 3-word patterns
        three_word_patterns = re.findall(r"\b([a-z]+\s+[a-z]+\s+[a-z]+)\b", lowercase_query)
        
        # Common stop words to skip
        stop_words = {
            'was', 'is', 'the', 'and', 'but', 'for', 'who', 'what', 'where', 'when', 
            'why', 'how', 'did', 'won', 'which', 'have', 'has', 'had', 'about', 'this',
            'that', 'from', 'with', 'are', 'were', 'been', 'being', 'have', 'his', 'her',
            'their', 'your', 'our', 'work', 'works', 'award', 'awards', 'prize'
        }
        
        # Process patterns
        for pattern in two_word_patterns + three_word_patterns:
            words = pattern.split()
            # Skip if starts with stop word or ends with common noun
            if words[0] not in stop_words and words[-1] not in stop_words:
                # Capitalize to match Neo4j format
                capitalized = ' '.join(word.capitalize() for word in words)
                entities.append(capitalized)
        
        # Deduplicate and clean
        entities = list(set(entities))
        
        # Remove entities that are substrings of others (keep longer ones)
        filtered_entities = []
        for entity in sorted(entities, key=len, reverse=True):
            if not any(entity in other and entity != other for other in filtered_entities):
                filtered_entities.append(entity)
        
        logger.info(f"Extracted entities from query '{query}': {filtered_entities}")
        return filtered_entities
    
    def _find_entity_ids(self, entity_names: list[str]) -> list[str]:
        """Find entity IDs in Neo4j by names with fuzzy matching."""
        entity_ids = []
        
        with self.neo4j.get_session() as session:
            for name in entity_names:
                # Try exact match first (case-insensitive)
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.canonical_name) = toLower($name) 
                       OR ANY(alias IN e.aliases WHERE toLower(alias) = toLower($name))
                    RETURN e.entity_id AS entity_id, e.canonical_name AS canonical_name
                    LIMIT 1
                    """,
                    name=name,
                )
                record = result.single()
                
                # If exact match fails, try partial CONTAINS match for typos
                if not record:
                    logger.info(f"Exact match failed for '{name}', trying partial match...")
                    # For multi-word names, try matching individual words
                    words = name.split()
                    if len(words) >= 2:
                        # Try matching with at least 2 words from the name
                        result = session.run(
                            """
                            MATCH (e:Entity)
                            WHERE ALL(word IN $words WHERE toLower(e.canonical_name) CONTAINS toLower(word))
                            RETURN e.entity_id AS entity_id, e.canonical_name AS canonical_name
                            ORDER BY size(e.canonical_name) ASC
                            LIMIT 1
                            """,
                            words=words,
                        )
                        record = result.single()
                
                if record:
                    entity_ids.append(record["entity_id"])
                    logger.info(f"Found entity: '{name}' -> '{record['canonical_name']}' (ID: {record['entity_id']})")
                else:
                    logger.warning(f"Entity not found in Neo4j: '{name}'")
        
        logger.info(f"Extracted {len(entity_ids)} entity IDs from {len(entity_names)} names")
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
        self.groq = get_groq_client()
        
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
        
        response = await self.groq.generate_reasoning(
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
            
            answer_raw = groq_response.get("response", "")
            
            # Parse the JSON response from LLM
            answer_obj = {}
            claims = []
            reasoning_trace = ""
            
            try:
                # Try to parse as JSON
                answer_obj = json.loads(answer_raw)
                answer = answer_obj.get("answer", answer_raw)
                claims = answer_obj.get("claims", [])
                reasoning_trace = answer_obj.get("reasoning_trace", "")
                logger.debug(f"Parsed structured answer with {len(claims)} claims")
            except json.JSONDecodeError:
                # Fallback: treat as plain text
                logger.warning("LLM response is not valid JSON, using as plain text")
                answer = answer_raw
                answer_obj = {
                    "answer": answer_raw,
                    "claims": [],
                    "sources": [],
                    "reasoning_trace": "LLM returned plain text instead of structured JSON"
                }
                # Extract claims from plain text (simple sentence splitting)
                claims_text = [s.strip() for s in answer_raw.split('.') if s.strip() and len(s.strip()) > 10]
                claims = [{"claim": c, "evidence_type": "text", "evidence_ids": [], "confidence": 0.5} for c in claims_text[:5]]
            
            # Build sources from retrieval
            sources = []
            if retrieval_result.chunks:
                sources.extend([f"Chunk:{c.chunk_id}" for c in retrieval_result.chunks[:3]])
            if retrieval_result.graphs:
                for graph in retrieval_result.graphs[:2]:
                    # Extract entity IDs from the subgraph nodes
                    for node in graph.subgraph.get("nodes", [])[:3]:
                        entity_id = node.get("entity_id", "unknown")
                        sources.append(f"Entity:{entity_id}")
            
            # Add sources to answer object
            answer_obj["sources"] = sources
            
            if not reasoning_trace:
                reasoning_trace = f"Groq {groq_response.get('model', '')} - {groq_response.get('usage', {}).get('total_tokens', 0)} tokens"
                answer_obj["reasoning_trace"] = reasoning_trace
            
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
                answer=answer_obj,  # Return the full structured object
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
