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
        
        # FIRST: Try simple direct query (most reliable)
        try:
            logger.info(f"Attempting direct entity query for: {query}")
            query_entities = self._extract_entities_simple(query)
            
            if query_entities:
                entity_ids = self._find_entity_ids(query_entities)
                
                if entity_ids:
                    logger.info(f"Direct query: Fetching subgraph for {len(entity_ids)} entities")
                    subgraph = self.neo4j.get_subgraph(
                        entity_ids=entity_ids,
                        depth=graph_depth,
                        min_confidence=self.settings.min_similarity,
                    )
                    
                    if subgraph and (subgraph.get("nodes") or subgraph.get("relationships")):
                        graph_match = GraphMatch(
                            subgraph=subgraph,
                            relevance_score=0.85,  # High score for direct entity match
                            node_count=len(subgraph.get("nodes", [])),
                            edge_count=len(subgraph.get("relationships", [])),
                        )
                        graphs.append(graph_match)
                        nl2cypher_success = True
                        
                        logger.info(
                            f"Direct query retrieved: {graph_match.node_count} nodes, "
                            f"{graph_match.edge_count} edges"
                        )
        except Exception as e:
            logger.warning(f"Direct entity query failed: {e}", exc_info=True)
        
        # SECOND: Try NL2Cypher if direct query didn't work
        if not nl2cypher_success:
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
                        logger.warning("NL2Cypher returned empty subgraph")
                else:
                    logger.warning("NL2Cypher did not generate a valid Cypher query")
            except Exception as e:
                logger.warning(f"NL2Cypher failed with exception: {e}", exc_info=True)
        
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
                        # Handle paths
                        if hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                            # It's a path object
                            for node in value.nodes:
                                node_id = node.element_id
                                if node_id not in seen_nodes:
                                    nodes.append({
                                        "entity_id": node.get("entity_id", node_id),
                                        "canonical_name": node.get("canonical_name", ""),
                                        "entity_type": node.get("entity_type", "Entity"),
                                        "aliases": node.get("aliases", []),
                                    })
                                    seen_nodes.add(node_id)
                            
                            for rel in value.relationships:
                                rel_id = rel.element_id
                                if rel_id not in seen_rels:
                                    relationships.append({
                                        "edge_id": rel.get("edge_id", rel_id),
                                        "source_id": rel.start_node.get("entity_id"),
                                        "target_id": rel.end_node.get("entity_id"),
                                        "relationship_type": rel.type,
                                        "confidence": rel.get("confidence", 0.5),
                                        "evidence_ids": rel.get("evidence_ids", []),
                                    })
                                    seen_rels.add(rel_id)
                        
                        elif hasattr(value, 'labels'):  # It's a node
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
                        
                        # Handle lists of nodes/relationships
                        elif isinstance(value, list):
                            for item in value:
                                if hasattr(item, 'labels'):  # Node
                                    node_id = item.element_id
                                    if node_id not in seen_nodes:
                                        nodes.append({
                                            "entity_id": item.get("entity_id", node_id),
                                            "canonical_name": item.get("canonical_name", ""),
                                            "entity_type": item.get("entity_type", "Entity"),
                                            "aliases": item.get("aliases", []),
                                        })
                                        seen_nodes.add(node_id)
                                elif hasattr(item, 'type'):  # Relationship
                                    rel_id = item.element_id
                                    if rel_id not in seen_rels:
                                        relationships.append({
                                            "edge_id": item.get("edge_id", rel_id),
                                            "source_id": item.start_node.get("entity_id"),
                                            "target_id": item.end_node.get("entity_id"),
                                            "relationship_type": item.type,
                                            "confidence": item.get("confidence", 0.5),
                                            "evidence_ids": item.get("evidence_ids", []),
                                        })
                                        seen_rels.add(rel_id)
                
                logger.info(f"Cypher execution: {len(nodes)} nodes, {len(relationships)} relationships")
                
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
        
        # Find capitalized phrases (e.g., "Albert Einstein", "Nobel Prize")
        # This catches proper nouns which are usually entities
        capitalized_patterns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", clean_query)
        
        # Aggressive stop words and common words to filter
        stop_words = {
            'was', 'is', 'the', 'and', 'but', 'for', 'who', 'what', 'where', 'when', 
            'why', 'how', 'did', 'won', 'which', 'have', 'has', 'had', 'about', 'this',
            'that', 'from', 'with', 'are', 'were', 'been', 'being', 'his', 'her',
            'their', 'your', 'our', 'work', 'works', 'award', 'awards', 'prize',
            'in', 'on', 'at', 'to', 'of', 'by', 'as'
        }
        
        # Filter capitalized patterns to remove noise
        for pattern in capitalized_patterns:
            words = pattern.lower().split()
            # Only keep if:
            # 1. Not a single common word
            # 2. Doesn't contain only stop words
            # 3. Has at least one meaningful word
            if len(words) == 1 and words[0] in stop_words:
                continue
            if all(w in stop_words for w in words):
                continue
            
            # Remove leading/trailing stop words
            while words and words[0] in stop_words:
                words.pop(0)
            while words and words[-1] in stop_words:
                words.pop()
            
            if words:  # If anything left after filtering
                cleaned = ' '.join(words)
                # Capitalize properly
                capitalized = ' '.join(word.capitalize() for word in cleaned.split())
                entities.append(capitalized)
        
        # Deduplicate
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
                # Try CONTAINS match first (most flexible)
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.canonical_name) CONTAINS toLower($name) 
                       OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($name))
                    RETURN e.entity_id AS entity_id, e.canonical_name AS canonical_name
                    ORDER BY size(e.canonical_name) ASC
                    LIMIT 5
                    """,
                    name=name,
                )
                records = list(result)
                
                if records:
                    # Take all matches (up to 5) for better coverage
                    for record in records:
                        entity_ids.append(record["entity_id"])
                        logger.info(f"Matched '{name}' → '{record['canonical_name']}' ({record['entity_id']})")
                else:
                    logger.warning(f"No match found for entity: '{name}'")
        
        logger.info(f"Found {len(entity_ids)} entity IDs: {entity_ids}")
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
                max_tokens=2048,  # Ensure enough tokens for full answer
            )
            
            answer_raw = groq_response.get("response", "")
            
            # Clean response - remove markdown code blocks if present
            answer_raw = answer_raw.strip()
            if answer_raw.startswith("```json"):
                answer_raw = answer_raw[7:]  # Remove ```json
            if answer_raw.startswith("```"):
                answer_raw = answer_raw[3:]  # Remove ```
            if answer_raw.endswith("```"):
                answer_raw = answer_raw[:-3]  # Remove trailing ```
            answer_raw = answer_raw.strip()
            
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
            
            # If no claims extracted, create one from the answer
            if not claims and answer:
                logger.info("No claims extracted, creating default claim from answer")
                claims = [{
                    "claim": str(answer)[:200],  # First 200 chars of answer
                    "evidence_type": "mixed",
                    "evidence_ids": sources,
                    "confidence": 0.7
                }]
            
            if request.require_verification and claims:
                logger.info(f"Running GraphVerify on {len(claims)} claims")
                verification_result = await self.graphverify.verify(
                    claims=claims,
                    retrieval_result=retrieval_result,
                )
                
                logger.info(f"GraphVerify result: {verification_result.get('overall_status', 'UNKNOWN')}, score: {verification_result.get('verification_score', 0.0)}")
                
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
            
            # Calculate confidence score
            confidence_score = 0.5  # Default baseline
            
            if verification_result:
                # Use GraphVerify score if available
                confidence_score = verification_result.get("verification_score", 0.5)
                # Set minimum floor - never below 0.5 if we have any retrieval results
                if (retrieval_result.graphs or retrieval_result.chunks) and confidence_score < 0.5:
                    confidence_score = 0.5
            elif retrieval_result.graphs and retrieval_result.chunks:
                # Has both graph and text - medium-high confidence
                confidence_score = 0.7
            elif retrieval_result.graphs:
                # Has graph only - medium confidence
                confidence_score = 0.65
            elif retrieval_result.chunks:
                # Has text only - medium-low confidence
                confidence_score = 0.55
            
            # Build response
            response = QueryResponse(
                query_id=query_id,
                question=request.question,
                answer=answer_obj,  # Return the full structured object
                verification_status=verification_status,
                confidence=confidence_score,
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
