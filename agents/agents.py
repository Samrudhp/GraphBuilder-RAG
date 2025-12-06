"""
Agent Framework - Autonomous agents for KG maintenance and improvement

Agents:
- ReverifyAgent: Periodically re-validate triples using external sources
- ConflictResolverAgent: Detect and resolve contradictory edges
- SchemaSuggestorAgent: Detect ontology gaps and suggest schema extensions
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from shared.models.schemas import (
    Triple,
    ValidatedTriple,
    ValidationResult,
    VerificationStatus,
)
from shared.prompts.templates import (
    CONFLICT_RESOLUTION_SYSTEM_PROMPT,
    SCHEMA_SUGGESTION_SYSTEM_PROMPT,
    format_conflict_resolution_prompt,
    format_schema_suggestion_prompt,
)
from shared.utils.groq_client import get_groq_client

logger = logging.getLogger(__name__)


# ==================== Base Agent ====================

class BaseAgent(ABC):
    """Base class for autonomous agents."""
    
    def __init__(self):
        self.settings = get_settings()
        self.mongodb = get_mongodb()
        self.neo4j = get_neo4j()
        self.groq = get_groq_client()
        self.is_running = False
    
    @abstractmethod
    async def run_cycle(self) -> Dict:
        """Execute one agent cycle. Returns summary of actions."""
        pass
    
    async def run_forever(self, interval_seconds: int):
        """Run agent continuously with specified interval."""
        self.is_running = True
        logger.info(f"{self.__class__.__name__} started with {interval_seconds}s interval")
        
        while self.is_running:
            try:
                summary = await self.run_cycle()
                logger.info(f"{self.__class__.__name__} cycle complete: {summary}")
            except Exception as e:
                logger.error(f"{self.__class__.__name__} cycle failed: {e}", exc_info=True)
            
            await asyncio.sleep(interval_seconds)
    
    def stop(self):
        """Stop the agent."""
        self.is_running = False


# ==================== ReverifyAgent ====================

class ReverifyAgent(BaseAgent):
    """
    Periodically re-validate triples using external verification.
    
    Strategy:
    1. Query validated_triples collection for triples not verified recently
    2. For each triple, attempt external verification (Wikidata, DBpedia, etc.)
    3. Update confidence scores based on new evidence
    4. Flag triples with declining confidence for human review
    """
    
    async def run_cycle(self) -> Dict:
        """Run one reverification cycle."""
        logger.info("ReverifyAgent: Starting cycle")
        
        # Find triples not verified recently
        cutoff_date = datetime.utcnow() - timedelta(
            days=7  # Default to 7 days if not configured
        )
        
        collection = self.mongodb.get_async_collection("validated_triples")
        
        cursor = collection.find({
            "$or": [
                {"last_verified": {"$lt": cutoff_date}},
                {"last_verified": {"$exists": False}},
            ],
            "validation_result.confidence": {"$gte": self.settings.validation.min_confidence},
        }).limit(self.settings.agents.reverify_batch_size)
        
        triples_to_verify = await cursor.to_list(length=None)
        
        if not triples_to_verify:
            return {"triples_checked": 0, "confidence_updated": 0, "flagged": 0}
        
        logger.info(f"ReverifyAgent: Checking {len(triples_to_verify)} triples")
        
        updated_count = 0
        flagged_count = 0
        
        for triple_doc in triples_to_verify:
            try:
                triple = ValidatedTriple(**triple_doc)
                
                # Attempt external verification
                external_confidence = await self._verify_external(triple)
                
                # Compute new confidence (weighted avg with existing)
                old_confidence = triple.validation_result.confidence
                new_confidence = (
                    0.7 * old_confidence + 0.3 * external_confidence
                )
                
                # Update triple
                await collection.update_one(
                    {"_id": triple_doc["_id"]},
                    {
                        "$set": {
                            "validation_result.confidence": new_confidence,
                            "last_verified": datetime.utcnow(),
                            "external_verification_confidence": external_confidence,
                        }
                    }
                )
                
                updated_count += 1
                
                # Flag if confidence dropped significantly
                if new_confidence < old_confidence - 0.2:
                    await self._flag_for_review(triple, old_confidence, new_confidence)
                    flagged_count += 1
                    logger.warning(
                        f"Confidence dropped for {triple.subject} -> {triple.predicate} -> {triple.object}: "
                        f"{old_confidence:.2f} -> {new_confidence:.2f}"
                    )
                
            except Exception as e:
                logger.error(f"Failed to reverify triple {triple_doc.get('_id')}: {e}")
        
        return {
            "triples_checked": len(triples_to_verify),
            "confidence_updated": updated_count,
            "flagged": flagged_count,
        }
    
    async def _verify_external(self, triple: ValidatedTriple) -> float:
        """
        Verify triple against external sources (Wikidata, DBpedia, Wikipedia).
        
        Returns confidence score [0,1] based on:
        - Wikidata: 0.9 weight (structured knowledge base)
        - DBpedia: 0.8 weight (extracted from Wikipedia)
        - Wikipedia: 0.7 weight (text search)
        """
        results = []
        
        try:
            # 1. Check Wikidata SPARQL
            wikidata_score = await self._verify_wikidata(triple)
            if wikidata_score is not None:
                results.append((wikidata_score, 0.9))
            
            # 2. Check DBpedia SPARQL
            dbpedia_score = await self._verify_dbpedia(triple)
            if dbpedia_score is not None:
                results.append((dbpedia_score, 0.8))
            
            # 3. Check Wikipedia API
            wikipedia_score = await self._verify_wikipedia(triple)
            if wikipedia_score is not None:
                results.append((wikipedia_score, 0.7))
            
            # Calculate weighted average
            if results:
                weighted_sum = sum(score * weight for score, weight in results)
                total_weight = sum(weight for _, weight in results)
                return weighted_sum / total_weight
            
            # No external verification available
            return 0.5  # Neutral confidence
            
        except Exception as e:
            logger.error(f"External verification failed: {e}")
            return 0.5
    
    async def _verify_wikidata(self, triple: ValidatedTriple) -> Optional[float]:
        """Query Wikidata SPARQL endpoint to verify triple."""
        try:
            import aiohttp
            sparql_endpoint = "https://query.wikidata.org/sparql"
            
            # Build SPARQL query based on predicate type
            query = f"""
            SELECT ?item ?itemLabel ?value ?valueLabel WHERE {{
              ?item rdfs:label "{triple.triple.subject}"@en .
              ?item ?predicate ?value .
              ?value rdfs:label "{triple.triple.object}"@en .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 10
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    sparql_endpoint,
                    params={"query": query, "format": "json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", {}).get("bindings", [])
                        
                        if results:
                            logger.info(f"Wikidata verified: {triple.triple.subject} -> {triple.triple.object}")
                            return 1.0  # Found in Wikidata
                        else:
                            return 0.3  # Not found
            
            return None
            
        except asyncio.TimeoutError:
            logger.warning("Wikidata query timeout")
            return None
        except Exception as e:
            logger.error(f"Wikidata verification error: {e}")
            return None
    
    async def _verify_dbpedia(self, triple: ValidatedTriple) -> Optional[float]:
        """Query DBpedia SPARQL endpoint to verify triple."""
        try:
            import aiohttp
            sparql_endpoint = "https://dbpedia.org/sparql"
            
            # Build SPARQL query for DBpedia
            subject_uri = triple.triple.subject.replace(" ", "_")
            object_uri = triple.triple.object.replace(" ", "_")
            
            query = f"""
            SELECT ?s ?p ?o WHERE {{
              ?s rdfs:label "{triple.triple.subject}"@en .
              ?s ?p ?o .
              ?o rdfs:label "{triple.triple.object}"@en .
            }}
            LIMIT 10
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    sparql_endpoint,
                    params={"query": query, "format": "json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", {}).get("bindings", [])
                        
                        if results:
                            logger.info(f"DBpedia verified: {triple.triple.subject} -> {triple.triple.object}")
                            return 1.0  # Found in DBpedia
                        else:
                            return 0.3  # Not found
            
            return None
            
        except asyncio.TimeoutError:
            logger.warning("DBpedia query timeout")
            return None
        except Exception as e:
            logger.error(f"DBpedia verification error: {e}")
            return None
    
    async def _verify_wikipedia(self, triple: ValidatedTriple) -> Optional[float]:
        """Search Wikipedia API to verify triple relationship."""
        try:
            import aiohttp
            wikipedia_api = "https://en.wikipedia.org/w/api.php"
            
            # Search for subject page
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    wikipedia_api,
                    params={
                        "action": "query",
                        "format": "json",
                        "titles": triple.triple.subject,
                        "prop": "extracts",
                        "exintro": 1,  # Use 1 instead of True
                        "explaintext": 1,  # Use 1 instead of True
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data.get("query", {}).get("pages", {})
                        
                        for page in pages.values():
                            extract = page.get("extract", "").lower()
                            
                            # Check if object is mentioned in subject's Wikipedia page
                            if triple.triple.object.lower() in extract:
                                logger.info(f"Wikipedia verified: {triple.triple.subject} mentions {triple.triple.object}")
                                return 0.8  # High confidence (co-occurrence)
                            else:
                                return 0.4  # Subject exists but no mention
            
            return None
            
        except asyncio.TimeoutError:
            logger.warning("Wikipedia query timeout")
            return None
        except Exception as e:
            logger.error(f"Wikipedia verification error: {e}")
            return None
    
    async def _flag_for_review(
        self,
        triple: ValidatedTriple,
        old_confidence: float,
        new_confidence: float,
    ):
        """Flag triple for human review."""
        collection = self.mongodb.get_async_collection("human_review_queue")
        
        await collection.insert_one({
            "triple_id": triple.triple_id,
            "triple": triple.model_dump(),
            "reason": "confidence_drop",
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "created_at": datetime.utcnow(),
            "status": "pending",
        })


# ==================== ConflictResolverAgent ====================

class ConflictResolverAgent(BaseAgent):
    """
    Detect and resolve contradictory edges in the knowledge graph.
    
    Strategy:
    1. Query Neo4j for entities with multiple outgoing edges of the same relationship type
    2. For each conflict, gather evidence from source documents
    3. Use LLM with CONFLICT_RESOLUTION_SYSTEM_PROMPT to resolve
    4. Deprecate incorrect edges, promote correct ones
    """
    
    async def run_cycle(self) -> Dict:
        """Run one conflict resolution cycle."""
        logger.info("ConflictResolverAgent: Starting cycle")
        
        # Find conflicts in Neo4j
        conflicts = self.neo4j.find_all_conflicts()
        
        if not conflicts:
            return {"conflicts_found": 0, "resolved": 0}
        
        logger.info(f"ConflictResolverAgent: Found {len(conflicts)} conflicts")
        
        resolved_count = 0
        
        for conflict in conflicts[:self.settings.agents.conflict_batch_size]:
            try:
                resolution = await self._resolve_conflict(conflict)
                
                if resolution:
                    await self._apply_resolution(conflict, resolution)
                    resolved_count += 1
                
            except Exception as e:
                logger.error(f"Failed to resolve conflict {conflict}: {e}")
        
        return {
            "conflicts_found": len(conflicts),
            "resolved": resolved_count,
        }
    
    async def _resolve_conflict(self, conflict: Dict) -> Optional[Dict]:
        """
        Use LLM to resolve conflict between edges.
        
        Args:
            conflict: Dict with keys:
                - entity_id: Source entity
                - relationship_type: Conflicting relationship
                - edges: List of conflicting edge dicts
        
        Returns:
            Resolution dict with winner_edge_id and reasoning, or None if unresolvable.
        """
        entity_id = conflict["entity_id"]
        relationship_type = conflict["relationship_type"]
        edges = conflict["edges"]
        
        logger.info(
            f"Resolving conflict: {entity_id} -{relationship_type}-> "
            f"{len(edges)} different targets"
        )
        
        # Gather evidence for each edge
        evidence_list = []
        for edge in edges:
            evidence = await self._gather_edge_evidence(edge["edge_id"])
            evidence_list.append({
                "edge_id": edge["edge_id"],
                "target": edge["target"],
                "confidence": edge.get("confidence", 0.0),
                "evidence": evidence,
            })
        
        # Format prompt
        prompt = format_conflict_resolution_prompt(
            entity_id=entity_id,
            relationship_type=relationship_type,
            conflicting_edges=evidence_list,
        )
        
        # Call LLM
        try:
            response = await self.groq.generate_reasoning(
                system_prompt=CONFLICT_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Response is already parsed JSON
            if "winner_edge_id" not in response:
                logger.warning("LLM did not provide winner_edge_id")
                return None
            
            return response
            
        except Exception as e:
            logger.error(f"LLM conflict resolution failed: {e}")
            return None
    
    async def _gather_edge_evidence(self, edge_id: str) -> List[Dict]:
        """Gather source documents and evidence for an edge."""
        collection = self.mongodb.get_async_collection("validated_triples")
        
        cursor = collection.find({"edge_id": edge_id})
        triples = await cursor.to_list(length=10)
        
        evidence = []
        for triple_doc in triples:
            triple = ValidatedTriple(**triple_doc)
            for ev in triple.evidence:
                evidence.append({
                    "document_id": ev.document_id,
                    "section": ev.section,
                    "text": ev.text,
                })
        
        return evidence
    
    async def _apply_resolution(self, conflict: Dict, resolution: Dict):
        """Apply conflict resolution to Neo4j."""
        winner_edge_id = resolution["winner_edge_id"]
        reasoning = resolution.get("reasoning", "")
        
        # Update edges in Neo4j
        with self.neo4j.get_session() as session:
            # Deprecate losing edges
            for edge in conflict["edges"]:
                if edge["edge_id"] != winner_edge_id:
                    session.run(
                        """
                        MATCH ()-[r]->()
                        WHERE elementId(r) = $edge_id
                        SET r.deprecated = true,
                            r.deprecated_at = datetime(),
                            r.deprecation_reason = $reason
                        """,
                        edge_id=edge["edge_id"],
                        reason=f"Conflict resolved in favor of {winner_edge_id}: {reasoning}",
                    )
            
            # Promote winner edge
            session.run(
                """
                MATCH ()-[r]->()
                WHERE elementId(r) = $edge_id
                SET r.confidence = r.confidence + 0.1,
                    r.verified = true,
                    r.verification_reason = $reason
                """,
                edge_id=winner_edge_id,
                reason=f"Conflict resolution: {reasoning}",
            )
        
        # Log resolution
        collection = self.mongodb.get_async_collection("conflict_resolutions")
        await collection.insert_one({
            "conflict": conflict,
            "resolution": resolution,
            "resolved_at": datetime.utcnow(),
        })
        
        logger.info(f"Conflict resolved: Winner edge {winner_edge_id}")


# ==================== SchemaSuggestorAgent ====================

class SchemaSuggestorAgent(BaseAgent):
    """
    Detect ontology gaps and suggest schema extensions.
    
    Strategy:
    1. Analyze extracted triples for novel predicate patterns
    2. Cluster similar predicates to identify ontology gaps
    3. Use LLM with SCHEMA_SUGGESTION_SYSTEM_PROMPT to suggest formal schema
    4. Store suggestions in schema_suggestions collection for admin review
    """
    
    async def run_cycle(self) -> Dict:
        """Run one schema suggestion cycle."""
        logger.info("SchemaSuggestorAgent: Starting cycle")
        
        # Find novel predicates
        novel_predicates = await self._find_novel_predicates()
        
        if not novel_predicates:
            return {"novel_predicates": 0, "suggestions": 0}
        
        logger.info(f"SchemaSuggestorAgent: Found {len(novel_predicates)} novel predicates")
        
        suggestions_count = 0
        
        for predicate_group in self._cluster_predicates(novel_predicates):
            try:
                suggestion = await self._generate_schema_suggestion(predicate_group)
                
                if suggestion:
                    await self._store_suggestion(suggestion)
                    suggestions_count += 1
                
            except Exception as e:
                logger.error(f"Failed to generate suggestion for {predicate_group}: {e}")
        
        return {
            "novel_predicates": len(novel_predicates),
            "suggestions": suggestions_count,
        }
    
    async def _find_novel_predicates(self) -> List[Dict]:
        """Find predicates not in current ontology."""
        # Get current ontology predicates (if available)
        # For now, use an empty set as baseline - all predicates are "novel"
        # In production, this would query the ontology schema from database
        known_predicates = set()
        
        # Try to get ontology from settings if available
        if hasattr(self.settings.validation, 'ontology_rules'):
            ontology = self.settings.validation.ontology_rules
            known_predicates = set(rule.get("predicate", "") for rule in ontology)
        
        # Get all predicates from validated triples
        collection = self.mongodb.get_async_collection("validated_triples")
        
        pipeline = [
            {"$group": {
                "_id": "$predicate",
                "count": {"$sum": 1},
                "examples": {"$push": {
                    "subject": "$subject",
                    "object": "$object",
                    "subject_type": "$subject_type",
                    "object_type": "$object_type",
                }},
            }},
            {"$match": {"count": {"$gte": self.settings.agents.min_predicate_frequency}}},
        ]
        
        cursor = collection.aggregate(pipeline)
        all_predicates = await cursor.to_list(length=None)
        
        # Filter to novel predicates
        novel = [
            p for p in all_predicates
            if p["_id"] not in known_predicates
        ]
        
        return novel
    
    def _cluster_predicates(self, predicates: List[Dict]) -> List[List[Dict]]:
        """
        Cluster similar predicates for batch suggestion.
        
        Simple string similarity clustering.
        Production should use embeddings.
        """
        clusters = []
        used = set()
        
        for i, pred in enumerate(predicates):
            if i in used:
                continue
            
            cluster = [pred]
            used.add(i)
            
            for j, other in enumerate(predicates):
                if j in used:
                    continue
                
                # Simple Levenshtein distance check
                if self._string_similarity(pred["_id"], other["_id"]) > 0.6:
                    cluster.append(other)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity (simplified)."""
        # Simple token overlap
        tokens1 = set(s1.lower().split("_"))
        tokens2 = set(s2.lower().split("_"))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    async def _generate_schema_suggestion(self, predicate_group: List[Dict]) -> Optional[Dict]:
        """Use LLM to suggest formal schema for predicate group."""
        # Format prompt
        prompt = format_schema_suggestion_prompt(predicate_group)
        
        try:
            response = await self.groq.generate_reasoning(
                system_prompt=SCHEMA_SUGGESTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=2048,
            )
            
            # Response is already parsed JSON
            return response
            
        except Exception as e:
            logger.error(f"LLM schema suggestion failed: {e}")
            return None
    
    async def _store_suggestion(self, suggestion: Dict):
        """Store schema suggestion for admin review."""
        collection = self.mongodb.get_async_collection("schema_suggestions")
        
        await collection.insert_one({
            "suggestion": suggestion,
            "status": "pending",
            "created_at": datetime.utcnow(),
        })
        
        logger.info(f"Schema suggestion stored: {suggestion.get('predicate')}")


# ==================== Agent Manager ====================

class AgentManager:
    """Manage multiple agents running concurrently."""
    
    def __init__(self):
        self.settings = get_settings()
        self.agents = []
        self.tasks = []
    
    def register_agent(self, agent: BaseAgent, interval_seconds: int):
        """Register an agent to run with specified interval."""
        self.agents.append((agent, interval_seconds))
    
    async def start_all(self):
        """Start all registered agents."""
        logger.info(f"Starting {len(self.agents)} agents")
        
        for agent, interval in self.agents:
            task = asyncio.create_task(agent.run_forever(interval))
            self.tasks.append(task)
        
        # Wait for all agents
        await asyncio.gather(*self.tasks)
    
    def stop_all(self):
        """Stop all agents."""
        logger.info("Stopping all agents")
        
        for agent, _ in self.agents:
            agent.stop()
        
        for task in self.tasks:
            task.cancel()


# ==================== CLI Entry Point ====================

async def main():
    """Run agents."""
    settings = get_settings()
    
    manager = AgentManager()
    
    # Register agents
    manager.register_agent(
        ReverifyAgent(),
        interval_seconds=settings.agents.reverify_interval_seconds,
    )
    
    manager.register_agent(
        ConflictResolverAgent(),
        interval_seconds=settings.agents.conflict_resolution_interval_seconds,
    )
    
    manager.register_agent(
        SchemaSuggestorAgent(),
        interval_seconds=settings.agents.schema_suggestion_interval_seconds,
    )
    
    # Start all
    try:
        await manager.start_all()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down agents")
        manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
