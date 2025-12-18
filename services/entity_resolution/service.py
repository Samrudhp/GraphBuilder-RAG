"""
Entity Resolution Service - Links and deduplicates entities.

Handles:
- Alias matching
- FAISS similarity search for entity names
- Provisional entity creation
- Entity merging and canonicalization
"""
import asyncio
import logging
from typing import Optional
from uuid import uuid4

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from shared.models.schemas import Entity, EntityType, ProvisionalEntity

logger = logging.getLogger(__name__)


class EntityResolutionService:
    """Service for entity resolution and canonicalization."""
    
    def __init__(self):
        self.settings = get_settings().entity_resolution
        self.mongodb = get_mongodb()
        self.neo4j = get_neo4j()
        
        self.provisional_entities = self.mongodb.get_async_collection("provisional_entities")
        
        # Entity name embedding service (reuse embedding service)
        from services.embedding.service import EmbeddingService
        self.embedding_service = EmbeddingService()
        
    async def resolve_entity(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> str:
        """
        Resolve an entity name to a canonical entity ID.
        
        Strategy:
        1. Check for exact alias match in Neo4j
        2. Check provisional entities in MongoDB
        3. Use FAISS similarity search
        4. Create new entity if no match (with locking to prevent duplicates)
        
        Args:
            name: Entity name
            entity_type: Optional entity type hint
            
        Returns:
            Canonical entity ID
        """
        logger.debug(f"Resolving entity: {name}")
        
        # Step 1: Check Neo4j for exact match (alias or canonical name)
        entity_id = self._check_neo4j_exact_match(name)
        if entity_id:
            logger.debug(f"Found exact match in Neo4j: {entity_id}")
            return entity_id
        
        # Step 2: Check provisional entities
        provisional = await self.provisional_entities.find_one({
            "name": name,
            "resolution_status": "resolved",
        })
        if provisional and provisional.get("resolved_to"):
            logger.debug(f"Found in provisional entities: {provisional['resolved_to']}")
            # Double-check Neo4j to ensure entity exists
            verify_id = self._check_neo4j_exact_match(provisional["resolved_to"])
            if verify_id:
                return verify_id
            else:
                logger.warning(f"Provisional entity {provisional['resolved_to']} not found in Neo4j, recreating")
        
        # Step 3: FAISS similarity search
        similar_entities = await self._find_similar_entities(name)
        
        if similar_entities:
            # Check if similarity exceeds threshold
            best_match = similar_entities[0]
            if best_match["score"] >= self.settings.similarity_threshold:
                logger.debug(f"Found similar entity: {best_match['entity_id']} (score: {best_match['score']:.3f})")
                
                # Link provisional entity
                await self._link_provisional(name, best_match["entity_id"])
                
                return best_match["entity_id"]
        
        # Step 4: Create new entity with locking to prevent race conditions
        # Try to acquire lock by inserting provisional entity first
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try to insert provisional entity (unique constraint on name)
                temp_id = f"entity_{uuid4().hex[:12]}"
                await self.provisional_entities.insert_one({
                    "name": name,
                    "entity_id": temp_id,
                    "resolution_status": "creating",
                    "created_at": time.time()
                })
                
                # We got the lock, create the entity
                new_entity_id = await self._create_entity(name, entity_type)
                logger.debug(f"Created new entity: {new_entity_id}")
                
                # Update provisional entity with final ID
                await self.provisional_entities.update_one(
                    {"name": name, "entity_id": temp_id},
                    {"$set": {"entity_id": new_entity_id, "resolution_status": "resolved", "resolved_to": new_entity_id}}
                )
                
                return new_entity_id
                
            except Exception as e:
                # Another task is creating this entity, wait and retry
                if "duplicate" in str(e).lower() or attempt < max_retries - 1:
                    logger.debug(f"Entity creation conflict for '{name}', retrying... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    
                    # Check if entity was created by other task
                    entity_id = self._check_neo4j_exact_match(name)
                    if entity_id:
                        logger.debug(f"Entity created by another task: {entity_id}")
                        return entity_id
                else:
                    logger.error(f"Failed to create entity '{name}' after {max_retries} attempts: {e}")
                    raise
        
        raise RuntimeError(f"Failed to resolve entity '{name}' after all retries")
    
    def _check_neo4j_exact_match(self, name: str) -> Optional[str]:
        """Check Neo4j for exact entity name match."""
        with self.neo4j.get_session() as session:
            # Check canonical name
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.canonical_name = $name
                RETURN e.entity_id AS entity_id
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            if record:
                return record["entity_id"]
            
            # Check aliases
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE $name IN e.aliases
                RETURN e.entity_id AS entity_id
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            if record:
                return record["entity_id"]
        
        return None
    
    async def _find_similar_entities(
        self,
        name: str,
    ) -> list[dict]:
        """
        Find similar entities using FAISS.
        
        Note: Entities should be embedded and indexed separately.
        For now, this is a simplified version.
        """
        # In production, maintain separate FAISS index for entity names
        # For simplicity, query Neo4j with fuzzy matching
        
        similar = []
        
        with self.neo4j.get_session() as session:
            # Get all entities (in production, use embedding similarity)
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.entity_id AS entity_id,
                       e.canonical_name AS canonical_name,
                       e.aliases AS aliases
                LIMIT 100
                """
            )
            
            for record in result:
                # Simple string similarity (in production, use embeddings)
                similarity = self._compute_string_similarity(
                    name, record["canonical_name"]
                )
                
                similar.append({
                    "entity_id": record["entity_id"],
                    "name": record["canonical_name"],
                    "score": similarity,
                })
        
        # Sort by score
        similar.sort(key=lambda x: x["score"], reverse=True)
        
        return similar[:self.settings.faiss_top_k]
    
    def _compute_string_similarity(self, str1: str, str2: str) -> float:
        """
        Compute string similarity (simplified).
        
        In production, use embedding cosine similarity.
        """
        # Normalize
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Jaccard similarity (token-based)
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _link_provisional(self, name: str, entity_id: str):
        """Link provisional entity to canonical entity."""
        await self.provisional_entities.update_one(
            {"name": name},
            {
                "$set": {
                    "resolved_to": entity_id,
                    "resolution_status": "resolved",
                }
            },
            upsert=True,
        )
    
    async def _create_entity(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> str:
        """Create new entity in Neo4j."""
        entity_id = f"entity_{uuid4().hex[:12]}"
        
        try:
            # Upsert in Neo4j
            logger.info(f"Creating entity in Neo4j: {entity_id} ({name})")
            result = self.neo4j.upsert_entity(
                entity_id=entity_id,
                canonical_name=name,
                entity_type=entity_type.value if entity_type else "Other",
                aliases=[],
                attributes={},
            )
            logger.info(f"Entity created successfully: {entity_id} - {result}")
            
            # Verify entity exists immediately (force read-your-writes consistency)
            with self.neo4j.get_session() as session:
                verify = session.run(
                    "MATCH (e:Entity {entity_id: $id}) RETURN e.canonical_name AS name",
                    id=entity_id
                ).single()
                
                if not verify:
                    logger.error(f"Entity {entity_id} not found immediately after creation!")
                    raise RuntimeError(f"Failed to verify entity {entity_id} creation")
                    
                logger.info(f"Entity verified in Neo4j: {entity_id} -> {verify['name']}")
            
        except Exception as e:
            logger.error(f"Failed to create entity {entity_id} in Neo4j: {e}", exc_info=True)
            raise
        
        try:
            # Create provisional entry
            provisional = ProvisionalEntity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                source_triple_ids=[],
                resolution_status="resolved",
                resolved_to=entity_id,
            )
            
            await self.provisional_entities.insert_one(provisional.model_dump())
            logger.debug(f"Provisional entity created: {entity_id}")
            
        except Exception as e:
            logger.error(f"Failed to create provisional entity {entity_id}: {e}", exc_info=True)
            # Don't raise - Neo4j entity already created
        
        return entity_id
