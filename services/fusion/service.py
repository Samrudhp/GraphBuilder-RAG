"""
Fusion Service - Fuses validated triples into Neo4j knowledge graph.

Handles:
- Idempotent upsert using audit logs
- Entity resolution and linking
- Relationship versioning
- Conflict detection
- Provenance tracking
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.database.neo4j import get_neo4j
from shared.models.schemas import (
    GraphEdge,
    TripleStatus,
    UpsertAudit,
    ValidatedTriple,
)

logger = logging.getLogger(__name__)


class FusionService:
    """Service for fusing validated triples into Neo4j."""
    
    def __init__(self):
        self.settings = get_settings().fusion
        self.mongodb = get_mongodb()
        self.neo4j = get_neo4j()
        
        self.validated_triples = self.mongodb.get_async_collection("validated_triples")
        self.upsert_audit = self.mongodb.get_async_collection("upsert_audit")
        self.conflict_records = self.mongodb.get_async_collection("conflict_records")
        
    async def fuse_triple(
        self,
        validated: ValidatedTriple,
    ) -> GraphEdge:
        """
        Fuse a single validated triple into Neo4j.
        
        Performs:
        1. Entity resolution (map to canonical entity IDs)
        2. Check for existing edges
        3. Detect conflicts
        4. Upsert with versioning
        5. Audit logging
        
        Args:
            validated: Validated triple
            
        Returns:
            GraphEdge record
        """
        logger.debug(f"Fusing triple: {validated.triple_id}")
        
        triple = validated.triple
        
        # Step 1: Resolve entities
        from services.entity_resolution.service import EntityResolutionService
        resolver = EntityResolutionService()
        
        logger.debug(f"Resolving source entity: {triple.subject} ({triple.subject_type})")
        source_entity_id = await resolver.resolve_entity(
            name=triple.subject,
            entity_type=triple.subject_type,
        )
        logger.info(f"Source entity resolved: {triple.subject} -> {source_entity_id}")
        
        logger.debug(f"Resolving target entity: {triple.object} ({triple.object_type})")
        target_entity_id = await resolver.resolve_entity(
            name=triple.object,
            entity_type=triple.object_type,
        )
        logger.info(f"Target entity resolved: {triple.object} -> {target_entity_id}")
        
        # Step 2: Check for existing edge
        existing_edges = self.neo4j.find_conflicting_edges(
            source_id=source_entity_id,
            relationship_type=self._normalize_relationship(triple.predicate),
            threshold=self.settings.conflict_threshold,
        )
        
        # Step 3: Detect conflicts
        conflict_detected = False
        if existing_edges:
            for edge in existing_edges:
                if edge["target_id"] != target_entity_id:
                    # Conflict: same source + relationship, different target
                    conflict_detected = True
                    await self._log_conflict(
                        validated, edge, source_entity_id, target_entity_id
                    )
        
        # Step 4: Determine edge ID and version
        edge_id = f"edge_{uuid4().hex[:12]}"
        version = 1
        
        if existing_edges:
            # Check if updating existing edge
            matching = [e for e in existing_edges if e["target_id"] == target_entity_id]
            if matching:
                edge_id = matching[0]["edge_id"]
                version = matching[0]["version"] + 1
        
        # Step 5: Upsert into Neo4j
        evidence_ids = [ev.document_id for ev in validated.evidence]
        
        # Verify entities exist before creating relationship
        logger.debug(f"Verifying entities exist: source={source_entity_id}, target={target_entity_id}")
        with self.neo4j.get_session() as session:
            source_exists = session.run(
                "MATCH (e:Entity {entity_id: $id}) RETURN count(e) AS count",
                id=source_entity_id
            ).single()["count"] > 0
            
            target_exists = session.run(
                "MATCH (e:Entity {entity_id: $id}) RETURN count(e) AS count",
                id=target_entity_id
            ).single()["count"] > 0
            
            if not source_exists or not target_exists:
                logger.error(
                    f"Entities missing before upsert! "
                    f"Source {source_entity_id} exists: {source_exists}, "
                    f"Target {target_entity_id} exists: {target_exists}"
                )
                raise ValueError(
                    f"Entities must exist. Source exists: {source_exists}, Target exists: {target_exists}"
                )
        
        logger.debug(f"Both entities verified to exist")
        
        relationship_props = self.neo4j.upsert_relationship(
            edge_id=edge_id,
            source_id=source_entity_id,
            target_id=target_entity_id,
            relationship_type=self._normalize_relationship(triple.predicate),
            properties={
                "validated_triple_id": validated.triple_id,
                "confidence": validated.validation.confidence_score,
            },
            confidence=validated.validation.confidence_score,
            evidence_ids=evidence_ids,
            version=version,
        )
        
        # Step 6: Create graph edge record
        graph_edge = GraphEdge(
            edge_id=edge_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=self._normalize_relationship(triple.predicate),
            properties=relationship_props,
            confidence=validated.validation.confidence_score,
            evidence_ids=evidence_ids,
            version=version,
        )
        
        # Step 7: Audit log
        await self._log_upsert(
            edge_id=edge_id,
            operation="insert" if version == 1 else "update",
            previous_version=version - 1 if version > 1 else None,
            new_version=version,
            conflict_detected=conflict_detected,
        )
        
        # Step 8: Update triple status
        await self.validated_triples.update_one(
            {"triple_id": validated.triple_id},
            {"$set": {"status": TripleStatus.FUSED.value}}
        )
        
        logger.info(
            f"Triple fused: {edge_id} v{version} "
            f"(conflict: {conflict_detected})"
        )
        
        return graph_edge
    
    async def fuse_document_triples(
        self,
        document_id: str,
    ) -> list[GraphEdge]:
        """
        Fuse all validated triples from a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of graph edges created/updated
        """
        logger.info(f"Fusing triples for document: {document_id}")
        
        # Find validated triples
        cursor = self.validated_triples.find({
            "evidence.document_id": document_id,
            "status": TripleStatus.VALIDATED.value,
        })
        
        validated_list = []
        async for doc in cursor:
            validated_list.append(ValidatedTriple(**doc))
        
        logger.info(f"Found {len(validated_list)} validated triples to fuse")
        
        # Fuse in batches
        graph_edges = []
        batch_size = self.settings.batch_size
        
        for i in range(0, len(validated_list), batch_size):
            batch = validated_list[i:i + batch_size]
            
            for validated in batch:
                try:
                    edge = await self.fuse_triple(validated)
                    graph_edges.append(edge)
                except Exception as e:
                    logger.error(f"Failed to fuse triple {validated.triple_id}: {e}")
        
        logger.info(f"Fused {len(graph_edges)} triples into knowledge graph")
        return graph_edges
    
    def _normalize_relationship(self, predicate: str) -> str:
        """Normalize relationship type name."""
        # Convert to snake_case, uppercase
        normalized = predicate.strip().replace(" ", "_").upper()
        return normalized
    
    async def _log_conflict(
        self,
        validated: ValidatedTriple,
        existing_edge: dict,
        source_id: str,
        target_id: str,
    ):
        """Log detected conflict."""
        from shared.models.schemas import ConflictRecord
        
        conflict = ConflictRecord(
            conflict_id=f"conflict_{uuid4().hex[:12]}",
            edge_ids=[existing_edge["edge_id"], validated.triple_id],
            conflict_type="contradiction",
            description=(
                f"Same source ({source_id}) and relationship, "
                f"but different targets: {existing_edge['target_name']} vs {validated.triple.object}"
            ),
            severity=0.8,
            resolution_status="pending",
        )
        
        await self.conflict_records.insert_one(conflict.model_dump())
        logger.warning(f"Conflict logged: {conflict.conflict_id}")
    
    async def _log_upsert(
        self,
        edge_id: str,
        operation: str,
        previous_version: Optional[int],
        new_version: int,
        conflict_detected: bool,
    ):
        """Log upsert operation."""
        audit = UpsertAudit(
            audit_id=f"audit_{uuid4().hex[:12]}",
            edge_id=edge_id,
            operation=operation,
            previous_version=previous_version,
            new_version=new_version,
            changes={},
            conflict_detected=conflict_detected,
        )
        
        await self.upsert_audit.insert_one(audit.model_dump())
