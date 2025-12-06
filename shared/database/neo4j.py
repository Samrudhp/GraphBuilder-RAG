"""Neo4j database connector and utilities."""
import logging
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Neo4j connection manager with versioning support."""
    
    def __init__(self):
        self.settings = get_settings().neo4j
        self._driver: Optional[Driver] = None
        
    @property
    def driver(self) -> Driver:
        """Get Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password),
                max_connection_lifetime=self.settings.max_connection_lifetime,
                max_connection_pool_size=self.settings.max_connection_pool_size,
            )
            logger.info("Neo4j driver connected")
        return self._driver
    
    def get_session(self, database: Optional[str] = None) -> Session:
        """Get Neo4j session."""
        db = database or self.settings.database
        return self.driver.session(database=db)
    
    def ping(self) -> bool:
        """Test database connection."""
        try:
            with self.get_session() as session:
                result = session.run("RETURN 1 AS ping")
                return result.single()["ping"] == 1
        except ServiceUnavailable:
            logger.error("Neo4j connection failed")
            return False
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for the knowledge graph."""
        with self.get_session() as session:
            # Entity constraints
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT entity_canonical_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS NOT NULL",
            ]
            
            # Indexes for entities
            indexes = [
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                "CREATE INDEX entity_created IF NOT EXISTS FOR (e:Entity) ON (e.created_at)",
                "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.canonical_name, e.aliases]",
            ]
            
            # Relationship indexes
            rel_indexes = [
                "CREATE INDEX rel_confidence IF NOT EXISTS FOR ()-[r:RELATED]->() ON (r.confidence)",
                "CREATE INDEX rel_version IF NOT EXISTS FOR ()-[r:RELATED]->() ON (r.version)",
                "CREATE INDEX rel_created IF NOT EXISTS FOR ()-[r:RELATED]->() ON (r.created_at)",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"Constraint creation skipped: {e}")
            
            for index in indexes + rel_indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation skipped: {e}")
    
    def upsert_entity(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: str,
        aliases: list[str] = None,
        attributes: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Upsert an entity node.
        
        Args:
            entity_id: Unique entity identifier
            canonical_name: Primary name
            entity_type: Type of entity
            aliases: Alternative names
            attributes: Additional properties
            
        Returns:
            Entity node properties
        """
        with self.get_session() as session:
            query = """
            MERGE (e:Entity {entity_id: $entity_id})
            ON CREATE SET 
                e.canonical_name = $canonical_name,
                e.entity_type = $entity_type,
                e.aliases = $aliases,
                e.created_at = datetime(),
                e.updated_at = datetime()
            ON MATCH SET
                e.canonical_name = $canonical_name,
                e.entity_type = $entity_type,
                e.aliases = $aliases,
                e.updated_at = datetime()
            SET e += $attributes
            RETURN e
            """
            result = session.run(
                query,
                entity_id=entity_id,
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=aliases or [],
                attributes=attributes or {},
            )
            record = result.single()
            return dict(record["e"]) if record else {}
    
    def upsert_relationship(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: dict[str, Any] = None,
        confidence: float = 1.0,
        evidence_ids: list[str] = None,
        version: int = 1,
    ) -> dict[str, Any]:
        """
        Upsert a relationship with versioning.
        
        Args:
            edge_id: Unique edge identifier
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship
            properties: Additional properties
            confidence: Confidence score
            evidence_ids: Supporting evidence
            version: Edge version
            
        Returns:
            Relationship properties
        """
        with self.get_session() as session:
            # Ensure entities exist - but they must already have canonical_name from upsert_entity
            # Just verify they exist, don't create empty entities
            source_check = session.run(
                "MATCH (e:Entity {entity_id: $entity_id}) RETURN e.canonical_name AS name",
                entity_id=source_id
            ).single()
            
            target_check = session.run(
                "MATCH (e:Entity {entity_id: $entity_id}) RETURN e.canonical_name AS name",
                entity_id=target_id
            ).single()
            
            if not source_check or not target_check:
                raise ValueError(
                    f"Entities must exist before creating relationship. "
                    f"Source {source_id} exists: {bool(source_check)}, "
                    f"Target {target_id} exists: {bool(target_check)}"
                )
            
            # Upsert relationship
            query = f"""
            MATCH (source:Entity {{entity_id: $source_id}})
            MATCH (target:Entity {{entity_id: $target_id}})
            MERGE (source)-[r:{relationship_type} {{edge_id: $edge_id}}]->(target)
            ON CREATE SET
                r.confidence = $confidence,
                r.evidence_ids = $evidence_ids,
                r.version = $version,
                r.created_at = datetime(),
                r.updated_at = datetime()
            ON MATCH SET
                r.confidence = $confidence,
                r.evidence_ids = $evidence_ids,
                r.version = $version,
                r.updated_at = datetime()
            SET r += $properties
            RETURN r
            """
            result = session.run(
                query,
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                confidence=confidence,
                evidence_ids=evidence_ids or [],
                version=version,
                properties=properties or {},
            )
            record = result.single()
            return dict(record["r"]) if record else {}
    
    def get_subgraph(
        self,
        entity_ids: list[str],
        depth: int = 2,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """
        Extract subgraph around given entities.
        
        Args:
            entity_ids: Starting entity IDs
            depth: Traversal depth
            min_confidence: Minimum edge confidence
            
        Returns:
            Subgraph with nodes and edges
        """
        with self.get_session() as session:
            # Depth must be hardcoded in the query, not a parameter
            query = f"""
            MATCH path = (start:Entity)-[r*1..{depth}]-(end:Entity)
            WHERE start.entity_id IN $entity_ids
                AND ALL(rel IN relationships(path) WHERE rel.confidence >= $min_confidence)
            WITH nodes(path) AS path_nodes, relationships(path) AS path_rels
            UNWIND path_nodes AS node
            WITH COLLECT(DISTINCT node) AS all_nodes, path_rels
            UNWIND path_rels AS rel_list
            UNWIND rel_list AS rel
            RETURN 
                all_nodes AS nodes,
                COLLECT(DISTINCT rel) AS relationships
            """
            result = session.run(
                query,
                entity_ids=entity_ids,
                min_confidence=min_confidence,
            )
            record = result.single()
            if record:
                nodes = [dict(n) for n in record["nodes"]] if record["nodes"] else []
                relationships = [dict(r) for r in record["relationships"]] if record["relationships"] else []
                return {
                    "nodes": nodes,
                    "relationships": relationships,
                }
            return {"nodes": [], "relationships": []}
    
    def find_conflicting_edges(
        self,
        source_id: str,
        relationship_type: str,
        threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """
        Find potentially conflicting edges from the same source.
        
        Args:
            source_id: Source entity ID
            relationship_type: Relationship type to check
            threshold: Confidence threshold for conflicts
            
        Returns:
            List of conflicting edge groups
        """
        with self.get_session() as session:
            query = f"""
            MATCH (source:Entity {{entity_id: $source_id}})-[r:{relationship_type}]->(target:Entity)
            WHERE r.confidence >= $threshold
            RETURN 
                r.edge_id AS edge_id,
                target.entity_id AS target_id,
                target.canonical_name AS target_name,
                r.confidence AS confidence,
                r.version AS version
            ORDER BY r.confidence DESC
            """
            result = session.run(
                query,
                source_id=source_id,
                threshold=threshold,
            )
            return [dict(record) for record in result]
    
    def find_all_conflicts(self, min_confidence: float = 0.7) -> list[dict[str, Any]]:
        """
        Find all conflicting edges across the entire graph.
        Conflicts occur when same entity has multiple different targets for same relationship.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of conflict dictionaries with entity_id, relationship_type, and edges
        """
        with self.get_session() as session:
            query = """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE r.confidence >= $min_confidence
            AND r.deprecated IS NULL
            WITH source, type(r) as rel_type, collect({
                edge_id: r.edge_id,
                target_id: target.entity_id,
                target_name: target.canonical_name,
                confidence: r.confidence,
                evidence: r.evidence,
                created_at: r.created_at
            }) as edges
            WHERE size(edges) > 1
            RETURN 
                source.entity_id as entity_id,
                source.canonical_name as entity_name,
                rel_type as relationship_type,
                edges
            """
            result = session.run(query, min_confidence=min_confidence)
            return [dict(record) for record in result]
    
    def close(self):
        """Close Neo4j driver."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")


# Global connector instance
_neo4j_connector: Optional[Neo4jConnector] = None


def get_neo4j() -> Neo4jConnector:
    """Get global Neo4j connector instance."""
    global _neo4j_connector
    if _neo4j_connector is None:
        _neo4j_connector = Neo4jConnector()
    return _neo4j_connector
