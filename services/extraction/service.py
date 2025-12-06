"""
Extraction Service - Extract knowledge triples from normalized documents.

Two extraction methods:
1. TableExtractor - Deterministic extraction from structured tables
2. LLMExtractor - DeepSeek-based extraction from unstructured text

Handles:
- Batch processing
- Evidence span tracking
- Deduplication with evidence union
- Confidence scoring
"""
import logging
import re
from datetime import datetime
from typing import Optional
from uuid import uuid4

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.models.schemas import (
    CandidateTriple,
    EvidenceSpan,
    Triple,
    EntityType,
)
from shared.prompts.templates import (
    EXTRACTION_SYSTEM_PROMPT,
    format_extraction_prompt,
)
from shared.utils.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)


class TableExtractor:
    """Deterministic triple extraction from tables."""
    
    def extract_from_table(
        self,
        table: dict,
        document_id: str,
    ) -> list[CandidateTriple]:
        """
        Extract triples from a table using rule-based logic.
        
        Strategies:
        - First column as subject, headers as predicates
        - Row-wise relationships
        - Infer entity types from column names
        
        Args:
            table: Table data
            document_id: Source document ID
            
        Returns:
            List of candidate triples
        """
        triples = []
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        table_id = table.get("table_id", "unknown")
        
        if not headers or not rows:
            return triples
        
        # Strategy 1: First column is subject, others are predicates
        if len(headers) >= 2:
            subject_col = headers[0]
            
            for row_idx, row in enumerate(rows):
                if len(row) < 2:
                    continue
                
                subject = str(row[0]).strip()
                if not subject or subject == "":
                    continue
                
                for col_idx in range(1, len(headers)):
                    if col_idx >= len(row):
                        continue
                    
                    obj = str(row[col_idx]).strip()
                    if not obj or obj == "":
                        continue
                    
                    predicate = headers[col_idx].strip()
                    
                    # Infer entity types
                    subject_type = self._infer_entity_type(subject, subject_col)
                    object_type = self._infer_entity_type(obj, predicate)
                    
                    triple = Triple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        subject_type=subject_type,
                        object_type=object_type,
                    )
                    
                    evidence = EvidenceSpan(
                        document_id=document_id,
                        start_char=0,
                        end_char=0,
                        text=f"Table: {subject} | {predicate} | {obj}",
                        table_id=table_id,
                    )
                    
                    candidate = CandidateTriple(
                        triple_id=f"triple_{uuid4().hex[:12]}",
                        triple=triple,
                        evidence=[evidence],
                        confidence=0.95,  # High confidence for table data
                        extraction_method="table",
                    )
                    
                    triples.append(candidate)
        
        logger.debug(f"Extracted {len(triples)} triples from table {table_id}")
        return triples
    
    def _infer_entity_type(self, value: str, context: str) -> Optional[EntityType]:
        """Infer entity type from value and context."""
        value_lower = value.lower()
        context_lower = context.lower()
        
        # Date patterns
        if re.search(r"\d{4}-\d{2}-\d{2}", value) or re.search(r"\d{1,2}/\d{1,2}/\d{4}", value):
            return EntityType.DATE
        
        # Context-based inference
        if any(word in context_lower for word in ["name", "person", "author", "founder"]):
            return EntityType.PERSON
        
        if any(word in context_lower for word in ["company", "organization", "org", "institution"]):
            return EntityType.ORGANIZATION
        
        if any(word in context_lower for word in ["city", "country", "location", "place", "address"]):
            return EntityType.LOCATION
        
        if any(word in context_lower for word in ["product", "model", "item"]):
            return EntityType.PRODUCT
        
        if any(word in context_lower for word in ["event", "conference", "meeting"]):
            return EntityType.EVENT
        
        # Value-based inference
        if value.replace(".", "").replace(",", "").replace("$", "").replace("%", "").replace(" ", "").isdigit():
            return EntityType.CONCEPT  # Numeric concept
        
        return EntityType.OTHER


class LLMExtractor:
    """LLM-based triple extraction using DeepSeek."""
    
    def __init__(self):
        self.settings = get_settings().extraction
        self.ollama = get_ollama_client()
        
    async def extract_from_text(
        self,
        text: str,
        document_id: str,
        section_id: Optional[str] = None,
        domain: str = "general",
        section_heading: str = "main",
        start_char: int = 0,
    ) -> list[CandidateTriple]:
        """
        Extract triples from text using LLM.
        
        Args:
            text: Text to extract from
            document_id: Source document ID
            section_id: Section ID if applicable
            domain: Domain context
            section_heading: Section heading
            start_char: Start character offset
            
        Returns:
            List of candidate triples
        """
        if len(text) < 20:
            logger.debug("Text too short for extraction")
            return []
        
        # Truncate if too long
        max_chars = 4000  # Leave room for prompt
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        # Format prompt
        user_prompt = format_extraction_prompt(
            text=text,
            document_id=document_id,
            domain=domain,
            section_heading=section_heading,
        )
        
        # Call LLM
        try:
            response = await self.ollama.generate_extraction(
                text=text,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            
            # Parse triples
            triples_data = response.get("triples", [])
            candidates = []
            
            for triple_data in triples_data:
                # Validate triple
                if not all(k in triple_data for k in ["subject", "predicate", "object"]):
                    logger.warning(f"Invalid triple: {triple_data}")
                    continue
                
                confidence = triple_data.get("confidence", 0.5)
                if confidence < self.settings.min_confidence:
                    logger.debug(f"Skipping low confidence triple: {confidence}")
                    continue
                
                # Map entity types
                subject_type = None
                object_type = None
                
                if "subject_type" in triple_data:
                    try:
                        subject_type = EntityType(triple_data["subject_type"])
                    except ValueError:
                        pass
                
                if "object_type" in triple_data:
                    try:
                        object_type = EntityType(triple_data["object_type"])
                    except ValueError:
                        pass
                
                triple = Triple(
                    subject=triple_data["subject"],
                    predicate=triple_data["predicate"],
                    object=triple_data["object"],
                    subject_type=subject_type,
                    object_type=object_type,
                )
                
                # Create evidence span
                evidence = EvidenceSpan(
                    document_id=document_id,
                    start_char=start_char,
                    end_char=start_char + len(text),
                    text=text[:500],  # Store snippet
                    section_id=section_id,
                )
                
                candidate = CandidateTriple(
                    triple_id=f"triple_{uuid4().hex[:12]}",
                    triple=triple,
                    evidence=[evidence],
                    confidence=confidence,
                    extraction_method="llm",
                )
                
                candidates.append(candidate)
            
            logger.debug(f"Extracted {len(candidates)} triples from text")
            return candidates
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []


class ExtractionService:
    """Main extraction service coordinating both methods."""
    
    def __init__(self):
        self.mongodb = get_mongodb()
        self.normalized_docs = self.mongodb.get_async_collection("normalized_docs")
        self.candidate_triples = self.mongodb.get_async_collection("candidate_triples")
        
        self.table_extractor = TableExtractor()
        self.llm_extractor = LLMExtractor()
        
    async def extract_from_document(
        self,
        document_id: str,
    ) -> list[CandidateTriple]:
        """
        Extract triples from a normalized document.
        
        Args:
            document_id: Normalized document ID
            
        Returns:
            List of candidate triples
        """
        logger.info(f"Extracting triples from document: {document_id}")
        
        # Get document
        doc = await self.normalized_docs.find_one({"document_id": document_id})
        if not doc:
            raise ValueError(f"Document not found: {document_id}")
        
        all_candidates = []
        
        # Extract from tables (deterministic)
        tables = doc.get("tables", [])
        for table in tables:
            table_triples = self.table_extractor.extract_from_table(
                table, document_id
            )
            all_candidates.extend(table_triples)
        
        # Extract from sections (LLM)
        sections = doc.get("sections", [])
        domain = doc.get("metadata", {}).get("domain", "general")
        
        for section in sections:
            section_text = section.get("text", "")
            if not section_text:
                continue
            
            section_triples = await self.llm_extractor.extract_from_text(
                text=section_text,
                document_id=document_id,
                section_id=section.get("section_id"),
                domain=domain,
                section_heading=section.get("heading", ""),
                start_char=section.get("start_char", 0),
            )
            all_candidates.extend(section_triples)
        
        # Deduplicate and merge evidence
        deduplicated = self._deduplicate_triples(all_candidates)
        
        # Insert into MongoDB
        if deduplicated:
            await self.candidate_triples.insert_many(
                [t.model_dump() for t in deduplicated]
            )
        
        logger.info(
            f"Extracted {len(deduplicated)} unique triples "
            f"(from {len(all_candidates)} candidates)"
        )
        
        # Emit validation task
        await self._emit_validation_task(document_id)
        
        return deduplicated
    
    def _deduplicate_triples(
        self,
        candidates: list[CandidateTriple],
    ) -> list[CandidateTriple]:
        """
        Deduplicate triples and merge evidence.
        
        Triples with identical (subject, predicate, object) are merged,
        combining their evidence and taking the max confidence.
        """
        triple_map = {}
        
        for candidate in candidates:
            key = (
                candidate.triple.subject.lower(),
                candidate.triple.predicate.lower(),
                candidate.triple.object.lower(),
            )
            
            if key in triple_map:
                # Merge evidence
                existing = triple_map[key]
                existing.evidence.extend(candidate.evidence)
                # Take max confidence
                existing.confidence = max(existing.confidence, candidate.confidence)
                # Prefer LLM method if available
                if candidate.extraction_method == "llm":
                    existing.extraction_method = "llm"
            else:
                triple_map[key] = candidate
        
        return list(triple_map.values())
    
    async def _emit_validation_task(self, document_id: str):
        """Emit validation task for extracted triples."""
        try:
            from workers.tasks import validate_triples
            
            validate_triples.delay(document_id)
            logger.info(f"Validation task emitted for: {document_id}")
        except Exception as e:
            logger.error(f"Failed to emit validation task: {e}")
