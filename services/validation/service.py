"""
Validation Engine - Validates candidate triples.

Validation methods:
1. Ontology rules - Schema conformance checks
2. Domain constraints - Type checking, range validation
3. External verification - API calls to authoritative sources
4. Confidence fusion - Combine multiple signals

Produces validated_triples collection with confidence scores.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

import httpx

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb
from shared.models.schemas import (
    CandidateTriple,
    TripleStatus,
    ValidationResult,
    ValidatedTriple,
)

logger = logging.getLogger(__name__)


class OntologyValidator:
    """Validates triples against ontology rules."""
    
    def __init__(self):
        # Define ontology rules (in production, load from OWL/RDF)
        self.rules = self._load_rules()
        
    def _load_rules(self) -> dict:
        """Load ontology rules."""
        return {
            "type_constraints": {
                "founded_by": {"subject": ["Organization"], "object": ["Person"]},
                "located_in": {"subject": ["Organization", "Person"], "object": ["Location"]},
                "ceo_of": {"subject": ["Person"], "object": ["Organization"]},
                "happens_on": {"subject": ["Event"], "object": ["Date"]},
            },
            "domain_rules": {
                "has_population": {"subject": ["Location"], "object_type": "numeric"},
                "established_year": {"subject": ["Organization"], "object_type": "year"},
            },
        }
    
    def validate(self, candidate: CandidateTriple) -> dict[str, bool]:
        """
        Validate triple against ontology rules.
        
        Returns:
            Dictionary of rule checks (rule_name -> passed)
        """
        checks = {}
        triple = candidate.triple
        predicate = triple.predicate.lower().replace(" ", "_")
        
        # Type constraint check
        if predicate in self.rules["type_constraints"]:
            constraint = self.rules["type_constraints"][predicate]
            
            # Check subject type
            if triple.subject_type and constraint.get("subject"):
                checks["subject_type_valid"] = (
                    triple.subject_type.value in constraint["subject"]
                )
            
            # Check object type
            if triple.object_type and constraint.get("object"):
                checks["object_type_valid"] = (
                    triple.object_type.value in constraint["object"]
                )
        
        # Domain rule check
        if predicate in self.rules["domain_rules"]:
            rule = self.rules["domain_rules"][predicate]
            
            if rule.get("object_type") == "numeric":
                try:
                    float(triple.object.replace(",", "").replace("$", ""))
                    checks["object_is_numeric"] = True
                except ValueError:
                    checks["object_is_numeric"] = False
            
            if rule.get("object_type") == "year":
                import re
                checks["object_is_year"] = bool(re.match(r"^\d{4}$", triple.object))
        
        # If no specific rules, pass by default
        if not checks:
            checks["no_conflicts"] = True
        
        return checks


class DomainConstraintValidator:
    """Validates domain-specific constraints."""
    
    def validate(self, candidate: CandidateTriple) -> dict[str, bool]:
        """
        Validate domain constraints.
        
        Returns:
            Dictionary of constraint checks
        """
        checks = {}
        triple = candidate.triple
        
        # Basic sanity checks
        checks["subject_not_empty"] = len(triple.subject.strip()) > 0
        checks["object_not_empty"] = len(triple.object.strip()) > 0
        checks["predicate_not_empty"] = len(triple.predicate.strip()) > 0
        
        # No self-loops
        checks["no_self_loop"] = triple.subject.lower() != triple.object.lower()
        
        # Reasonable lengths
        checks["subject_length_ok"] = 2 < len(triple.subject) < 200
        checks["object_length_ok"] = 2 < len(triple.object) < 200
        checks["predicate_length_ok"] = 2 < len(triple.predicate) < 100
        
        # Evidence quality
        checks["has_evidence"] = len(candidate.evidence) > 0
        
        return checks


class ExternalVerifier:
    """External verification via APIs (Wikipedia + Wikidata)."""
    
    def __init__(self):
        self.settings = get_settings().validation
        self.client = httpx.AsyncClient(
            timeout=self.settings.external_timeout,
            headers={
                "User-Agent": "GraphBuilder-RAG/1.0 (Educational; contact@example.com)"
            }
        )
        self.cache = {}  # Simple in-memory cache
        
    async def verify(self, candidate: CandidateTriple, graph_size: int = 0) -> dict[str, any]:
        """
        Verify triple using external sources.
        
        Bootstrap phase (< 1000 triples): MANDATORY verification
        Mature phase (>= 1000 triples): Only on conflicts
        
        Returns:
            Dictionary with verification results and confidence scores
        """
        triple = candidate.triple
        
        # Build query string for caching
        query_key = f"{triple.subject}|{triple.predicate}|{triple.object}"
        
        # Check cache first
        if query_key in self.cache:
            logger.debug(f"Cache hit for: {query_key}")
            return self.cache[query_key]
        
        verifications = {
            "wikipedia": {"found": False, "confidence": 0.0},
            "wikidata": {"found": False, "confidence": 0.0},
        }
        
        # Wikipedia verification
        try:
            wiki_result = await self._verify_wikipedia(triple)
            verifications["wikipedia"] = wiki_result
        except Exception as e:
            logger.warning(f"Wikipedia verification failed: {e}")
            verifications["wikipedia"] = {"found": False, "confidence": 0.0, "error": str(e)}
        
        # Wikidata verification (for structured data)
        try:
            wikidata_result = await self._verify_wikidata(triple)
            verifications["wikidata"] = wikidata_result
        except Exception as e:
            logger.warning(f"Wikidata verification failed: {e}")
            verifications["wikidata"] = {"found": False, "confidence": 0.0, "error": str(e)}
        
        # Cache result
        self.cache[query_key] = verifications
        
        return verifications
    
    async def _verify_wikipedia(self, triple) -> dict:
        """
        Verify against Wikipedia API.
        
        Returns dict with: found, confidence, snippet, url
        """
        # Build search query
        query = f"{triple.subject} {triple.predicate} {triple.object}"
        
        try:
            response = await self.client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": 3,  # Top 3 results
                }
            )
            
            if response.status_code != 200:
                return {"found": False, "confidence": 0.0, "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            
            if not search_results:
                return {
                    "found": False,
                    "confidence": 0.3,
                    "message": "No Wikipedia articles found"
                }
            
            # Found relevant article(s)
            top_result = search_results[0]
            snippet = top_result.get("snippet", "")
            
            # Calculate confidence based on snippet relevance
            subject_in_snippet = triple.subject.lower() in snippet.lower()
            object_in_snippet = triple.object.lower() in snippet.lower()
            
            if subject_in_snippet and object_in_snippet:
                confidence = 0.9
            elif subject_in_snippet or object_in_snippet:
                confidence = 0.7
            else:
                confidence = 0.5
            
            return {
                "found": True,
                "confidence": confidence,
                "title": top_result["title"],
                "snippet": snippet,
                "url": f"https://en.wikipedia.org/wiki/{top_result['title'].replace(' ', '_')}",
                "source": "wikipedia"
            }
            
        except httpx.TimeoutException:
            return {"found": False, "confidence": 0.0, "error": "Wikipedia API timeout"}
        except Exception as e:
            return {"found": False, "confidence": 0.0, "error": str(e)}
    
    async def _verify_wikidata(self, triple) -> dict:
        """
        Verify against Wikidata API.
        
        Returns dict with: found, confidence, entity_id, property_value
        """
        try:
            # Step 1: Search for subject entity
            entity_id = await self._find_wikidata_entity(triple.subject)
            
            if not entity_id:
                return {
                    "found": False,
                    "confidence": 0.0,
                    "message": "Entity not found in Wikidata"
                }
            
            # Step 2: Get entity data
            response = await self.client.get(
                f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
            )
            
            if response.status_code != 200:
                return {"found": False, "confidence": 0.0, "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            entity_data = data.get("entities", {}).get(entity_id, {})
            
            if not entity_data:
                return {"found": False, "confidence": 0.0, "message": "No entity data"}
            
            # Step 3: Check if any property matches our triple
            # This is simplified - in production, map predicates to Wikidata properties
            # Example: "born_in" -> P19, "works_at" -> P108
            
            claims = entity_data.get("claims", {})
            
            # Check for any matching values in claims
            found_match = False
            for property_id, property_claims in claims.items():
                for claim in property_claims:
                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                        value = claim["mainsnak"]["datavalue"].get("value")
                        if value and str(triple.object).lower() in str(value).lower():
                            found_match = True
                            break
                if found_match:
                    break
            
            if found_match:
                return {
                    "found": True,
                    "confidence": 0.95,
                    "entity_id": entity_id,
                    "message": "Confirmed by Wikidata",
                    "source": "wikidata"
                }
            else:
                return {
                    "found": True,
                    "confidence": 0.5,
                    "entity_id": entity_id,
                    "message": "Entity found but property not confirmed"
                }
                
        except httpx.TimeoutException:
            return {"found": False, "confidence": 0.0, "error": "Wikidata API timeout"}
        except Exception as e:
            return {"found": False, "confidence": 0.0, "error": str(e)}
    
    async def _find_wikidata_entity(self, entity_name: str) -> Optional[str]:
        """
        Search for entity in Wikidata and return entity ID (Q-number).
        """
        try:
            response = await self.client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": entity_name,
                    "language": "en",
                    "format": "json",
                    "limit": 1,
                }
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            results = data.get("search", [])
            
            if results:
                return results[0]["id"]  # Return Q-number (e.g., "Q937")
            
            return None
            
        except Exception as e:
            logger.warning(f"Wikidata entity search failed: {e}")
            return None
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class ValidationEngine:
    """Main validation engine coordinating all validators."""
    
    def __init__(self):
        self.settings = get_settings().validation
        self.mongodb = get_mongodb()
        self.candidate_triples = self.mongodb.get_async_collection("candidate_triples")
        self.validated_triples = self.mongodb.get_async_collection("validated_triples")
        
        self.ontology_validator = OntologyValidator()
        self.domain_validator = DomainConstraintValidator()
        self.external_verifier = ExternalVerifier()
    
    async def get_graph_size(self) -> int:
        """Get current knowledge graph size (number of validated triples)."""
        try:
            count = await self.validated_triples.count_documents(
                {"status": TripleStatus.VALIDATED.value}
            )
            return count
        except Exception as e:
            logger.error(f"Failed to get graph size: {e}")
            return 0
    
    async def validate_triple(
        self,
        candidate: CandidateTriple,
    ) -> ValidatedTriple:
        """
        Validate a candidate triple.
        
        Uses strict bootstrap validation for first 1000 triples,
        then switches to graph-based validation.
        
        Args:
            candidate: Candidate triple to validate
            
        Returns:
            Validated triple with confidence score
        """
        logger.debug(f"Validating triple: {candidate.triple_id}")
        
        # Get current graph size
        graph_size = await self.get_graph_size()
        
        # Determine validation mode
        is_bootstrap = graph_size < self.settings.bootstrap_threshold
        
        if is_bootstrap:
            logger.info(f"Bootstrap mode: graph_size={graph_size}, using strict validation")
            return await self._strict_bootstrap_validation(candidate, graph_size)
        else:
            logger.debug(f"Mature graph mode: graph_size={graph_size}, using standard validation")
            return await self._standard_validation(candidate, graph_size)
    
    async def _strict_bootstrap_validation(
        self,
        candidate: CandidateTriple,
        graph_size: int,
    ) -> ValidatedTriple:
        """
        STRICT validation for bootstrap phase.
        MANDATORY external verification to prevent hallucinations.
        """
        logger.info(f"Strict bootstrap validation for: {candidate.triple_id}")
        
        # Run all validators
        ontology_checks = self.ontology_validator.validate(candidate)
        domain_checks = self.domain_validator.validate(candidate)
        external_checks = await self.external_verifier.verify(candidate, graph_size)
        
        # Extract confidence scores
        llm_confidence = candidate.confidence
        wikipedia_confidence = external_checks.get("wikipedia", {}).get("confidence", 0.0)
        wikidata_confidence = external_checks.get("wikidata", {}).get("confidence", 0.0)
        
        # Detect conflicts between LLM and external sources
        has_conflict, conflict_details = self._detect_conflicts(
            candidate, external_checks
        )
        
        # STRICT BOOTSTRAP LOGIC
        if has_conflict:
            logger.warning(f"Conflict detected for {candidate.triple_id}: {conflict_details}")
            
            # Trust external sources over LLM during bootstrap
            if wikipedia_confidence > 0.8 or wikidata_confidence > 0.9:
                # External sources are confident - accept their version
                confidence_score = max(wikipedia_confidence, wikidata_confidence)
                status = TripleStatus.VALIDATED
                validation_message = "External sources override LLM (high confidence)"
            else:
                # No high-confidence external confirmation - reject
                confidence_score = (llm_confidence + wikipedia_confidence + wikidata_confidence) / 3
                status = TripleStatus.REJECTED
                validation_message = "Conflict detected, no high-confidence external source"
        else:
            # No conflict - aggregate all sources
            confidence_score = self._aggregate_confidence_bootstrap(
                llm_confidence, wikipedia_confidence, wikidata_confidence
            )
            
            # Require higher confidence during bootstrap
            if confidence_score >= self.settings.bootstrap_min_confidence:
                status = TripleStatus.VALIDATED
                validation_message = "All sources agree - bootstrap accepted"
            else:
                status = TripleStatus.REJECTED
                validation_message = f"Insufficient confidence: {confidence_score:.2f}"
        
        # Collect validation errors
        errors = []
        for check_name, passed in {**ontology_checks, **domain_checks}.items():
            if not passed:
                errors.append(f"Failed: {check_name}")
        
        if has_conflict:
            errors.append(f"Conflict: {conflict_details}")
        
        # Create validation result
        validation_result = ValidationResult(
            rule_checks={**ontology_checks, **domain_checks},
            external_verifications={
                "wikipedia": wikipedia_confidence,
                "wikidata": wikidata_confidence,
            },
            confidence_score=confidence_score,
            validation_errors=errors,
        )
        
        # Create validated triple
        validated = ValidatedTriple(
            triple_id=f"valid_{uuid4().hex[:12]}",
            candidate_triple_id=candidate.triple_id,
            triple=candidate.triple,
            evidence=candidate.evidence,
            validation=validation_result,
            status=status,
        )
        
        logger.info(
            f"Bootstrap validation result: {status.value}, "
            f"confidence={confidence_score:.2f}, "
            f"LLM={llm_confidence:.2f}, Wiki={wikipedia_confidence:.2f}, Wikidata={wikidata_confidence:.2f}"
        )
        
        return validated
    
    async def _standard_validation(
        self,
        candidate: CandidateTriple,
        graph_size: int,
    ) -> ValidatedTriple:
        """
        Standard validation for mature graph.
        Trusts internal graph, only uses external verification on conflicts.
        """
        # Run validation checks
        ontology_checks = self.ontology_validator.validate(candidate)
        domain_checks = self.domain_validator.validate(candidate)
        
        # Check for conflicts with existing graph
        graph_conflicts = await self._check_graph_conflicts(candidate)
        
        # If conflicts found, verify externally
        if graph_conflicts:
            logger.info(f"Graph conflict detected for {candidate.triple_id}, verifying externally")
            external_checks = await self.external_verifier.verify(candidate, graph_size)
        else:
            # No conflicts - skip external verification
            external_checks = {"wikipedia": {"confidence": 0.5}, "wikidata": {"confidence": 0.5}}
        
        # Compute confidence score
        confidence_score = self._compute_confidence(
            candidate, ontology_checks, domain_checks, external_checks
        )
        
        # Collect validation errors
        errors = []
        for check_name, passed in {**ontology_checks, **domain_checks}.items():
            if not passed:
                errors.append(f"Failed: {check_name}")
        
        if graph_conflicts:
            errors.append(f"Graph conflicts: {len(graph_conflicts)}")
        
        # Create validation result
        validation_result = ValidationResult(
            rule_checks={**ontology_checks, **domain_checks},
            external_verifications={
                "wikipedia": external_checks.get("wikipedia", {}).get("confidence", 0.0),
                "wikidata": external_checks.get("wikidata", {}).get("confidence", 0.0),
            },
            confidence_score=confidence_score,
            validation_errors=errors,
        )
        
        # Determine status
        status = (
            TripleStatus.VALIDATED
            if confidence_score >= self.settings.min_confidence
            else TripleStatus.REJECTED
        )
        
        # Create validated triple
        validated = ValidatedTriple(
            triple_id=f"valid_{uuid4().hex[:12]}",
            candidate_triple_id=candidate.triple_id,
            triple=candidate.triple,
            evidence=candidate.evidence,
            validation=validation_result,
            status=status,
        )
        
        return validated
    
    def _detect_conflicts(
        self,
        candidate: CandidateTriple,
        external_checks: dict,
    ) -> tuple[bool, str]:
        """
        Detect conflicts between LLM extraction and external sources.
        
        Returns:
            (has_conflict, conflict_description)
        """
        import re
        
        triple = candidate.triple
        
        # Extract dates/years from triple object
        llm_years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', triple.object)
        
        # Check Wikipedia snippet for conflicting dates
        wiki_snippet = external_checks.get("wikipedia", {}).get("snippet", "")
        if wiki_snippet and llm_years:
            wiki_years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', wiki_snippet)
            
            if wiki_years and llm_years:
                llm_year = llm_years[0]
                wiki_year = wiki_years[0]
                
                if llm_year != wiki_year:
                    return True, f"Date conflict: LLM says {llm_year}, Wikipedia says {wiki_year}"
        
        # Check if Wikipedia explicitly contradicts
        wiki_confidence = external_checks.get("wikipedia", {}).get("confidence", 0.0)
        wikidata_confidence = external_checks.get("wikidata", {}).get("confidence", 0.0)
        
        # Low external confidence while LLM is confident = possible hallucination
        if candidate.confidence > 0.8 and wiki_confidence < 0.5 and wikidata_confidence < 0.5:
            return True, "LLM confident but external sources don't confirm"
        
        return False, ""
    
    def _aggregate_confidence_bootstrap(
        self,
        llm_conf: float,
        wiki_conf: float,
        wikidata_conf: float,
    ) -> float:
        """
        Aggregate confidence scores during bootstrap.
        Give balanced weight, trusting high-quality LLM (Groq) more.
        """
        # Weights: balanced approach, trust Groq's strong performance
        w_llm = 0.5  # Groq is very accurate
        w_wiki = 0.3
        w_wikidata = 0.2
        
        aggregated = (w_llm * llm_conf) + (w_wiki * wiki_conf) + (w_wikidata * wikidata_conf)
        
        return min(1.0, max(0.0, aggregated))
    
    async def _check_graph_conflicts(self, candidate: CandidateTriple) -> list[dict]:
        """
        Check for conflicts with existing graph.
        
        Returns list of conflicting triples.
        """
        triple = candidate.triple
        
        # Find triples with same subject and similar predicate
        conflicts = []
        
        try:
            # Search for potential conflicts
            cursor = self.validated_triples.find({
                "triple.subject": triple.subject,
                "status": TripleStatus.VALIDATED.value,
            })
            
            async for doc in cursor:
                existing_triple = doc.get("triple", {})
                
                # Check if predicates conflict (e.g., "born_in" vs "died_in")
                # This is simplified - in production, use ontology rules
                if existing_triple.get("predicate") == triple.predicate:
                    if existing_triple.get("object") != triple.object:
                        conflicts.append({
                            "existing": existing_triple,
                            "reason": "Same subject+predicate, different object"
                        })
        
        except Exception as e:
            logger.error(f"Error checking graph conflicts: {e}")
        
        return conflicts
    
    def _compute_confidence(
        self,
        candidate: CandidateTriple,
        ontology_checks: dict,
        domain_checks: dict,
        external_checks: dict,
    ) -> float:
        """
        Compute final confidence score using fusion formula.
        
        Confidence = w1 * extraction_confidence +
                    w2 * rule_check_ratio +
                    w3 * external_verification_ratio
        """
        # Weights
        w1 = 0.4  # Extraction confidence
        w2 = 0.3  # Rule checks
        w3 = 0.3  # External verification
        
        # Extraction confidence
        extraction_conf = candidate.confidence
        
        # Rule check ratio (passed / total)
        all_checks = {**ontology_checks, **domain_checks}
        if all_checks:
            passed = sum(1 for v in all_checks.values() if v)
            rule_ratio = passed / len(all_checks)
        else:
            rule_ratio = 1.0  # No rules to check
        
        # External verification ratio
        if external_checks:
            verified = sum(1 for v in external_checks.values() if v is True)
            unknown = sum(1 for v in external_checks.values() if v is None)
            total = len(external_checks)
            
            # Treat unknown as neutral (0.5)
            external_ratio = (verified + 0.5 * unknown) / total
        else:
            external_ratio = 0.5  # No external verification available
        
        # Compute weighted score
        confidence = (
            w1 * extraction_conf +
            w2 * rule_ratio +
            w3 * external_ratio
        )
        
        return min(1.0, max(0.0, confidence))
    
    async def validate_document_triples(
        self,
        document_id: str,
    ) -> list[ValidatedTriple]:
        """
        Validate all candidate triples from a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of validated triples
        """
        logger.info(f"Validating triples for document: {document_id}")
        
        # Find all candidate triples for document
        candidates_cursor = self.candidate_triples.find({
            "evidence.document_id": document_id
        })
        
        candidates = []
        async for doc in candidates_cursor:
            candidates.append(CandidateTriple(**doc))
        
        logger.info(f"Found {len(candidates)} candidate triples")
        
        # Validate in parallel batches
        validated_triples = []
        batch_size = self.settings.parallel_checks
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            tasks = [self.validate_triple(c) for c in batch]
            results = await asyncio.gather(*tasks)
            validated_triples.extend(results)
        
        # Filter by confidence threshold
        accepted = [
            v for v in validated_triples
            if v.status == TripleStatus.VALIDATED
        ]
        
        # Insert into MongoDB
        if validated_triples:
            await self.validated_triples.insert_many(
                [v.model_dump() for v in validated_triples]
            )
        
        logger.info(
            f"Validation complete: {len(accepted)} accepted, "
            f"{len(validated_triples) - len(accepted)} rejected"
        )
        
        # Emit fusion task
        if accepted:
            await self._emit_fusion_task(document_id)
        
        return validated_triples
    
    async def _emit_fusion_task(self, document_id: str):
        """Emit fusion task."""
        try:
            from workers.tasks import fuse_triples
            
            fuse_triples.delay(document_id)
            logger.info(f"Fusion task emitted for: {document_id}")
        except Exception as e:
            logger.error(f"Failed to emit fusion task: {e}")
    
    async def close(self):
        """Close resources."""
        await self.external_verifier.close()
