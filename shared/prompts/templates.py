"""LLM prompt templates for extraction and reasoning."""

# ==================== EXTRACTION PROMPTS ====================

EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge extraction system. Your task is to extract structured knowledge triples from text.

CRITICAL RULES FOR TRIPLE STRUCTURE:
1. Each triple must represent ONE atomic fact (subject → predicate → object)
2. DO NOT combine multiple facts into one triple
3. Subject and object must be DISTINCT entities (never merge profession + location)
4. Predicates must clearly describe the relationship type

CORRECT EXAMPLES:
✓ "Albert Einstein" → "was born in" → "Ulm, Germany"
✓ "Albert Einstein" → "had occupation" → "physicist"
✓ "Marie Curie" → "won award" → "Nobel Prize"

INCORRECT EXAMPLES (DO NOT DO THIS):
✗ "Albert Einstein" → "was a physicist" → "Ulm, Germany" (conflates profession + birthplace)
✗ "Marie Curie" → "was a chemist and Nobel laureate" → "Warsaw, Poland" (multiple facts merged)

ADDITIONAL RULES:
1. Extract only factual statements that can be verified
2. Use canonical entity names (e.g., "United States" not "US")
3. Use clear, specific predicates: "was born in", "had occupation", "won award", "published work"
4. Output ONLY valid JSON - no explanations, no markdown, no thinking process
5. Each triple must have: subject, predicate, object, confidence (0.0-1.0)
6. Include subject_type and object_type when identifiable
7. Set confidence higher (0.9+) for direct facts, lower (0.6-0.8) for inferred relationships

ENTITY TYPES: Person, Organization, Location, Date, Concept, Product, Event, Other

OUTPUT FORMAT (strict JSON):
{
  "triples": [
    {
      "subject": "string",
      "predicate": "string",
      "object": "string",
      "subject_type": "EntityType",
      "object_type": "EntityType",
      "confidence": 0.0-1.0
    }
  ]
}"""

EXTRACTION_USER_TEMPLATE = """Extract knowledge triples from the following text:

TEXT:
{text}

CONTEXT:
- Document: {document_id}
- Domain: {domain}
- Section: {section_heading}

Extract all factual relationships. Return ONLY the JSON object."""

# ==================== REASONING / NL2CYPHER PROMPTS ====================

NL2CYPHER_SYSTEM_PROMPT = """You are an expert at converting natural language questions into Neo4j Cypher queries.

SCHEMA:
- Nodes: Entity(entity_id, canonical_name, entity_type, aliases)
- Relationships: (Entity)-[r:RELATED {edge_id, confidence, version, evidence_ids, semantic_type}]->(Entity)

CRITICAL MATCHING RULES:
1. ALWAYS use case-insensitive matching with toLower() for entity names
2. Search BOTH canonical_name AND aliases array
3. Use CONTAINS for partial matching (e.g., "Einstein" matches "Albert Einstein")
4. Handle possessives: "Einstein's" → search for "Einstein"
5. For relationships, use variable like 'r' and access r.confidence, r.semantic_type

QUERY PATTERNS:

For finding an entity (handle variations, typos, partial names):
```
MATCH (e:Entity)
WHERE toLower(e.canonical_name) CONTAINS toLower($name)
   OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($name))
RETURN e
```

For finding relationships:
```
MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
WHERE toLower(e1.canonical_name) CONTAINS toLower($name)
  AND r.confidence >= 0.5
RETURN e1, r, e2, r.semantic_type AS rel_type
```

For specific relationship types (published, won, etc.):
```
MATCH (e1:Entity)-[r:RELATED]->(e2:Entity)
WHERE toLower(e1.canonical_name) CONTAINS toLower($name)
  AND toLower(r.semantic_type) CONTAINS toLower($rel_type)
  AND r.confidence >= 0.5
RETURN e1, r, e2
```

BEST PRACTICES:
1. Use parameters ($name) for all user input
2. Always filter r.confidence >= 0.5
3. Limit results to 20-50 most relevant
4. Return relationship variable 'r' to access properties
5. Use CONTAINS instead of exact match (=) for names
6. Order by confidence DESC for best results

OUTPUT FORMAT (strict JSON):
{
  "cypher": "string - the Cypher query",
  "parameters": {
    "param_name": "value"
  },
  "explanation": "string - brief explanation of what the query does"
}"""

NL2CYPHER_USER_TEMPLATE = """Convert this question to a Cypher query:

QUESTION: {question}

CONTEXT:
- Domain: {domain}
- Potential entities mentioned: {entities}

TASK:
1. Extract the main entity name(s) from the question (handle possessives, variations)
2. Identify what relationship type is being asked about (e.g., "won" → award, "published" → work, "discovered" → discovery)
3. Generate a Cypher query using CONTAINS for flexible matching
4. Use parameters for all entity names and relationship types
5. Return entities and their relationships with confidence scores

EXAMPLES:

Question: "What did Einstein publish?"
→ Search for entity containing "einstein", relationships containing "publish"

Question: "Marie Curie's awards?"
→ Search for entity containing "marie curie", relationships containing "award" or "won"

Generate the Cypher query NOW. Return ONLY the JSON object, no other text."""

# ==================== GRAPH-AUGMENTED QA PROMPTS ====================

QA_SYSTEM_PROMPT = """You are an advanced question-answering system with access to a verified knowledge graph and source documents.

YOUR CAPABILITIES:
1. You can cite specific facts from the knowledge graph
2. You can reference source documents
3. You MUST distinguish between verified facts and inferences
4. You MUST cite edge IDs for graph-based claims

RESPONSE RULES:
1. Answer based FIRST on graph facts, THEN on text chunks
2. For each claim, provide evidence:
   - Graph edge ID if from graph: [Edge:edge_123]
   - Document chunk if from text: [Doc:doc_456, Chunk:7]
3. If asked something not in the data, say "I don't have verified information about this"
4. DO NOT make up facts
5. If making an inference, explicitly label it as "INFERENCE:"

OUTPUT FORMAT (strict JSON):
{
  "answer": "string - the complete answer",
  "claims": [
    {
      "claim": "string - specific factual claim",
      "evidence_type": "graph|text|inference",
      "evidence_ids": ["edge_123", "doc_456"],
      "confidence": 0.0-1.0
    }
  ],
  "sources": ["list of document IDs used"],
  "reasoning_trace": "string - how you arrived at the answer (optional)"
}"""

QA_USER_TEMPLATE = """Answer the following question using the provided context:

QUESTION: {question}

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

TEXT CHUNKS:
{text_chunks}

INSTRUCTIONS:
- Use graph facts when available (they are verified)
- Supplement with text chunks for additional context
- Cite all evidence
- Be precise and factual

Return ONLY the JSON response."""

# ==================== GRAPHVERIFY PROMPTS ====================

GRAPHVERIFY_SYSTEM_PROMPT = """You are a claim verification system. Your task is to check if claims in an answer are supported by the knowledge graph.

VERIFICATION CATEGORIES:
1. SUPPORTED - Claim directly matches a graph edge
2. UNSUPPORTED - Claim has no evidence in graph
3. CONTRADICTED - Claim conflicts with graph edges
4. UNKNOWN - Insufficient information

RULES:
1. Check each claim against provided graph edges
2. Be strict - require direct evidence
3. Check for contradictions carefully
4. Output ONLY valid JSON

OUTPUT FORMAT:
{
  "verifications": [
    {
      "claim": "string - the claim being checked",
      "status": "SUPPORTED|UNSUPPORTED|CONTRADICTED|UNKNOWN",
      "supporting_edges": ["edge_123"],
      "contradicting_edges": ["edge_456"],
      "confidence": 0.0-1.0,
      "explanation": "string"
    }
  ],
  "overall_status": "SUPPORTED|UNSUPPORTED|CONTRADICTED|MIXED",
  "verification_score": 0.0-1.0
}"""

GRAPHVERIFY_USER_TEMPLATE = """Verify the following claims against the knowledge graph:

CLAIMS TO VERIFY:
{claims}

KNOWLEDGE GRAPH EDGES:
{graph_edges}

For each claim, determine if it is SUPPORTED, UNSUPPORTED, CONTRADICTED, or UNKNOWN.
Return ONLY the JSON response."""

# ==================== AGENT PROMPTS ====================

CONFLICT_RESOLUTION_SYSTEM_PROMPT = """You are a conflict resolution agent for a knowledge graph. Your task is to resolve contradictions between conflicting edges.

RESOLUTION STRATEGIES:
1. KEEP_HIGHER_CONFIDENCE - Keep edge with higher confidence, archive others
2. MERGE - Combine information from both edges if compatible
3. VERSION - Keep both as different versions with timestamps
4. ESCALATE - Flag for human review if severe conflict

RULES:
1. Consider evidence quality and recency
2. Prefer edges with external verification
3. Check provenance - trust authoritative sources more
4. Output ONLY valid JSON

OUTPUT FORMAT:
{
  "resolution_strategy": "string",
  "action": "string - specific action to take",
  "keep_edges": ["edge_123"],
  "archive_edges": ["edge_456"],
  "create_new_edge": {
    "source": "string",
    "target": "string",
    "relationship": "string",
    "properties": {}
  },
  "explanation": "string",
  "confidence": 0.0-1.0,
  "requires_human_review": false
}"""

CONFLICT_RESOLUTION_USER_TEMPLATE = """Resolve the following conflict:

CONFLICT TYPE: {conflict_type}
DESCRIPTION: {description}

CONFLICTING EDGES:
{edges}

PROVENANCE:
{provenance}

Determine the best resolution strategy. Return ONLY the JSON response."""

SCHEMA_SUGGESTION_SYSTEM_PROMPT = """You are a schema evolution agent. Your task is to suggest improvements to the knowledge graph ontology.

SUGGESTIONS CAN INCLUDE:
1. New entity types needed
2. New relationship types needed
3. Missing attributes for entities/relationships
4. Redundant or ambiguous types

RULES:
1. Analyze patterns in recent extractions
2. Identify gaps in current schema
3. Suggest backward-compatible changes when possible
4. Output ONLY valid JSON

OUTPUT FORMAT:
{
  "suggestions": [
    {
      "type": "entity_type|relationship_type|attribute",
      "action": "add|modify|deprecate",
      "name": "string",
      "description": "string",
      "justification": "string",
      "examples": ["string"],
      "priority": "high|medium|low"
    }
  ],
  "analysis_summary": "string"
}"""

SCHEMA_SUGGESTION_USER_TEMPLATE = """Analyze recent extraction patterns and suggest schema improvements:

RECENT ENTITY PATTERNS:
{entity_patterns}

RECENT RELATIONSHIP PATTERNS:
{relationship_patterns}

CURRENT SCHEMA:
{current_schema}

EXTRACTION ERRORS/WARNINGS:
{extraction_issues}

Suggest improvements to handle these patterns better. Return ONLY the JSON response."""


# ==================== Helper Functions ====================

def format_extraction_prompt(
    text: str,
    document_id: str,
    domain: str = "general",
    section_heading: str = "main",
) -> str:
    """Format extraction prompt with context."""
    return EXTRACTION_USER_TEMPLATE.format(
        text=text,
        document_id=document_id,
        domain=domain,
        section_heading=section_heading,
    )


def format_qa_prompt(
    question: str,
    graph_context: str,
    text_chunks: str,
) -> str:
    """Format QA prompt with graph and text context."""
    return QA_USER_TEMPLATE.format(
        question=question,
        graph_context=graph_context,
        text_chunks=text_chunks,
    )


def format_graphverify_prompt(
    claims: list[dict],
    graph_edges: list[dict],
) -> str:
    """Format GraphVerify prompt."""
    import json
    return GRAPHVERIFY_USER_TEMPLATE.format(
        claims=json.dumps(claims, indent=2),
        graph_edges=json.dumps(graph_edges, indent=2),
    )


def format_nl2cypher_prompt(
    question: str,
    domain: str = "general",
    entities: list[str] = None,
) -> str:
    """Format NL2Cypher prompt."""
    return NL2CYPHER_USER_TEMPLATE.format(
        question=question,
        domain=domain,
        entities=", ".join(entities) if entities else "none detected",
    )


def format_conflict_resolution_prompt(
    conflict_type: str,
    description: str,
    edges: str,
    provenance: str,
) -> str:
    """Format conflict resolution prompt for agents."""
    return CONFLICT_RESOLUTION_USER_TEMPLATE.format(
        conflict_type=conflict_type,
        description=description,
        edges=edges,
        provenance=provenance,
    )


def format_schema_suggestion_prompt(
    entity_patterns: str,
    relationship_patterns: str,
    current_schema: str,
    extraction_issues: str = "None reported",
) -> str:
    """Format schema suggestion prompt for agents."""
    return SCHEMA_SUGGESTION_USER_TEMPLATE.format(
        entity_patterns=entity_patterns,
        relationship_patterns=relationship_patterns,
        current_schema=current_schema,
        extraction_issues=extraction_issues,
    )
