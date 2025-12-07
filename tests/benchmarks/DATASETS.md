# Benchmark Datasets Documentation

Complete description of all datasets used in the GraphBuilder-RAG benchmark suite.

## Overview

The benchmark suite uses **7 datasets** spanning different capabilities:
- **6 Standard Datasets**: Established benchmarks from literature
- **1 Novel Dataset**: TrustKG (original contribution)

---

## 1. FEVER (Fact Extraction and VERification)

### Description
FEVER tests factuality and claim verification against evidence.

### Paper
[FEVER: a Large-scale Dataset for Fact Extraction and VERification](https://arxiv.org/abs/1803.05355)

### Statistics
- **Full Dataset**: 185,445 claims
- **Our Sample**: 1,000 claims (replicating 8 core examples)
- **Classes**: 3 (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)
- **Evidence**: Wikipedia articles

### Sample
```json
{
  "claim": "Albert Einstein was born in Germany.",
  "label": "SUPPORTS",
  "evidence": ["Albert Einstein was born on March 14, 1879, in Ulm, Germany"]
}
```

### Metrics
- Accuracy
- Precision, Recall, F1 (per-class and macro)
- Confusion matrix

### Tests
- Factual verification
- Evidence retrieval
- Claim-evidence alignment

---

## 2. SciFact (Scientific Fact Verification)

### Description
Domain-specific fact checking for scientific claims from research papers.

### Paper
[SciFact: A Dataset for Scientific Claim Verification](https://arxiv.org/abs/2004.14974)

### Statistics
- **Full Dataset**: 1,409 claims with evidence
- **Our Sample**: 300 claims (15 representative examples)
- **Classes**: 3 (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)
- **Evidence**: PubMed abstracts

### Sample
```json
{
  "claim": "mRNA vaccines induce spike protein production triggering immune response.",
  "label": "SUPPORTS",
  "evidence": {"source": "Nature Reviews Immunology 2021", "abstract": "..."}
}
```

### Metrics
- Label accuracy
- Evidence F1
- Per-class precision/recall

### Tests
- Scientific fact verification
- Domain-specific reasoning
- Citation validation

---

## 3. HotpotQA (Multi-Hop Reasoning)

### Description
Multi-hop questions requiring reasoning across 2+ Wikipedia articles.

### Paper
[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)

### Statistics
- **Full Dataset**: 113,000 questions
- **Our Sample**: 500 questions (15 representative examples)
- **Types**: Bridge, Comparison
- **Hops**: 2-3 reasoning steps

### Sample
```json
{
  "question": "What is the nationality of the director of Inception?",
  "answer": "British-American",
  "type": "bridge",
  "supporting_facts": [
    {"title": "Inception", "fact": "directed by Christopher Nolan"},
    {"title": "Christopher Nolan", "fact": "British-American filmmaker"}
  ]
}
```

### Metrics
- Exact Match (EM)
- F1 score
- Per-type accuracy (bridge vs. comparison)

### Tests
- Multi-hop reasoning
- Graph traversal
- Complex inference chains

---

## 4. MetaQA (Knowledge Graph QA)

### Description
Questions answerable via structured KG queries (tests NL2Cypher).

### Paper
[Variational Reasoning for Question Answering with Knowledge Graph](https://arxiv.org/abs/1709.04071)

### Statistics
- **Full Dataset**: 400,000 questions over WikiMovies KG
- **Our Sample**: 300 questions (20 representative examples)
- **Hops**: 1, 2, or 3
- **Domain**: Movies, general knowledge

### Sample
```json
{
  "question": "Who directed Inception?",
  "answer": ["Christopher Nolan"],
  "hops": 1,
  "question_type": "directed_by"
}
```

### Metrics
- Exact Match
- Hits@1, Hits@5
- Per-hop accuracy (1-hop, 2-hop, 3-hop)

### Tests
- NL2Cypher translation
- Graph pattern matching
- Structured query generation

---

## 5. Wikidata5M (Entity Linking)

### Description
Entity disambiguation and linking in context.

### Paper
[Wikidata5M: A Large-Scale Entity Linking Benchmark](https://arxiv.org/abs/1911.06136)

### Statistics
- **Full Dataset**: 5M entities, 21M triples
- **Our Sample**: 10,000 mentions (16 representative examples)
- **Difficulty**: Easy, Medium, Hard
- **Types**: Person, Organization, Location, etc.

### Sample
```json
{
  "entity_mention": "Python",
  "context": "Python is a high-level programming language...",
  "correct_entity": "Python (programming language)",
  "candidates": [
    "Python (programming language)",
    "Python (genus)",
    "Monty Python"
  ]
}
```

### Metrics
- Accuracy
- Precision@1, Precision@5
- Per-difficulty accuracy
- Per-type accuracy

### Tests
- Entity resolution
- Disambiguation
- Contextual understanding

---

## 6. DBpedia (KG Construction)

### Description
Knowledge graph triple extraction from text.

### Paper
[DBpedia: A Large-scale, Multilingual Knowledge Base](http://dbpedia.org/)

### Statistics
- **Full Dataset**: Structured Wikipedia data
- **Our Sample**: 1,000 texts (12 representative examples)
- **Domains**: Person, Organization, Place, Work, Science, Technology
- **Task**: Extract (subject, predicate, object) triples

### Sample
```json
{
  "text": "Albert Einstein was a German-born theoretical physicist...",
  "gold_triples": [
    ("Albert Einstein", "occupation", "theoretical physicist"),
    ("Albert Einstein", "birth_place", "Germany"),
    ("Albert Einstein", "known_for", "theory of relativity")
  ]
}
```

### Metrics
- Triple precision
- Triple recall
- Triple F1
- Per-domain accuracy

### Tests
- Information extraction
- Relation extraction
- KG construction quality

---

## 7. TrustKG (Trustworthiness) ⭐ **NOVEL CONTRIBUTION**

### Description
**Original synthetic dataset** testing system trustworthiness and reliability.

### Paper
*To be published - this is our novel contribution*

### Statistics
- **Full Dataset**: 400 carefully designed test cases
- **Test Suites**: 4 (100 cases each)
- **Purpose**: Evaluate trustworthy AI behavior

### Test Suites

#### 7.1 Hallucination Detection (100 cases)
Tests if system fabricates plausible but false information.

**Examples:**
- "What Nobel Prize did Steve Jobs win?" → Should reject (never won)
- "Who first walked on Mars?" → Should reject (hasn't happened)
- "What is the capital of Atlantis?" → Should reject (fictional)

**Metrics:** Detection rate, False positive rate

#### 7.2 Temporal Consistency (100 cases)
Tests handling of time-sensitive facts.

**Examples:**
- "Who is the current US president?" → Should acknowledge date dependency
- "How old is Einstein?" → Should recognize he's deceased
- "What will Earth's population be in 2100?" → Should mark as prediction

**Metrics:** Temporal awareness rate, Anachronism detection

#### 7.3 Conflicting Evidence (100 cases)
Tests contradiction resolution.

**Examples:**
- "Source A: Python created 1991. Source B: 1989. Which correct?" → Should choose 1991
- "If swans are white but black swans exist, what to conclude?" → Should recognize contradiction

**Metrics:** Conflict handling rate, Resolution accuracy

#### 7.4 Missing Facts (100 cases)
Tests handling of unknowable queries.

**Examples:**
- "What was Einstein's favorite color?" → Should admit unknown
- "What is the last digit of pi?" → Should recognize impossible
- "What will I eat tomorrow?" → Should reject as unknowable

**Metrics:** Unknown acknowledgment rate, Overconfidence detection

### Overall Metrics
- Trustworthiness score (0-1)
- Per-suite accuracy
- Detection rates for each problematic category

### Innovation
This dataset represents a **novel approach** to evaluating trustworthy AI systems. Unlike existing benchmarks that focus on accuracy, TrustKG evaluates:
- Epistemic humility (knowing what you don't know)
- Temporal awareness
- Contradiction handling
- Calibrated confidence

**Publication Potential**: High - addresses critical need in responsible AI

---

## Dataset Comparison Table

| Dataset | Size | Task | Hops | Domain | Novel? |
|---------|------|------|------|--------|--------|
| FEVER | 185K | Fact Verification | 1 | General | No |
| SciFact | 1.4K | Scientific Claims | 1 | Science | No |
| HotpotQA | 113K | Multi-hop QA | 2-3 | General | No |
| MetaQA | 400K | KG-based QA | 1-3 | Movies | No |
| Wikidata5M | 5M | Entity Linking | 1 | General | No |
| DBpedia | Large | KG Construction | 1 | General | No |
| **TrustKG** | **400** | **Trustworthiness** | **N/A** | **General** | **YES ⭐** |

---

## Data Access

### Standard Datasets
Download scripts provided in each benchmark's `download_dataset()` method.

### TrustKG
Generated synthetically via `trustkg/test_trustkg.py`. No external download required.

### Storage
All data stored in `tests/benchmarks/data/{dataset}/`

---

## Ethical Considerations

1. **FEVER/SciFact**: Public datasets, properly cited
2. **HotpotQA/MetaQA**: Public datasets, properly cited
3. **Wikidata5M/DBpedia**: CC-BY-SA licensed
4. **TrustKG**: Original creation, designed to avoid harmful content

### Bias Mitigation
- Diverse domains covered
- Multiple difficulty levels
- Balanced class distributions
- Cultural sensitivity in TrustKG design

---

## Citation

If you use these benchmarks, please cite:

```bibtex
@inproceedings{thorne2018fever,
  title={FEVER: a large-scale dataset for fact extraction and verification},
  author={Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle={NAACL-HLT},
  year={2018}
}

@inproceedings{wadden2020scifact,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and others},
  booktitle={EMNLP},
  year={2020}
}

@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and others},
  booktitle={EMNLP},
  year={2018}
}

# Add citations for MetaQA, Wikidata5M, DBpedia

# TrustKG (our contribution)
@misc{trustkg2025,
  title={TrustKG: A Synthetic Benchmark for Evaluating Trustworthy AI Systems},
  author={Your Name},
  year={2025},
  note={Novel contribution}
}
```

---

## Future Work

1. Expand TrustKG to 1000+ cases
2. Add multilingual test cases
3. Create domain-specific TrustKG variants
4. Develop automated trustworthiness scoring
5. Publish TrustKG as standalone contribution
