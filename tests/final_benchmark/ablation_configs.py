"""
Ablation Study Configurations

Defines system configurations for ablation testing:
- Full system (baseline)
- No GraphVerify
- No entity resolution
- No external verification
- Vector-only RAG
- Graph-only retrieval
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    description: str
    disable_components: List[str]
    retrieval_mode: str  # "hybrid", "vector_only", "graph_only"
    enable_graphverify: bool
    enable_entity_resolution: bool
    enable_external_verification: bool
    enable_fusion: bool
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "disable_components": self.disable_components,
            "retrieval_mode": self.retrieval_mode,
            "enable_graphverify": self.enable_graphverify,
            "enable_entity_resolution": self.enable_entity_resolution,
            "enable_external_verification": self.enable_external_verification,
            "enable_fusion": self.enable_fusion,
        }


# Configuration definitions
ABLATION_CONFIGS = {
    "fever_full_system": AblationConfig(
        name="FEVER Full System",
        description="All components enabled - baseline for ablation comparison",
        disable_components=[],
        retrieval_mode="hybrid",
        enable_graphverify=True,
        enable_entity_resolution=True,
        enable_external_verification=True,
        enable_fusion=True,
    ),
    
    "fever_no_graphverify": AblationConfig(
        name="FEVER No GraphVerify",
        description="Disable verification - shows impact on hallucination rate",
        disable_components=["graphverify"],
        retrieval_mode="hybrid",
        enable_graphverify=False,
        enable_entity_resolution=True,
        enable_external_verification=True,
        enable_fusion=True,
    ),
    
    "hotpotqa_vector_only": AblationConfig(
        name="HotpotQA Vector-only RAG",
        description="Standard baseline - FAISS + LLM (what everyone uses)",
        disable_components=["graph_retrieval", "graphverify", "entity_resolution"],
        retrieval_mode="vector_only",
        enable_graphverify=False,
        enable_entity_resolution=False,
        enable_external_verification=False,
        enable_fusion=False,
    ),
    
    "hotpotqa_graph_only": AblationConfig(
        name="HotpotQA Graph-only",
        description="Pure Neo4j retrieval - shows graph alone isn't enough",
        disable_components=["vector_retrieval"],
        retrieval_mode="graph_only",
        enable_graphverify=True,
        enable_entity_resolution=True,
        enable_external_verification=True,
        enable_fusion=True,
    ),
    
    "hotpotqa_hybrid": AblationConfig(
        name="HotpotQA Hybrid (Full System)",
        description="Our contribution - vector + graph + GraphVerify",
        disable_components=[],
        retrieval_mode="hybrid",
        enable_graphverify=True,
        enable_entity_resolution=True,
        enable_external_verification=True,
        enable_fusion=True,
    ),
}


def get_config(config_name: str) -> AblationConfig:
    """Get ablation configuration by name."""
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(ABLATION_CONFIGS.keys())}"
        )
    return ABLATION_CONFIGS[config_name]


def list_configs() -> List[str]:
    """List all available configuration names."""
    return list(ABLATION_CONFIGS.keys())


# Test sample sets
FEVER_TEST_SAMPLES = [
    # 25 FEVER fact verification claims (balanced: SUPPORTS, REFUTES, NEI)
    ("Albert Einstein won the Nobel Prize in Physics in 1921", "SUPPORTS"),
    ("Marie Curie was born in Warsaw, Poland", "SUPPORTS"),
    ("Isaac Newton published Principia Mathematica", "SUPPORTS"),
    ("Stephen Hawking wrote A Brief History of Time", "SUPPORTS"),
    ("Charles Darwin published On the Origin of Species in 1859", "SUPPORTS"),
    ("Python was created by Guido van Rossum", "SUPPORTS"),
    ("Mount Everest is 8,849 meters tall", "SUPPORTS"),
    ("Jupiter is the largest planet in Solar System", "SUPPORTS"),
    
    ("Albert Einstein won the Nobel Prize in Chemistry", "REFUTES"),
    ("Marie Curie was born in France", "REFUTES"),
    ("Isaac Newton was born in 1700", "REFUTES"),
    ("Stephen Hawking was born in 1960", "REFUTES"),
    ("Python was created by Linus Torvalds", "REFUTES"),
    ("Mount Everest is 9,000 meters tall", "REFUTES"),
    ("Mars is the largest planet in Solar System", "REFUTES"),
    
    ("Albert Einstein played the violin professionally", "NOT ENOUGH INFO"),
    ("Marie Curie had three children", "NOT ENOUGH INFO"),
    ("Isaac Newton owned a pet dog named Diamond", "NOT ENOUGH INFO"),
    ("Stephen Hawking enjoyed classical music", "NOT ENOUGH INFO"),
    ("Python's name comes from Monty Python", "NOT ENOUGH INFO"),
    ("Mount Everest was first climbed in May", "NOT ENOUGH INFO"),
    ("Jupiter has exactly 79 moons", "NOT ENOUGH INFO"),
    ("Saturn's rings are made of ice and rock", "NOT ENOUGH INFO"),
    ("The Pacific Ocean contains 25,000 islands", "NOT ENOUGH INFO"),
]

HOTPOTQA_TEST_SAMPLES = [
    # 25 HotpotQA multi-hop questions
    (
        "What year was the creator of Python born?",
        "1956",  # Guido van Rossum
        ["Python", "Guido van Rossum", "birth year"]
    ),
    (
        "Which university did the founder of Microsoft attend?",
        "Harvard University",
        ["Microsoft", "Bill Gates", "university"]
    ),
    (
        "What is the capital of the country where Marie Curie was born?",
        "Warsaw",
        ["Marie Curie", "Poland", "capital"]
    ),
    (
        "In what year did the author of A Brief History of Time receive his PhD?",
        "1966",
        ["A Brief History of Time", "Stephen Hawking", "PhD"]
    ),
    (
        "What programming language was created by the person born in 1969 in Finland?",
        "Linux",
        ["Linus Torvalds", "1969", "Finland", "programming"]
    ),
    (
        "Which company was founded in the same year as Microsoft?",
        "Apple Inc.",  # Both 1975-1976
        ["Microsoft", "1975", "founded", "company"]
    ),
    (
        "What is the highest mountain in the country where K2 is located?",
        "K2",
        ["K2", "Pakistan", "highest mountain"]
    ),
    (
        "Which ocean has the deepest point in the Mariana Trench?",
        "Pacific Ocean",
        ["Mariana Trench", "ocean", "deepest"]
    ),
    (
        "What is the largest moon of the largest planet in our solar system?",
        "Ganymede",
        ["Jupiter", "largest planet", "moon"]
    ),
    (
        "In which decade was the creator of the World Wide Web born?",
        "1950s",
        ["World Wide Web", "Tim Berners-Lee", "birth"]
    ),
    (
        "What is the second tallest mountain in the world?",
        "K2",
        ["mountain", "second tallest", "world"]
    ),
    (
        "Which planet has the most moons in our solar system?",
        "Saturn",
        ["planet", "moons", "solar system"]
    ),
    (
        "What year did the company founded by Jeff Bezos go public?",
        "1997",
        ["Jeff Bezos", "Amazon", "IPO", "public"]
    ),
    (
        "Which programming language shares its name with a British comedy group?",
        "Python",
        ["programming language", "British comedy", "Monty Python"]
    ),
    (
        "What is the capital of the country with the most pyramids?",
        "Khartoum",  # Sudan has 200+ pyramids
        ["pyramids", "most", "capital"]
    ),
    (
        "Which sea is the saltiest body of water on Earth?",
        "Dead Sea",
        ["saltiest", "water", "Earth"]
    ),
    (
        "What year was the Nobel Prize first awarded?",
        "1901",
        ["Nobel Prize", "first", "year"]
    ),
    (
        "Which element is named after the creator of the periodic table?",
        "Mendelevium",
        ["element", "periodic table", "Mendeleev"]
    ),
    (
        "What is the longest river in the continent where the Amazon flows?",
        "Amazon River",
        ["longest river", "South America", "Amazon"]
    ),
    (
        "Which physicist has a unit of frequency named after them?",
        "Heinrich Hertz",
        ["physicist", "frequency", "unit", "hertz"]
    ),
    (
        "What is the smallest country that is also a continent?",
        "Australia",
        ["smallest", "country", "continent"]
    ),
    (
        "Which chemical element has the symbol Au?",
        "Gold",
        ["chemical element", "symbol", "Au"]
    ),
    (
        "What is the speed of light in meters per second?",
        "299,792,458",
        ["speed of light", "meters per second"]
    ),
    (
        "Which planet is known as the Red Planet?",
        "Mars",
        ["planet", "Red Planet"]
    ),
    (
        "What year did World War II end?",
        "1945",
        ["World War II", "end", "year"]
    ),
]


def get_test_samples(dataset: str) -> List:
    """Get test samples for specified dataset."""
    if dataset.lower() == "fever":
        return FEVER_TEST_SAMPLES[:25]
    elif dataset.lower() == "hotpotqa":
        return HOTPOTQA_TEST_SAMPLES[:25]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
