"""
DBpedia Benchmark Implementation

Tests knowledge graph extraction and triple validation.

Dataset: DBpedia (structured Wikipedia data)
Task: Extract and validate RDF triples from text
Focus: KG construction quality, triple precision/recall

Paper: http://dbpedia.org/
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from services.query.service import QueryService

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.base_benchmark import BaseBenchmark
from tests.benchmarks.metrics import MetricsCalculator


@dataclass
class DBpediaSample:
    """DBpedia dataset sample"""
    id: str
    text: str
    gold_triples: List[Tuple[str, str, str]]  # (subject, predicate, object)
    domain: str


class DBpediaBenchmark(BaseBenchmark):
    """DBpedia benchmark implementation"""
    
    def __init__(self, query_service: Optional["QueryService"] = None):
        super().__init__(
            name="DBpedia",
            output_dir=Path(__file__).parent.parent / "reports" / "dbpedia"
        )
        self.query_service = query_service
        self.data_dir = Path(__file__).parent.parent / "data" / "dbpedia"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_dataset(self, force: bool = False) -> None:
        """
        Download or create DBpedia dataset.
        
        Creates representative KG extraction samples.
        """
        data_file = self.data_dir / "dbpedia_samples.jsonl"
        
        if data_file.exists() and not force:
            print(f"✓ DBpedia data already exists: {data_file}")
            return
        
        print("Creating DBpedia samples...")
        
        # Create representative extraction tasks
        samples = self._create_dbpedia_samples()
        
        # Save to JSONL
        with open(data_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps({
                    'id': sample.id,
                    'text': sample.text,
                    'gold_triples': sample.gold_triples,
                    'domain': sample.domain
                }) + '\n')
        
        print(f"✓ Created {len(samples)} DBpedia samples: {data_file}")
    
    def _create_dbpedia_samples(self) -> List[DBpediaSample]:
        """Create representative DBpedia test samples"""
        
        samples = [
            # Person domain
            DBpediaSample(
                id="person_1",
                text="Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. He was born on March 14, 1879, in Ulm, Germany, and died on April 18, 1955, in Princeton, New Jersey.",
                gold_triples=[
                    ("Albert Einstein", "occupation", "theoretical physicist"),
                    ("Albert Einstein", "nationality", "German-born"),
                    ("Albert Einstein", "birth_date", "March 14, 1879"),
                    ("Albert Einstein", "birth_place", "Ulm, Germany"),
                    ("Albert Einstein", "death_date", "April 18, 1955"),
                    ("Albert Einstein", "death_place", "Princeton, New Jersey"),
                    ("Albert Einstein", "known_for", "theory of relativity")
                ],
                domain="person"
            ),
            DBpediaSample(
                id="person_2",
                text="Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different sciences.",
                gold_triples=[
                    ("Marie Curie", "occupation", "physicist"),
                    ("Marie Curie", "occupation", "chemist"),
                    ("Marie Curie", "nationality", "Polish-French"),
                    ("Marie Curie", "research_area", "radioactivity"),
                    ("Marie Curie", "award", "Nobel Prize"),
                    ("Marie Curie", "achievement", "first woman to win Nobel Prize")
                ],
                domain="person"
            ),
            
            # Organization domain
            DBpediaSample(
                id="org_1",
                text="Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976.",
                gold_triples=[
                    ("Apple Inc.", "type", "technology company"),
                    ("Apple Inc.", "headquarters", "Cupertino, California"),
                    ("Apple Inc.", "founded_date", "April 1, 1976"),
                    ("Apple Inc.", "founder", "Steve Jobs"),
                    ("Apple Inc.", "founder", "Steve Wozniak"),
                    ("Apple Inc.", "founder", "Ronald Wayne"),
                    ("Apple Inc.", "country", "United States")
                ],
                domain="organization"
            ),
            DBpediaSample(
                id="org_2",
                text="Google LLC is an American multinational technology company focusing on search engine technology, online advertising, cloud computing, and artificial intelligence. It was founded by Larry Page and Sergey Brin in 1998.",
                gold_triples=[
                    ("Google LLC", "type", "technology company"),
                    ("Google LLC", "founded_date", "1998"),
                    ("Google LLC", "founder", "Larry Page"),
                    ("Google LLC", "founder", "Sergey Brin"),
                    ("Google LLC", "industry", "search engine"),
                    ("Google LLC", "industry", "cloud computing"),
                    ("Google LLC", "industry", "artificial intelligence")
                ],
                domain="organization"
            ),
            
            # Place domain
            DBpediaSample(
                id="place_1",
                text="Paris is the capital and largest city of France. Located on the River Seine, it has a population of approximately 2.2 million residents. The city is known for landmarks like the Eiffel Tower and the Louvre Museum.",
                gold_triples=[
                    ("Paris", "type", "city"),
                    ("Paris", "capital_of", "France"),
                    ("Paris", "population", "2.2 million"),
                    ("Paris", "located_on", "River Seine"),
                    ("Paris", "landmark", "Eiffel Tower"),
                    ("Paris", "landmark", "Louvre Museum")
                ],
                domain="place"
            ),
            DBpediaSample(
                id="place_2",
                text="Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border runs across its summit point. Its elevation of 8,848.86 meters was most recently established in 2020.",
                gold_triples=[
                    ("Mount Everest", "type", "mountain"),
                    ("Mount Everest", "elevation", "8,848.86 meters"),
                    ("Mount Everest", "location", "Himalayas"),
                    ("Mount Everest", "location", "China-Nepal border"),
                    ("Mount Everest", "feature", "highest mountain")
                ],
                domain="place"
            ),
            
            # Work domain (books, movies, etc.)
            DBpediaSample(
                id="work_1",
                text="Inception is a 2010 science fiction action film written and directed by Christopher Nolan. The film stars Leonardo DiCaprio, and follows a professional thief who steals information by infiltrating the subconscious of his targets.",
                gold_triples=[
                    ("Inception", "type", "film"),
                    ("Inception", "genre", "science fiction"),
                    ("Inception", "release_year", "2010"),
                    ("Inception", "director", "Christopher Nolan"),
                    ("Inception", "writer", "Christopher Nolan"),
                    ("Inception", "starring", "Leonardo DiCaprio")
                ],
                domain="work"
            ),
            DBpediaSample(
                id="work_2",
                text="Harry Potter and the Philosopher's Stone is a fantasy novel written by British author J.K. Rowling. It is the first novel in the Harry Potter series and was first published in 1997 by Bloomsbury Publishing.",
                gold_triples=[
                    ("Harry Potter and the Philosopher's Stone", "type", "novel"),
                    ("Harry Potter and the Philosopher's Stone", "genre", "fantasy"),
                    ("Harry Potter and the Philosopher's Stone", "author", "J.K. Rowling"),
                    ("Harry Potter and the Philosopher's Stone", "publication_year", "1997"),
                    ("Harry Potter and the Philosopher's Stone", "publisher", "Bloomsbury Publishing"),
                    ("Harry Potter and the Philosopher's Stone", "series", "Harry Potter")
                ],
                domain="work"
            ),
            
            # Scientific concept domain
            DBpediaSample(
                id="science_1",
                text="DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil around each other to form a double helix. It carries genetic instructions for development, functioning, growth and reproduction of all known organisms.",
                gold_triples=[
                    ("DNA", "full_name", "deoxyribonucleic acid"),
                    ("DNA", "structure", "double helix"),
                    ("DNA", "composed_of", "polynucleotide chains"),
                    ("DNA", "function", "carries genetic instructions"),
                    ("DNA", "role", "development"),
                    ("DNA", "role", "reproduction")
                ],
                domain="science"
            ),
            DBpediaSample(
                id="science_2",
                text="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. During this process, plants convert carbon dioxide and water into glucose and oxygen.",
                gold_triples=[
                    ("Photosynthesis", "type", "biological process"),
                    ("Photosynthesis", "performed_by", "green plants"),
                    ("Photosynthesis", "requires", "sunlight"),
                    ("Photosynthesis", "requires", "chlorophyll"),
                    ("Photosynthesis", "input", "carbon dioxide"),
                    ("Photosynthesis", "input", "water"),
                    ("Photosynthesis", "output", "glucose"),
                    ("Photosynthesis", "output", "oxygen")
                ],
                domain="science"
            ),
            
            # Technology domain
            DBpediaSample(
                id="tech_1",
                text="Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                gold_triples=[
                    ("Python", "type", "programming language"),
                    ("Python", "creator", "Guido van Rossum"),
                    ("Python", "release_year", "1991"),
                    ("Python", "paradigm", "procedural"),
                    ("Python", "paradigm", "object-oriented"),
                    ("Python", "paradigm", "functional"),
                    ("Python", "feature", "code readability")
                ],
                domain="technology"
            ),
            DBpediaSample(
                id="tech_2",
                text="Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, which are linked using cryptography. Each block contains a cryptographic hash of the previous block, timestamp, and transaction data.",
                gold_triples=[
                    ("Blockchain", "type", "distributed ledger technology"),
                    ("Blockchain", "component", "blocks"),
                    ("Blockchain", "uses", "cryptography"),
                    ("Blockchain", "contains", "cryptographic hash"),
                    ("Blockchain", "contains", "timestamp"),
                    ("Blockchain", "contains", "transaction data"),
                    ("Blockchain", "feature", "continuously growing")
                ],
                domain="technology"
            ),
        ]
        
        return samples
    
    def load_data(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load DBpedia data from JSONL file"""
        data_file = self.data_dir / "dbpedia_samples.jsonl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"DBpedia data not found: {data_file}")
        
        samples = []
        with open(data_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Replicate samples to reach desired size
        if sample_size and sample_size > len(samples):
            multiplier = (sample_size // len(samples)) + 1
            samples = (samples * multiplier)[:sample_size]
        elif sample_size:
            samples = samples[:sample_size]
        
        return samples
    
    def prepare_input(self, sample: Dict[str, Any]) -> str:
        """Format text for KG extraction"""
        return f"Extract knowledge graph triples from the following text: {sample['text']}"
    
    async def run_system(self, input_text: str) -> Dict[str, Any]:
        """Run GraphBuilder system for KG extraction"""
        if not self.query_service:
            raise ValueError("QueryService not initialized")
        
        try:
            # Use the system to extract and store triples
            result = await self.query_service.process_query(
                query=input_text,
                user_id="benchmark_dbpedia",
                session_id="dbpedia_test",
                use_graph_verify=False,
                use_nl2cypher=False
            )
            return result
        except Exception as e:
            print(f"Error running system: {str(e)}")
            return {"error": str(e)}
    
    def extract_prediction(self, system_output: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Extract predicted triples from system output"""
        if "error" in system_output:
            return []
        
        # Extract triples from graph_data if available
        triples = []
        graph_data = system_output.get("graph_data", {})
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Convert edges to triples
        for edge in edges:
            subject = edge.get("source", "")
            predicate = edge.get("type", "")
            obj = edge.get("target", "")
            if subject and predicate and obj:
                triples.append((subject, predicate, obj))
        
        return triples
    
    def extract_gold_label(self, sample: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Extract gold standard triples"""
        return [tuple(triple) for triple in sample['gold_triples']]
    
    def calculate_metrics(
        self,
        predictions: List[Any],  # List of triple lists
        gold_labels: List[Any],  # List of triple lists
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate DBpedia metrics"""
        calculator = MetricsCalculator()
        
        # Calculate triple-level precision, recall, F1
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for pred_triples, gold_triples in zip(predictions, gold_labels):
            if not gold_triples:
                continue
            
            pred_set = set(pred_triples) if pred_triples else set()
            gold_set = set(gold_triples)
            
            # Calculate triple accuracy
            precision, recall, f1 = calculator.triple_accuracy(
                list(pred_set),
                list(gold_set)
            )
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
        
        avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
        avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
        
        # Per-domain metrics
        domain_metrics = {}
        for domain in ["person", "organization", "place", "work", "science", "technology"]:
            domain_samples = [
                (pred, gold) for pred, gold, sample in zip(predictions, gold_labels, samples)
                if sample['domain'] == domain
            ]
            
            if domain_samples:
                domain_preds, domain_golds = zip(*domain_samples)
                domain_precisions = []
                domain_recalls = []
                
                for pred, gold in zip(domain_preds, domain_golds):
                    pred_set = set(pred) if pred else set()
                    gold_set = set(gold)
                    precision, recall, _ = calculator.triple_accuracy(list(pred_set), list(gold_set))
                    domain_precisions.append(precision)
                    domain_recalls.append(recall)
                
                domain_metrics[f"{domain}_precision"] = sum(domain_precisions) / len(domain_precisions)
                domain_metrics[f"{domain}_recall"] = sum(domain_recalls) / len(domain_recalls)
        
        return {
            "triple_precision": avg_precision,
            "triple_recall": avg_recall,
            "triple_f1": avg_f1,
            **domain_metrics
        }


async def run_dbpedia_benchmark(sample_size: int = 100):
    """Run DBpedia benchmark"""
    from api.core.database import Database
    from api.core.redis_client import RedisClient
    from api.core.neo4j_client import Neo4jClient
    from api.core.groq_client import GroqClient
    
    # Initialize services
    db = Database()
    await db.connect()
    
    redis_client = RedisClient()
    await redis_client.connect()
    
    neo4j = Neo4jClient()
    await neo4j.connect()
    
    groq = GroqClient()
    await groq.initialize()
    
    query_service = QueryService(db, redis_client, neo4j, groq)
    
    # Run benchmark
    benchmark = DBpediaBenchmark(query_service)
    
    try:
        await benchmark.download_dataset()
        results = await benchmark.run(sample_size=sample_size)
        print(f"\n{'='*60}")
        print("DBpedia Benchmark Results:")
        print(f"{'='*60}")
        print(benchmark.get_summary())
        return results
    finally:
        await db.disconnect()
        await redis_client.disconnect()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(run_dbpedia_benchmark(sample_size=100))
