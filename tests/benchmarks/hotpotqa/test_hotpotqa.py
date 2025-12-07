"""
HotpotQA Benchmark Implementation

Tests multi-hop reasoning capabilities using GraphBuilder-RAG system.

Dataset: HotpotQA (113K Wikipedia-based multi-hop questions)
Task: Answer questions requiring 2+ reasoning hops across documents
Focus: Graph traversal, multi-step inference, complex reasoning chains

Paper: https://arxiv.org/abs/1809.09600
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from services.query.service import QueryService

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.base_benchmark import BaseBenchmark
from tests.benchmarks.metrics import MetricsCalculator


@dataclass
class HotpotQASample:
    """HotpotQA dataset sample"""
    id: str
    question: str
    answer: str
    type: str  # bridge or comparison
    level: str  # easy, medium, hard
    supporting_facts: List[Dict[str, Any]]


class HotpotQABenchmark(BaseBenchmark):
    """HotpotQA benchmark implementation"""
    
    def __init__(self, query_service: Optional["QueryService"] = None):
        super().__init__(
            name="HotpotQA",
            output_dir=Path(__file__).parent.parent / "reports" / "hotpotqa"
        )
        self.query_service = query_service
        self.data_dir = Path(__file__).parent.parent / "data" / "hotpotqa"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_dataset(self, force: bool = False) -> None:
        """
        Download or create HotpotQA dataset.
        
        Creates representative multi-hop reasoning questions.
        """
        data_file = self.data_dir / "hotpotqa_samples.jsonl"
        
        if data_file.exists() and not force:
            print(f"✓ HotpotQA data already exists: {data_file}")
            return
        
        print("Creating HotpotQA samples...")
        
        # Create representative multi-hop questions
        samples = self._create_hotpotqa_samples()
        
        # Save to JSONL
        with open(data_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps({
                    'id': sample.id,
                    'question': sample.question,
                    'answer': sample.answer,
                    'type': sample.type,
                    'level': sample.level,
                    'supporting_facts': sample.supporting_facts
                }) + '\n')
        
        print(f"✓ Created {len(samples)} HotpotQA samples: {data_file}")
    
    def _create_hotpotqa_samples(self) -> List[HotpotQASample]:
        """Create representative HotpotQA test samples"""
        
        samples = [
            # Bridge questions (connecting two entities through intermediate)
            HotpotQASample(
                id="bridge_1",
                question="What is the nationality of the director of the film Inception?",
                answer="British-American",
                type="bridge",
                level="easy",
                supporting_facts=[
                    {"title": "Inception", "fact": "Inception is directed by Christopher Nolan"},
                    {"title": "Christopher Nolan", "fact": "Christopher Nolan is a British-American filmmaker"}
                ]
            ),
            HotpotQASample(
                id="bridge_2",
                question="In what year was the university where Albert Einstein worked as a professor founded?",
                answer="1905",
                type="bridge",
                level="medium",
                supporting_facts=[
                    {"title": "Albert Einstein", "fact": "Einstein was a professor at University of Bern"},
                    {"title": "University of Bern", "fact": "University of Bern was founded in 1834"}
                ]
            ),
            HotpotQASample(
                id="bridge_3",
                question="What award did the author of 'Harry Potter' receive in 2001?",
                answer="Hugo Award",
                type="bridge",
                level="easy",
                supporting_facts=[
                    {"title": "Harry Potter", "fact": "Harry Potter was written by J.K. Rowling"},
                    {"title": "J.K. Rowling", "fact": "J.K. Rowling received the Hugo Award in 2001"}
                ]
            ),
            HotpotQASample(
                id="bridge_4",
                question="Which programming language was developed by the creator of Python?",
                answer="Python",
                type="bridge",
                level="easy",
                supporting_facts=[
                    {"title": "Python", "fact": "Python was created by Guido van Rossum"},
                    {"title": "Guido van Rossum", "fact": "Guido van Rossum developed Python"}
                ]
            ),
            HotpotQASample(
                id="bridge_5",
                question="What is the capital of the country where the Eiffel Tower is located?",
                answer="Paris",
                type="bridge",
                level="easy",
                supporting_facts=[
                    {"title": "Eiffel Tower", "fact": "The Eiffel Tower is located in Paris, France"},
                    {"title": "France", "fact": "The capital of France is Paris"}
                ]
            ),
            
            # Comparison questions (comparing two entities)
            HotpotQASample(
                id="comp_1",
                question="Which company was founded first, Apple or Microsoft?",
                answer="Apple",
                type="comparison",
                level="medium",
                supporting_facts=[
                    {"title": "Apple Inc.", "fact": "Apple was founded on April 1, 1976"},
                    {"title": "Microsoft", "fact": "Microsoft was founded on April 4, 1975"}
                ]
            ),
            HotpotQASample(
                id="comp_2",
                question="Who was born earlier, Isaac Newton or Galileo Galilei?",
                answer="Galileo Galilei",
                type="comparison",
                level="easy",
                supporting_facts=[
                    {"title": "Isaac Newton", "fact": "Isaac Newton was born on January 4, 1643"},
                    {"title": "Galileo Galilei", "fact": "Galileo Galilei was born on February 15, 1564"}
                ]
            ),
            HotpotQASample(
                id="comp_3",
                question="Which mountain is taller, Mount Everest or K2?",
                answer="Mount Everest",
                type="comparison",
                level="easy",
                supporting_facts=[
                    {"title": "Mount Everest", "fact": "Mount Everest is 8,849 meters tall"},
                    {"title": "K2", "fact": "K2 is 8,611 meters tall"}
                ]
            ),
            HotpotQASample(
                id="comp_4",
                question="Which planet is larger, Jupiter or Saturn?",
                answer="Jupiter",
                type="comparison",
                level="easy",
                supporting_facts=[
                    {"title": "Jupiter", "fact": "Jupiter has a diameter of 139,820 km"},
                    {"title": "Saturn", "fact": "Saturn has a diameter of 116,460 km"}
                ]
            ),
            HotpotQASample(
                id="comp_5",
                question="Which ocean is deeper, the Atlantic or the Pacific?",
                answer="Pacific Ocean",
                type="comparison",
                level="medium",
                supporting_facts=[
                    {"title": "Atlantic Ocean", "fact": "The Atlantic Ocean has an average depth of 3,646 meters"},
                    {"title": "Pacific Ocean", "fact": "The Pacific Ocean has an average depth of 4,280 meters"}
                ]
            ),
            
            # Hard multi-hop questions (3+ hops)
            HotpotQASample(
                id="hard_1",
                question="What is the population of the city where the headquarters of the company that developed ChatGPT is located?",
                answer="San Francisco",
                type="bridge",
                level="hard",
                supporting_facts=[
                    {"title": "ChatGPT", "fact": "ChatGPT was developed by OpenAI"},
                    {"title": "OpenAI", "fact": "OpenAI is headquartered in San Francisco, California"},
                    {"title": "San Francisco", "fact": "San Francisco has a population of approximately 873,965"}
                ]
            ),
            HotpotQASample(
                id="hard_2",
                question="In which year did the country that won the most recent FIFA World Cup gain independence?",
                answer="1816",
                type="bridge",
                level="hard",
                supporting_facts=[
                    {"title": "2022 FIFA World Cup", "fact": "Argentina won the 2022 FIFA World Cup"},
                    {"title": "Argentina", "fact": "Argentina gained independence on July 9, 1816"}
                ]
            ),
            HotpotQASample(
                id="hard_3",
                question="What is the birth year of the person who invented the device that Alexander Graham Bell is famous for?",
                answer="1847",
                type="bridge",
                level="hard",
                supporting_facts=[
                    {"title": "Alexander Graham Bell", "fact": "Bell is famous for inventing the telephone"},
                    {"title": "Alexander Graham Bell", "fact": "Alexander Graham Bell was born on March 3, 1847"}
                ]
            ),
            HotpotQASample(
                id="hard_4",
                question="Which came first: the founding of the university attended by the first President of the United States, or the Declaration of Independence?",
                answer="University founding",
                type="comparison",
                level="hard",
                supporting_facts=[
                    {"title": "George Washington", "fact": "George Washington did not attend university"},
                    {"title": "Declaration of Independence", "fact": "The Declaration of Independence was signed on July 4, 1776"}
                ]
            ),
            HotpotQASample(
                id="hard_5",
                question="What language family does the primary language of the country with the tallest building in the world belong to?",
                answer="Semitic",
                type="bridge",
                level="hard",
                supporting_facts=[
                    {"title": "Tallest building", "fact": "The Burj Khalifa in Dubai is the tallest building"},
                    {"title": "United Arab Emirates", "fact": "The primary language of UAE is Arabic"},
                    {"title": "Arabic", "fact": "Arabic belongs to the Semitic language family"}
                ]
            ),
        ]
        
        return samples
    
    def load_data(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load HotpotQA data from JSONL file"""
        data_file = self.data_dir / "hotpotqa_samples.jsonl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"HotpotQA data not found: {data_file}")
        
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
        """Format question for multi-hop reasoning"""
        return sample['question']
    
    async def run_system(self, input_text: str) -> Dict[str, Any]:
        """Run GraphBuilder system with graph traversal"""
        if not self.query_service:
            raise ValueError("QueryService not initialized")
        
        try:
            result = await self.query_service.process_query(
                query=input_text,
                user_id="benchmark_hotpotqa",
                session_id="hotpotqa_test",
                use_graph_verify=False,  # Not verification task
                use_nl2cypher=True  # Enable NL2Cypher for complex queries
            )
            return result
        except Exception as e:
            print(f"Error running system: {str(e)}")
            return {"error": str(e)}
    
    def extract_prediction(self, system_output: Dict[str, Any]) -> str:
        """Extract predicted answer from system output"""
        if "error" in system_output:
            return ""
        
        # Extract answer from response
        answer = system_output.get("answer", "")
        if isinstance(answer, str):
            return answer.strip()
        return ""
    
    def extract_gold_label(self, sample: Dict[str, Any]) -> str:
        """Extract gold standard answer"""
        return sample['answer'].strip()
    
    def calculate_metrics(
        self,
        predictions: List[str],
        gold_labels: List[str],
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate HotpotQA metrics"""
        calculator = MetricsCalculator()
        
        # Exact Match (EM)
        em = calculator.exact_match(predictions, gold_labels)
        
        # F1 score (token-level overlap)
        f1_scores = [
            calculator.f1_score_qa(pred, gold)
            for pred, gold in zip(predictions, gold_labels)
        ]
        f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # Per-type metrics
        bridge_em, bridge_f1 = 0.0, 0.0
        comp_em, comp_f1 = 0.0, 0.0
        
        bridge_samples = [(p, g) for (p, g, s) in zip(predictions, gold_labels, samples) if s['type'] == 'bridge']
        comp_samples = [(p, g) for (p, g, s) in zip(predictions, gold_labels, samples) if s['type'] == 'comparison']
        
        if bridge_samples:
            bridge_preds, bridge_golds = zip(*bridge_samples)
            bridge_em = calculator.exact_match(list(bridge_preds), list(bridge_golds))
            bridge_f1 = sum(calculator.f1_score_qa(p, g) for p, g in bridge_samples) / len(bridge_samples)
        
        if comp_samples:
            comp_preds, comp_golds = zip(*comp_samples)
            comp_em = calculator.exact_match(list(comp_preds), list(comp_golds))
            comp_f1 = sum(calculator.f1_score_qa(p, g) for p, g in comp_samples) / len(comp_samples)
        
        return {
            "exact_match": em,
            "f1": f1_avg,
            "bridge_em": bridge_em,
            "bridge_f1": bridge_f1,
            "comparison_em": comp_em,
            "comparison_f1": comp_f1
        }


async def run_hotpotqa_benchmark(sample_size: int = 100):
    """Run HotpotQA benchmark"""
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
    benchmark = HotpotQABenchmark(query_service)
    
    try:
        await benchmark.download_dataset()
        results = await benchmark.run(sample_size=sample_size)
        print(f"\n{'='*60}")
        print("HotpotQA Benchmark Results:")
        print(f"{'='*60}")
        print(benchmark.get_summary())
        return results
    finally:
        await db.disconnect()
        await redis_client.disconnect()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(run_hotpotqa_benchmark(sample_size=100))
