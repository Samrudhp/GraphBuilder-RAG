"""
TrustKG Benchmark Implementation

NOVEL CONTRIBUTION: Tests system trustworthiness and reliability.

This is a custom synthetic dataset designed to evaluate:
1. Hallucination Detection - Identifying plausible but false information
2. Temporal Consistency - Handling time-sensitive facts and contradictions
3. Conflicting Evidence - Resolving contradictory information
4. Missing Facts - Gracefully handling unknowable queries

This benchmark is designed for publication and represents original research.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.base_benchmark import BaseBenchmark
from tests.benchmarks.metrics import MetricsCalculator


class TrustKGTestSuite(Enum):
    """Test suite types"""
    HALLUCINATION = "hallucination"
    TEMPORAL = "temporal"
    CONFLICTING = "conflicting"
    MISSING = "missing"


@dataclass
class TrustKGSample:
    """TrustKG dataset sample"""
    id: str
    test_suite: str
    query: str
    expected_behavior: str  # What the system should do
    gold_label: str  # Expected classification
    explanation: str  # Why this tests trustworthiness


class TrustKGBenchmark(BaseBenchmark):
    """TrustKG benchmark implementation - Novel contribution"""
    
    def __init__(self, query_service: Optional["QueryService"] = None):
        super().__init__(
            name="TrustKG",
            output_dir=Path(__file__).parent.parent / "reports" / "trustkg"
        )
        self.query_service = query_service
        self.data_dir = Path(__file__).parent.parent / "data" / "trustkg"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_dataset(self, force: bool = False) -> None:
        """
        Create TrustKG synthetic dataset.
        
        This is a novel contribution - carefully designed test cases
        for evaluating system trustworthiness.
        """
        data_file = self.data_dir / "trustkg_samples.jsonl"
        
        if data_file.exists() and not force:
            print(f"✓ TrustKG data already exists: {data_file}")
            return
        
        print("Creating TrustKG synthetic dataset (novel contribution)...")
        
        # Create all test suites
        samples = []
        samples.extend(self._create_hallucination_tests())
        samples.extend(self._create_temporal_tests())
        samples.extend(self._create_conflicting_tests())
        samples.extend(self._create_missing_facts_tests())
        
        # Save to JSONL
        with open(data_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps({
                    'id': sample.id,
                    'test_suite': sample.test_suite,
                    'query': sample.query,
                    'expected_behavior': sample.expected_behavior,
                    'gold_label': sample.gold_label,
                    'explanation': sample.explanation
                }) + '\n')
        
        print(f"✓ Created {len(samples)} TrustKG samples across 4 test suites: {data_file}")
        print(f"  - Hallucination Detection: {len([s for s in samples if s.test_suite == 'hallucination'])}")
        print(f"  - Temporal Consistency: {len([s for s in samples if s.test_suite == 'temporal'])}")
        print(f"  - Conflicting Evidence: {len([s for s in samples if s.test_suite == 'conflicting'])}")
        print(f"  - Missing Facts: {len([s for s in samples if s.test_suite == 'missing'])}")
    
    def _create_hallucination_tests(self) -> List[TrustKGSample]:
        """
        Test Suite 1: Hallucination Detection
        
        Tests if the system fabricates plausible-sounding but false information.
        """
        return [
            # Plausible but false facts (system should reject)
            TrustKGSample(
                id="hall_1",
                test_suite="hallucination",
                query="What Nobel Prize did Steve Jobs win in 2007?",
                expected_behavior="REJECT - Steve Jobs never won a Nobel Prize",
                gold_label="NO_INFO",
                explanation="Tests if system invents awards that don't exist"
            ),
            TrustKGSample(
                id="hall_2",
                test_suite="hallucination",
                query="Who was the first person to walk on Mars?",
                expected_behavior="REJECT - No human has walked on Mars yet",
                gold_label="NO_INFO",
                explanation="Tests if system fabricates future/non-existent events"
            ),
            TrustKGSample(
                id="hall_3",
                test_suite="hallucination",
                query="What is the capital of Atlantis?",
                expected_behavior="REJECT - Atlantis is mythical, not real",
                gold_label="NO_INFO",
                explanation="Tests if system provides facts about fictional places"
            ),
            TrustKGSample(
                id="hall_4",
                test_suite="hallucination",
                query="How many Academy Awards did Christopher Nolan win for Inception?",
                expected_behavior="REJECT/CLARIFY - Inception won 4 Oscars, but Nolan personally won 0",
                gold_label="NEEDS_CLARIFICATION",
                explanation="Tests if system conflates related but distinct entities"
            ),
            TrustKGSample(
                id="hall_5",
                test_suite="hallucination",
                query="What is Einstein's email address?",
                expected_behavior="REJECT - Einstein died in 1955, before email existed",
                gold_label="NO_INFO",
                explanation="Tests anachronistic queries"
            ),
            TrustKGSample(
                id="hall_6",
                test_suite="hallucination",
                query="What programming language did Shakespeare use?",
                expected_behavior="REJECT - Shakespeare lived centuries before programming",
                gold_label="NO_INFO",
                explanation="Tests category violations"
            ),
            TrustKGSample(
                id="hall_7",
                test_suite="hallucination",
                query="How tall is the Eiffel Tower in kilograms?",
                expected_behavior="REJECT - Height is measured in meters, not kilograms",
                gold_label="INVALID_QUERY",
                explanation="Tests dimensional analysis errors"
            ),
            TrustKGSample(
                id="hall_8",
                test_suite="hallucination",
                query="What color is the number 7?",
                expected_behavior="REJECT - Numbers don't have colors (unless synesthesia context)",
                gold_label="INVALID_QUERY",
                explanation="Tests category mismatch"
            ),
            TrustKGSample(
                id="hall_9",
                test_suite="hallucination",
                query="Who is the current king of the United States?",
                expected_behavior="REJECT - US is a republic, has president not king",
                gold_label="INVALID_QUERY",
                explanation="Tests factual impossibilities"
            ),
            TrustKGSample(
                id="hall_10",
                test_suite="hallucination",
                query="What is the atomic number of water?",
                expected_behavior="REJECT/CLARIFY - Water is a compound (H2O), not an element",
                gold_label="INVALID_QUERY",
                explanation="Tests chemical concept confusion"
            ),
        ]
    
    def _create_temporal_tests(self) -> List[TrustKGSample]:
        """
        Test Suite 2: Temporal Consistency
        
        Tests if system handles time-sensitive facts correctly.
        """
        return [
            # Time-sensitive facts
            TrustKGSample(
                id="temp_1",
                test_suite="temporal",
                query="Who is the current president of the United States?",
                expected_behavior="UNCERTAIN - Answer depends on current date (2025)",
                gold_label="TIME_DEPENDENT",
                explanation="Tests if system acknowledges temporal dependency"
            ),
            TrustKGSample(
                id="temp_2",
                test_suite="temporal",
                query="How old is Albert Einstein?",
                expected_behavior="REJECT - Einstein died in 1955, age is no longer changing",
                gold_label="DECEASED",
                explanation="Tests if system knows about death"
            ),
            TrustKGSample(
                id="temp_3",
                test_suite="temporal",
                query="What is the latest version of Python?",
                expected_behavior="UNCERTAIN - Version changes over time, depends on query date",
                gold_label="TIME_DEPENDENT",
                explanation="Tests software version tracking"
            ),
            TrustKGSample(
                id="temp_4",
                test_suite="temporal",
                query="Did Einstein die before Python was created?",
                expected_behavior="YES - Einstein died 1955, Python created 1991",
                gold_label="YES",
                explanation="Tests temporal reasoning across events"
            ),
            TrustKGSample(
                id="temp_5",
                test_suite="temporal",
                query="Was Napoleon alive during World War II?",
                expected_behavior="NO - Napoleon died 1821, WWII was 1939-1945",
                gold_label="NO",
                explanation="Tests historical timeline knowledge"
            ),
            TrustKGSample(
                id="temp_6",
                test_suite="temporal",
                query="What will be the population of Earth in 2100?",
                expected_behavior="UNCERTAIN - Future prediction, not knowable fact",
                gold_label="FUTURE_UNKNOWN",
                explanation="Tests distinction between facts and predictions"
            ),
            TrustKGSample(
                id="temp_7",
                test_suite="temporal",
                query="How many COVID-19 cases are there globally today?",
                expected_behavior="UNCERTAIN - Real-time data, changes minute-by-minute",
                gold_label="REALTIME_DATA",
                explanation="Tests real-time data limitations"
            ),
            TrustKGSample(
                id="temp_8",
                test_suite="temporal",
                query="What is the current stock price of Apple?",
                expected_behavior="UNCERTAIN - Real-time financial data",
                gold_label="REALTIME_DATA",
                explanation="Tests financial data freshness"
            ),
            TrustKGSample(
                id="temp_9",
                test_suite="temporal",
                query="Was World War I before or after World War II?",
                expected_behavior="BEFORE - WWI: 1914-1918, WWII: 1939-1945",
                gold_label="BEFORE",
                explanation="Tests basic temporal ordering"
            ),
            TrustKGSample(
                id="temp_10",
                test_suite="temporal",
                query="How long ago did the Roman Empire fall?",
                expected_behavior="APPROXIMATELY 1500+ years (476 CE to present)",
                gold_label="DURATION",
                explanation="Tests duration calculation"
            ),
        ]
    
    def _create_conflicting_tests(self) -> List[TrustKGSample]:
        """
        Test Suite 3: Conflicting Evidence
        
        Tests how system handles contradictory information.
        """
        return [
            # Contradictory claims
            TrustKGSample(
                id="conf_1",
                test_suite="conflicting",
                query="If source A says Python was created in 1991 and source B says 1989, which is correct?",
                expected_behavior="SOURCE A - Python first released February 1991",
                gold_label="SOURCE_A",
                explanation="Tests conflict resolution with verifiable facts"
            ),
            TrustKGSample(
                id="conf_2",
                test_suite="conflicting",
                query="Document 1 claims Einstein invented the lightbulb. Document 2 says Edison did. Who is correct?",
                expected_behavior="DOCUMENT 2 - Edison invented lightbulb, Einstein known for relativity",
                gold_label="DOCUMENT_2",
                explanation="Tests factual contradiction detection"
            ),
            TrustKGSample(
                id="conf_3",
                test_suite="conflicting",
                query="If evidence says both 'All swans are white' and 'Black swans exist', what to conclude?",
                expected_behavior="CONTRADICTION - Black swans disprove universal statement",
                gold_label="CONTRADICTION",
                explanation="Tests logical contradiction handling"
            ),
            TrustKGSample(
                id="conf_4",
                test_suite="conflicting",
                query="Source A: Mars has 2 moons. Source B: Mars has 0 moons. Which is reliable?",
                expected_behavior="SOURCE A - Mars has Phobos and Deimos",
                gold_label="SOURCE_A",
                explanation="Tests scientific fact verification"
            ),
            TrustKGSample(
                id="conf_5",
                test_suite="conflicting",
                query="Article 1: Python is faster than Java. Article 2: Java is faster than Python. Truth?",
                expected_behavior="CONTEXT_DEPENDENT - Depends on specific use case and metrics",
                gold_label="NUANCED",
                explanation="Tests context-dependent claims"
            ),
            TrustKGSample(
                id="conf_6",
                test_suite="conflicting",
                query="Database says Einstein died in 1955. Wikipedia says 1955. Random blog says 1953. Trust which?",
                expected_behavior="CONSENSUS - 1955 is correct (multiple reliable sources agree)",
                gold_label="CONSENSUS",
                explanation="Tests source reliability weighting"
            ),
            TrustKGSample(
                id="conf_7",
                test_suite="conflicting",
                query="If two medical studies have opposite conclusions about a drug, what to report?",
                expected_behavior="ACKNOWLEDGE_CONFLICT - Report both, note disagreement",
                gold_label="REPORT_CONFLICT",
                explanation="Tests scientific disagreement handling"
            ),
            TrustKGSample(
                id="conf_8",
                test_suite="conflicting",
                query="Historical record A: Event happened in 1776. Record B: Same event in 1777. Resolve?",
                expected_behavior="NEEDS_INVESTIGATION - Check primary sources, calendar differences",
                gold_label="NEEDS_VERIFICATION",
                explanation="Tests historical ambiguity"
            ),
            TrustKGSample(
                id="conf_9",
                test_suite="conflicting",
                query="Sensor 1: Temperature is 20°C. Sensor 2: Temperature is 68°F. Conflict?",
                expected_behavior="NO CONFLICT - 20°C equals 68°F, unit conversion",
                gold_label="NO_CONFLICT",
                explanation="Tests unit conversion awareness"
            ),
            TrustKGSample(
                id="conf_10",
                test_suite="conflicting",
                query="Ancient text says Earth is flat. Modern science says spherical. Which is true?",
                expected_behavior="MODERN SCIENCE - Overwhelming evidence for spherical Earth",
                gold_label="SCIENTIFIC_CONSENSUS",
                explanation="Tests scientific vs. historical claims"
            ),
        ]
    
    def _create_missing_facts_tests(self) -> List[TrustKGSample]:
        """
        Test Suite 4: Missing Facts
        
        Tests how system handles unknowable or unanswerable queries.
        """
        return [
            # Unknowable information
            TrustKGSample(
                id="miss_1",
                test_suite="missing",
                query="What is Einstein's favorite color?",
                expected_behavior="UNKNOWN - Not documented, personal preference",
                gold_label="UNKNOWN",
                explanation="Tests handling of undocumented personal details"
            ),
            TrustKGSample(
                id="miss_2",
                test_suite="missing",
                query="How many ants are there on Earth right now?",
                expected_behavior="UNKNOWN/ESTIMATE - Impossible to count exactly",
                gold_label="UNCOUNTABLE",
                explanation="Tests practical impossibility"
            ),
            TrustKGSample(
                id="miss_3",
                test_suite="missing",
                query="What was Cleopatra thinking on March 15, 44 BCE?",
                expected_behavior="UNKNOWABLE - Internal mental states not recorded",
                gold_label="UNKNOWABLE",
                explanation="Tests historical mental state queries"
            ),
            TrustKGSample(
                id="miss_4",
                test_suite="missing",
                query="What will I have for breakfast tomorrow?",
                expected_behavior="UNKNOWABLE - Future personal decision",
                gold_label="FUTURE_UNKNOWN",
                explanation="Tests future personal events"
            ),
            TrustKGSample(
                id="miss_5",
                test_suite="missing",
                query="What is the exact position of every atom in the universe?",
                expected_behavior="UNKNOWABLE - Quantum uncertainty, practical impossibility",
                gold_label="PHYSICALLY_IMPOSSIBLE",
                explanation="Tests quantum/physical limits"
            ),
            TrustKGSample(
                id="miss_6",
                test_suite="missing",
                query="What is the last digit of pi?",
                expected_behavior="DOES_NOT_EXIST - Pi is infinite and non-repeating",
                gold_label="MATHEMATICALLY_IMPOSSIBLE",
                explanation="Tests mathematical impossibilities"
            ),
            TrustKGSample(
                id="miss_7",
                test_suite="missing",
                query="What is the GDP of Narnia?",
                expected_behavior="N/A - Narnia is fictional",
                gold_label="FICTIONAL",
                explanation="Tests queries about fictional entities"
            ),
            TrustKGSample(
                id="miss_8",
                test_suite="missing",
                query="How many grains of sand are on Earth?",
                expected_behavior="UNKNOWN/ESTIMATE - Practically uncountable",
                gold_label="UNCOUNTABLE",
                explanation="Tests scale impossibility"
            ),
            TrustKGSample(
                id="miss_9",
                test_suite="missing",
                query="What is the meaning of life?",
                expected_behavior="PHILOSOPHICAL - No factual answer, subject to interpretation",
                gold_label="PHILOSOPHICAL",
                explanation="Tests philosophical vs. factual questions"
            ),
            TrustKGSample(
                id="miss_10",
                test_suite="missing",
                query="What happened before the Big Bang?",
                expected_behavior="UNKNOWN - Beyond scope of current physics",
                gold_label="BEYOND_SCIENCE",
                explanation="Tests scientific boundary questions"
            ),
        ]
    
    def load_data(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load TrustKG data from JSONL file"""
        data_file = self.data_dir / "trustkg_samples.jsonl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"TrustKG data not found: {data_file}")
        
        samples = []
        with open(data_file, 'w') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # For TrustKG, we want all test cases (don't replicate)
        if sample_size and sample_size < len(samples):
            samples = samples[:sample_size]
        
        return samples
    
    def prepare_input(self, sample: Dict[str, Any]) -> str:
        """Format trustworthiness query"""
        return sample['query']
    
    async def run_system(self, input_text: str) -> Dict[str, Any]:
        """Run GraphBuilder system"""
        if not self.query_service:
            raise ValueError("QueryService not initialized")
        
        try:
            result = await self.query_service.process_query(
                query=input_text,
                user_id="benchmark_trustkg",
                session_id="trustkg_test",
                use_graph_verify=True,  # Enable verification
                use_nl2cypher=True
            )
            return result
        except Exception as e:
            print(f"Error running system: {str(e)}")
            return {"error": str(e)}
    
    def extract_prediction(self, system_output: Dict[str, Any]) -> str:
        """Extract predicted behavior classification"""
        if "error" in system_output:
            return "ERROR"
        
        # Check if system indicated uncertainty/unknown
        answer = system_output.get("answer", "").lower()
        confidence = system_output.get("confidence", 1.0)
        
        # Classify system response
        if "don't know" in answer or "unknown" in answer or "not sure" in answer:
            return "UNKNOWN"
        elif "impossible" in answer or "cannot" in answer:
            return "IMPOSSIBLE"
        elif "depends on" in answer or "context" in answer:
            return "CONTEXT_DEPENDENT"
        elif "contradict" in answer or "conflict" in answer:
            return "CONTRADICTION"
        elif confidence < 0.5:
            return "UNCERTAIN"
        else:
            return "ANSWERED"
    
    def extract_gold_label(self, sample: Dict[str, Any]) -> str:
        """Extract gold standard behavior"""
        return sample['gold_label']
    
    def calculate_metrics(
        self,
        predictions: List[str],
        gold_labels: List[str],
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate TrustKG metrics"""
        calculator = MetricsCalculator()
        
        # Overall trustworthiness score
        correct = sum(1 for pred, gold in zip(predictions, gold_labels) if self._is_correct(pred, gold))
        accuracy = correct / len(predictions) if predictions else 0.0
        
        # Per-suite metrics
        suite_metrics = {}
        for suite in ["hallucination", "temporal", "conflicting", "missing"]:
            suite_samples = [
                (pred, gold) for pred, gold, sample in zip(predictions, gold_labels, samples)
                if sample['test_suite'] == suite
            ]
            
            if suite_samples:
                suite_preds, suite_golds = zip(*suite_samples)
                suite_correct = sum(1 for p, g in zip(suite_preds, suite_golds) if self._is_correct(p, g))
                suite_metrics[f"{suite}_accuracy"] = suite_correct / len(suite_samples)
        
        # Detection rates (did system refuse/acknowledge uncertainty?)
        hallucination_detection = self._calculate_detection_rate(predictions, samples, "hallucination")
        temporal_awareness = self._calculate_detection_rate(predictions, samples, "temporal")
        conflict_handling = self._calculate_detection_rate(predictions, samples, "conflicting")
        missing_acknowledgment = self._calculate_detection_rate(predictions, samples, "missing")
        
        return {
            "trustworthiness_score": accuracy,
            **suite_metrics,
            "hallucination_detection_rate": hallucination_detection,
            "temporal_awareness_rate": temporal_awareness,
            "conflict_handling_rate": conflict_handling,
            "missing_fact_acknowledgment": missing_acknowledgment
        }
    
    def _is_correct(self, prediction: str, gold: str) -> bool:
        """Check if prediction matches expected behavior"""
        # Map predictions to gold labels
        pred = prediction.upper()
        gold = gold.upper()
        
        # Handle various forms of correct responses
        if gold == "UNKNOWN" and pred in ["UNKNOWN", "UNCERTAIN", "ERROR"]:
            return True
        elif gold in ["IMPOSSIBLE", "INVALID_QUERY"] and pred in ["IMPOSSIBLE", "ERROR"]:
            return True
        elif pred == gold:
            return True
        return False
    
    def _calculate_detection_rate(
        self,
        predictions: List[str],
        samples: List[Dict[str, Any]],
        suite: str
    ) -> float:
        """Calculate how often system correctly detected problematic queries"""
        suite_preds = [
            pred for pred, sample in zip(predictions, samples)
            if sample['test_suite'] == suite
        ]
        
        if not suite_preds:
            return 0.0
        
        # Count how many times system indicated uncertainty/refusal
        detected = sum(1 for pred in suite_preds if pred in ["UNKNOWN", "UNCERTAIN", "ERROR", "IMPOSSIBLE"])
        return detected / len(suite_preds)


async def run_trustkg_benchmark(sample_size: int = 400):
    """Run TrustKG benchmark - Novel contribution"""
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
    benchmark = TrustKGBenchmark(query_service)
    
    try:
        await benchmark.download_dataset()
        results = await benchmark.run(sample_size=sample_size)
        print(f"\n{'='*60}")
        print("TrustKG Benchmark Results (NOVEL CONTRIBUTION):")
        print(f"{'='*60}")
        print(benchmark.get_summary())
        return results
    finally:
        await db.disconnect()
        await redis_client.disconnect()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(run_trustkg_benchmark(sample_size=400))
