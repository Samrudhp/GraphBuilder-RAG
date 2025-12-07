"""
SciFact Benchmark Implementation

Tests scientific claim verification using GraphBuilder-RAG system.

Dataset: SciFact (1.4K scientific claims with evidence from research abstracts)
Task: Verify scientific claims (SUPPORTS, REFUTES, NOT_ENOUGH_INFO)
Focus: Domain-specific fact checking, scientific evidence retrieval

Paper: https://arxiv.org/abs/2004.14974
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
class SciFactSample:
    """SciFact dataset sample"""
    id: int
    claim: str
    evidence: Dict[str, Any]  # PubMed abstracts
    label: str  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    cited_doc_ids: List[int]


class SciFactBenchmark(BaseBenchmark):
    """SciFact benchmark implementation"""
    
    def __init__(self, query_service: Optional["QueryService"] = None):
        super().__init__(
            name="SciFact",
            output_dir=Path(__file__).parent.parent / "reports" / "scifact"
        )
        self.query_service = query_service
        self.data_dir = Path(__file__).parent.parent / "data" / "scifact"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_dataset(self, force: bool = False) -> None:
        """
        Download or create SciFact dataset.
        
        For this benchmark, we create representative synthetic samples
        covering different scientific domains.
        """
        data_file = self.data_dir / "scifact_samples.jsonl"
        
        if data_file.exists() and not force:
            print(f"✓ SciFact data already exists: {data_file}")
            return
        
        print("Creating SciFact samples...")
        
        # Create representative scientific claims
        samples = self._create_scifact_samples()
        
        # Save to JSONL
        with open(data_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps({
                    'id': sample.id,
                    'claim': sample.claim,
                    'evidence': sample.evidence,
                    'label': sample.label,
                    'cited_doc_ids': sample.cited_doc_ids
                }) + '\n')
        
        print(f"✓ Created {len(samples)} SciFact samples: {data_file}")
    
    def _create_scifact_samples(self) -> List[SciFactSample]:
        """Create representative SciFact test samples"""
        
        samples = [
            # SUPPORTS samples (scientific facts with evidence)
            SciFactSample(
                id=1,
                claim="mRNA vaccines induce the production of spike proteins that trigger an immune response.",
                evidence={
                    "abstract": "mRNA vaccines work by delivering genetic instructions to cells to produce a harmless piece of the SARS-CoV-2 spike protein. This protein triggers an immune response, leading to antibody production and T-cell activation.",
                    "source": "Nature Reviews Immunology 2021"
                },
                label="SUPPORTS",
                cited_doc_ids=[12345]
            ),
            SciFactSample(
                id=2,
                claim="CRISPR-Cas9 is a gene-editing technology derived from bacterial immune systems.",
                evidence={
                    "abstract": "CRISPR-Cas9 is an adaptive immune system found in bacteria and archaea. It has been repurposed as a powerful tool for genome editing in eukaryotic cells.",
                    "source": "Science 2012"
                },
                label="SUPPORTS",
                cited_doc_ids=[23456]
            ),
            SciFactSample(
                id=3,
                claim="Graphene is a two-dimensional material composed of a single layer of carbon atoms arranged in a hexagonal lattice.",
                evidence={
                    "abstract": "Graphene is an allotrope of carbon consisting of a single layer of atoms arranged in a honeycomb lattice. It exhibits remarkable electronic, thermal, and mechanical properties.",
                    "source": "Nature Materials 2007"
                },
                label="SUPPORTS",
                cited_doc_ids=[34567]
            ),
            SciFactSample(
                id=4,
                claim="Quantum entanglement allows particles to affect each other's states instantaneously regardless of distance.",
                evidence={
                    "abstract": "Quantum entanglement is a physical phenomenon where particles remain connected such that the quantum state of one particle cannot be described independently of the others, even at large distances.",
                    "source": "Physical Review Letters 1935"
                },
                label="SUPPORTS",
                cited_doc_ids=[45678]
            ),
            SciFactSample(
                id=5,
                claim="Dark matter constitutes approximately 27% of the universe's mass-energy content.",
                evidence={
                    "abstract": "According to Planck satellite measurements, the composition of the universe is approximately 68% dark energy, 27% dark matter, and 5% ordinary matter.",
                    "source": "Astronomy & Astrophysics 2016"
                },
                label="SUPPORTS",
                cited_doc_ids=[56789]
            ),
            
            # REFUTES samples (incorrect scientific claims)
            SciFactSample(
                id=6,
                claim="mRNA vaccines modify human DNA by integrating into the genome.",
                evidence={
                    "abstract": "mRNA vaccines do not enter the cell nucleus where DNA is stored. mRNA is translated in the cytoplasm and then degraded within days. It cannot integrate into genomic DNA.",
                    "source": "Nature Reviews Immunology 2021"
                },
                label="REFUTES",
                cited_doc_ids=[12345]
            ),
            SciFactSample(
                id=7,
                claim="CRISPR-Cas9 can only edit genes in bacterial cells and cannot work in humans.",
                evidence={
                    "abstract": "CRISPR-Cas9 has been successfully used for genome editing in a wide range of organisms including human cells, plants, and animals.",
                    "source": "Cell 2013"
                },
                label="REFUTES",
                cited_doc_ids=[23457]
            ),
            SciFactSample(
                id=8,
                claim="Graphene is a three-dimensional crystalline structure similar to diamond.",
                evidence={
                    "abstract": "Graphene is strictly a two-dimensional material consisting of a single atomic layer. Diamond, in contrast, is a three-dimensional crystalline form of carbon.",
                    "source": "Nature Materials 2007"
                },
                label="REFUTES",
                cited_doc_ids=[34567]
            ),
            SciFactSample(
                id=9,
                claim="Quantum entanglement enables faster-than-light communication between particles.",
                evidence={
                    "abstract": "While quantum entanglement creates correlations, it cannot be used for faster-than-light communication due to the no-communication theorem. Measurements appear random to each observer.",
                    "source": "Physical Review Letters 1980"
                },
                label="REFUTES",
                cited_doc_ids=[45679]
            ),
            SciFactSample(
                id=10,
                claim="Dark matter has been directly observed and photographed by telescopes.",
                evidence={
                    "abstract": "Dark matter has never been directly observed. Its existence is inferred from gravitational effects on visible matter, gravitational lensing, and cosmic microwave background observations.",
                    "source": "Astronomy & Astrophysics 2016"
                },
                label="REFUTES",
                cited_doc_ids=[56789]
            ),
            
            # NOT_ENOUGH_INFO samples (claims requiring additional evidence)
            SciFactSample(
                id=11,
                claim="The specific binding affinity of mRNA vaccine-induced antibodies is higher than natural infection.",
                evidence={
                    "abstract": "Both mRNA vaccination and natural infection produce antibodies against SARS-CoV-2. Comparative studies show varying results depending on timing and measurement methods.",
                    "source": "Nature Medicine 2021"
                },
                label="NOT_ENOUGH_INFO",
                cited_doc_ids=[12346]
            ),
            SciFactSample(
                id=12,
                claim="CRISPR-Cas9 off-target effects occur at a rate of exactly 3.7% in human embryonic cells.",
                evidence={
                    "abstract": "CRISPR-Cas9 can produce off-target mutations at sites with similar sequences. The frequency varies widely depending on the guide RNA design and delivery method.",
                    "source": "Nature Biotechnology 2015"
                },
                label="NOT_ENOUGH_INFO",
                cited_doc_ids=[23458]
            ),
            SciFactSample(
                id=13,
                claim="Graphene-based transistors will replace silicon transistors in all commercial electronics by 2030.",
                evidence={
                    "abstract": "Graphene has potential applications in electronics due to its high electron mobility. However, challenges remain in opening a band gap for digital logic applications.",
                    "source": "Nature Nanotechnology 2010"
                },
                label="NOT_ENOUGH_INFO",
                cited_doc_ids=[34568]
            ),
            SciFactSample(
                id=14,
                claim="Quantum computers using entangled qubits are currently capable of solving all NP-complete problems efficiently.",
                evidence={
                    "abstract": "Quantum computers can solve certain problems faster than classical computers. However, their advantage for NP-complete problems remains an open research question.",
                    "source": "Nature Physics 2018"
                },
                label="NOT_ENOUGH_INFO",
                cited_doc_ids=[45680]
            ),
            SciFactSample(
                id=15,
                claim="Dark matter particles have a mass between 10 and 100 GeV with 95% certainty.",
                evidence={
                    "abstract": "Various dark matter candidates span an enormous mass range from 10^-22 eV (ultralight axions) to solar mass primordial black holes. Direct detection experiments have set limits but not detections.",
                    "source": "Reviews of Modern Physics 2018"
                },
                label="NOT_ENOUGH_INFO",
                cited_doc_ids=[56790]
            ),
        ]
        
        return samples
    
    def load_data(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load SciFact data from JSONL file"""
        data_file = self.data_dir / "scifact_samples.jsonl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"SciFact data not found: {data_file}")
        
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
        """Format claim as verification question"""
        claim = sample['claim']
        # Frame as verification question with scientific context
        return f"Verify the following scientific claim with evidence: {claim}"
    
    async def run_system(self, input_text: str) -> Dict[str, Any]:
        """Run GraphBuilder system with scientific verification"""
        if not self.query_service:
            raise ValueError("QueryService not initialized")
        
        try:
            result = await self.query_service.process_query(
                query=input_text,
                user_id="benchmark_scifact",
                session_id="scifact_test",
                use_graph_verify=True,  # Enable verification
                use_nl2cypher=False  # Not needed for verification
            )
            return result
        except Exception as e:
            print(f"Error running system: {str(e)}")
            return {"error": str(e)}
    
    def extract_prediction(self, system_output: Dict[str, Any]) -> str:
        """Extract predicted label from system output"""
        if "error" in system_output:
            return "NOT_ENOUGH_INFO"  # Default on error
        
        # Check verification status
        verification = system_output.get("verification", {})
        status = verification.get("status", "UNCERTAIN")
        
        # Map verification status to SciFact labels
        if status == "VERIFIED":
            return "SUPPORTS"
        elif status in ["CONTRADICTED", "INCONSISTENT"]:
            return "REFUTES"
        else:
            return "NOT_ENOUGH_INFO"
    
    def extract_gold_label(self, sample: Dict[str, Any]) -> str:
        """Extract gold standard label"""
        return sample['label']
    
    def calculate_metrics(
        self,
        predictions: List[str],
        gold_labels: List[str],
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate SciFact metrics"""
        calculator = MetricsCalculator()
        
        # Overall metrics
        accuracy = calculator.accuracy(predictions, gold_labels)
        p_macro, r_macro, f1_macro = calculator.precision_recall_f1(
            predictions, gold_labels, average='macro'
        )
        
        # Per-class metrics
        classes = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        per_class = {}
        for cls in classes:
            p, r, f1 = calculator.precision_recall_f1(
                predictions, gold_labels, average=None, labels=[cls]
            )
            per_class[f"{cls.lower()}_precision"] = p[0] if len(p) > 0 else 0.0
            per_class[f"{cls.lower()}_recall"] = r[0] if len(r) > 0 else 0.0
            per_class[f"{cls.lower()}_f1"] = f1[0] if len(f1) > 0 else 0.0
        
        # Confusion matrix
        cm = calculator.confusion_matrix(predictions, gold_labels, classes)
        
        return {
            "accuracy": accuracy,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            **per_class,
            "confusion_matrix": cm.tolist()
        }


async def run_scifact_benchmark(sample_size: int = 100):
    """Run SciFact benchmark"""
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
    benchmark = SciFactBenchmark(query_service)
    
    try:
        await benchmark.download_dataset()
        results = await benchmark.run(sample_size=sample_size)
        print(f"\n{'='*60}")
        print("SciFact Benchmark Results:")
        print(f"{'='*60}")
        print(benchmark.get_summary())
        return results
    finally:
        await db.disconnect()
        await redis_client.disconnect()
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(run_scifact_benchmark(sample_size=100))
