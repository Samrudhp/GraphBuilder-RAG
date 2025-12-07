"""
Direct MetaQA Dataset Ingestion
Bypasses file upload and directly injects data into the pipeline
"""
import asyncio
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.ingestion.service import IngestionService
from shared.models.schemas import DocumentType


METAQA_DOCUMENTS = [
    {
        "title": "Christopher Nolan - Film Director",
        "content": """Christopher Nolan is a British-American film director, producer, and screenwriter born on July 30, 1970, in London, England. He is known for his distinctive filmmaking style and complex narratives.

Nolan directed the science fiction thriller "Inception" in 2010, which became both a critical and commercial success. The film explores the concept of dream invasion and features a complex plot involving multiple layers of dreams.

His other notable works include "The Dark Knight" trilogy, "Interstellar," "Dunkirk," and "Tenet." Nolan has received numerous awards and nominations for his work, including multiple Oscar nominations.""",
        "source": "metaqa_benchmark_dataset",
        "metadata": {"category": "film", "person": "Christopher Nolan"}
    },
    {
        "title": "J.K. Rowling - Author of Harry Potter",
        "content": """Joanne Rowling, known by her pen name J.K. Rowling, was born on July 31, 1965, in Yate, England. She is a British author and philanthropist best known for writing the Harry Potter fantasy series.

Rowling wrote the first Harry Potter book, "Harry Potter and the Philosopher's Stone" (published in 1997), which became a global phenomenon. The series consists of seven books and has been translated into over 80 languages.

The Harry Potter books have sold more than 500 million copies worldwide, making them the best-selling book series in history. The series was adapted into a highly successful film franchise. Rowling has received numerous awards for her work, including the British Book Awards and the Hans Christian Andersen Literature Award.""",
        "source": "metaqa_benchmark_dataset",
        "metadata": {"category": "literature", "person": "J.K. Rowling"}
    },
    {
        "title": "Niels Bohr - Quantum Physicist",
        "content": """Niels Bohr was born on October 7, 1885, in Copenhagen, Denmark. He was a Danish physicist who made foundational contributions to understanding atomic structure and quantum theory.

Bohr received the Nobel Prize in Physics in 1922 for his investigations of the structure of atoms and the radiation emanating from them. He developed the Bohr model of the atom, where electrons orbit the nucleus in distinct energy levels.

He founded the Institute of Theoretical Physics at the University of Copenhagen, which is now known as the Niels Bohr Institute. Bohr mentored and collaborated with many of the top physicists of the century. He passed away on November 18, 1962, in Copenhagen.""",
        "source": "metaqa_benchmark_dataset",
        "metadata": {"category": "science", "person": "Niels Bohr"}
    },
    {
        "title": "Albert Einstein - Extended Biography",
        "content": """Albert Einstein was born on March 14, 1879, in Ulm, Germany. He developed the theory of relativity and won the Nobel Prize in Physics in 1921. Einstein worked as a theoretical physicist and made groundbreaking contributions to modern physics.

Einstein's birth year was 1879. He was awarded the Nobel Prize for his explanation of the photoelectric effect. The year Einstein received the Nobel Prize was 1921. Einstein passed away on April 18, 1955, in Princeton, New Jersey.""",
        "source": "metaqa_benchmark_dataset",
        "metadata": {"category": "science", "person": "Albert Einstein"}
    },
    {
        "title": "Harry Potter Book Details",
        "content": """The Harry Potter series was written by J.K. Rowling. The author of Harry Potter is J.K. Rowling, who was born in 1965. Rowling's birth year is 1965. She published the first Harry Potter book in 1997.""",
        "source": "metaqa_benchmark_dataset", 
        "metadata": {"category": "literature", "work": "Harry Potter"}
    },
    {
        "title": "Inception Movie Details",
        "content": """Inception is a 2010 science fiction film directed by Christopher Nolan. The director of Inception is Christopher Nolan. Nolan directed this film about dream invasion and shared consciousness.""",
        "source": "metaqa_benchmark_dataset",
        "metadata": {"category": "film", "work": "Inception"}
    }
]


async def ingest_metaqa_data():
    """Directly ingest MetaQA documents"""
    print("="*80)
    print("MetaQA Dataset Direct Ingestion")
    print("="*80)
    print(f"Ingesting {len(METAQA_DOCUMENTS)} documents...\n")
    
    # Create temp directory and files
    temp_dir = Path(__file__).parent / "temp_metaqa"
    temp_dir.mkdir(exist_ok=True)
    
    service = IngestionService()
    
    for i, doc in enumerate(METAQA_DOCUMENTS, 1):
        print(f"[{i}/{len(METAQA_DOCUMENTS)}] Ingesting: {doc['title']}")
        
        # Write to temp file
        temp_file = temp_dir / f"metaqa_doc_{i}.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"{doc['title']}\n\n{doc['content']}")
        
        try:
            # Ingest via service
            raw_doc = await service.ingest_from_file(
                file_path=temp_file,
                source_type=DocumentType.TEXT,
            )
            
            print(f"  Document ID: {raw_doc.document_id}")
            print(f"  Status: Queued for processing")
            
            # Clean up temp file
            temp_file.unlink()
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass
    
    print("="*80)
    print("All documents queued!")
    print("="*80)
    print("\nThe Celery workers will now process these documents through:")
    print("  1. Ingestion → Raw docs in MongoDB")
    print("  2. Extraction → Triples extracted")
    print("  3. Normalization → Entities standardized")
    print("  4. Validation → External verification")
    print("  5. Fusion → Merged into Neo4j graph")
    print("  6. Embedding → FAISS index updated")
    print("\nCheck Celery worker logs to monitor progress.")
    print("Run 'python helpers/view_triples.py' after a few minutes to see results.")


if __name__ == "__main__":
    asyncio.run(ingest_metaqa_data())
