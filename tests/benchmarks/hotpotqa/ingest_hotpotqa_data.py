"""
Direct HotpotQA Dataset Ingestion
Inserts all required entities for HotpotQA benchmark
"""
import asyncio
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.ingestion.service import IngestionService
from shared.models.schemas import DocumentType


HOTPOTQA_DOCUMENTS = [
    {
        "title": "Galileo Galilei - Father of Modern Science",
        "content": """Galileo Galilei was born on February 15, 1564, in Pisa, Italy. He was an Italian astronomer, physicist, and engineer who played a major role in the scientific revolution.

Galileo improved the telescope and made several important astronomical discoveries. He observed the moons of Jupiter, the phases of Venus, and sunspots. His support of heliocentrism brought him into conflict with the Catholic Church.

Galileo made fundamental contributions to the sciences of motion, astronomy, and strength of materials. He died on January 8, 1642, in Arcetri, near Florence, Italy.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "science", "person": "Galileo Galilei"}
    },
    {
        "title": "Apple Inc. - Technology Company",
        "content": """Apple Inc. is an American multinational technology company founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company is headquartered in Cupertino, California.

Apple designs, develops, and sells consumer electronics, computer software, and online services. Its hardware products include the iPhone, iPad, Mac computers, Apple Watch, and AirPods.

Apple is one of the world's most valuable companies and a leader in innovation. Steve Jobs served as CEO and was known for his visionary leadership until his death in 2011.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "technology", "company": "Apple"}
    },
    {
        "title": "Microsoft Corporation - Software Giant",
        "content": """Microsoft Corporation is an American multinational technology company founded on April 4, 1975, by Bill Gates and Paul Allen. The company is headquartered in Redmond, Washington.

Microsoft develops, manufactures, licenses, supports, and sells computer software, consumer electronics, and personal computers. Its best-known software products are the Microsoft Windows operating system and the Microsoft Office suite.

Bill Gates served as CEO and chairman of Microsoft. The company is one of the Big Five American information technology companies. Microsoft has been a pioneer in personal computing and enterprise software.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "technology", "company": "Microsoft"}
    },
    {
        "title": "Eiffel Tower - Paris Landmark",
        "content": """The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was designed by engineer Gustave Eiffel and completed in 1889. The tower stands 330 meters (1,083 feet) tall.

The Eiffel Tower is located on the Champ de Mars in Paris. Paris is the capital and largest city of France. The tower was built as the entrance arch for the 1889 World's Fair.

The Eiffel Tower is one of the most recognizable structures in the world and has become a global cultural icon of France. It receives millions of visitors each year.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "geography", "landmark": "Eiffel Tower"}
    },
    {
        "title": "Mount Everest - World's Highest Peak",
        "content": """Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The peak stands at 8,849 meters (29,032 feet) above sea level.

Mount Everest is located on the border between Nepal and Tibet. The mountain was named after Sir George Everest, a British surveyor. The first successful ascent was made by Edmund Hillary and Tenzing Norgay on May 29, 1953.

Mount Everest attracts climbers from around the world, though it presents significant dangers including altitude sickness, avalanches, and extreme weather conditions.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "geography", "mountain": "Mount Everest"}
    },
    {
        "title": "K2 - Second Highest Mountain",
        "content": """K2, also known as Mount Godwin-Austen, is the second-highest mountain on Earth at 8,611 meters (28,251 feet) above sea level. It is located in the Karakoram Range on the border between Pakistan and China.

K2 is known as one of the most difficult and dangerous mountains to climb. The mountain has a much higher fatality rate than Mount Everest. The first successful ascent was achieved by an Italian expedition in 1954.

K2 is often called the "Savage Mountain" due to the extreme difficulty of ascent and the second-highest fatality rate among the eight-thousanders for those who climb it.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "geography", "mountain": "K2"}
    },
    {
        "title": "Jupiter - Largest Planet",
        "content": """Jupiter is the largest planet in our Solar System. It is a gas giant with a diameter of 139,820 kilometers (86,881 miles). Jupiter is the fifth planet from the Sun.

Jupiter has at least 79 known moons, including the four large Galilean moons discovered by Galileo Galilei in 1610: Io, Europa, Ganymede, and Callisto. The planet is known for its Great Red Spot, a giant storm.

Jupiter has a mass more than twice that of all the other planets combined. The planet completes a rotation on its axis every 10 hours, making it the fastest rotating planet in the Solar System.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "astronomy", "planet": "Jupiter"}
    },
    {
        "title": "Saturn - Ringed Planet",
        "content": """Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. Saturn has a diameter of 116,460 kilometers (72,366 miles).

Saturn is a gas giant known for its prominent ring system, which is made up of ice and rock particles. The planet has at least 83 known moons, with Titan being the largest.

Saturn is named after the Roman god of agriculture. The planet is visible to the naked eye and has been known since prehistoric times. Its rings were first observed by Galileo Galilei in 1610.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "astronomy", "planet": "Saturn"}
    },
    {
        "title": "Atlantic Ocean - Second Largest Ocean",
        "content": """The Atlantic Ocean is the second-largest of the world's oceans, covering approximately 20% of Earth's surface. The Atlantic Ocean has an average depth of 3,646 meters (11,962 feet).

The Atlantic separates the Americas from Europe and Africa. The ocean's name derives from Greek mythology, referring to Atlas, the Titan who held up the sky.

The Atlantic Ocean is an important route for maritime trade and transportation. Major ports on the Atlantic include New York, London, Hamburg, and Rio de Janeiro.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "geography", "ocean": "Atlantic Ocean"}
    },
    {
        "title": "Pacific Ocean - Largest and Deepest Ocean",
        "content": """The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south. The Pacific Ocean has an average depth of 4,280 meters (14,040 feet).

The Pacific Ocean covers more than 30% of Earth's surface. The ocean's name comes from the Latin name "Mare Pacificum," meaning "peaceful sea," given by Portuguese explorer Ferdinand Magellan.

The deepest point in the Pacific Ocean is the Mariana Trench, which reaches a depth of about 11,034 meters (36,201 feet). The Pacific is larger than all of Earth's land area combined.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "geography", "ocean": "Pacific Ocean"}
    },
    {
        "title": "University of Bern - Swiss Institution",
        "content": """The University of Bern is a public research university in Bern, Switzerland. It was founded in 1834, making it one of Switzerland's oldest universities.

Albert Einstein worked at the University of Bern as a professor. The university is known for its research in various fields including physics, medicine, and humanities.

The University of Bern has produced several Nobel Prize winners and is consistently ranked among the top universities in Switzerland and Europe.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "education", "institution": "University of Bern"}
    },
    {
        "title": "Guido van Rossum - Creator of Python",
        "content": """Guido van Rossum is a Dutch programmer born on January 31, 1956. He is best known as the creator of the Python programming language.

Van Rossum developed Python in the late 1980s and released the first version in 1991. Python has become one of the most popular programming languages in the world, used extensively in web development, data science, and artificial intelligence.

Guido van Rossum served as Python's "Benevolent Dictator For Life" (BDFL) until 2018. He has worked at several major technology companies including Google and Dropbox.""",
        "source": "hotpotqa_benchmark_dataset",
        "metadata": {"category": "technology", "person": "Guido van Rossum"}
    },
]


async def ingest_hotpotqa_data():
    """Directly ingest HotpotQA documents"""
    print("="*80)
    print("HotpotQA Dataset Direct Ingestion")
    print("="*80)
    print(f"Ingesting {len(HOTPOTQA_DOCUMENTS)} documents...\n")
    
    # Create temp directory and files
    temp_dir = Path(__file__).parent / "temp_hotpotqa"
    temp_dir.mkdir(exist_ok=True)
    
    service = IngestionService()
    
    for i, doc in enumerate(HOTPOTQA_DOCUMENTS, 1):
        print(f"[{i}/{len(HOTPOTQA_DOCUMENTS)}] Ingesting: {doc['title']}")
        
        # Write to temp file
        temp_file = temp_dir / f"hotpotqa_doc_{i}.txt"
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
    asyncio.run(ingest_hotpotqa_data())
