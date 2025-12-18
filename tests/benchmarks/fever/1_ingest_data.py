"""
FEVER Benchmark - Step 1: Data Ingestion (50 Documents)

Ingests 50 diverse documents covering multiple domains.
This provides enough data for FEVER-style fact verification without overwhelming the system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.ingestion.service import IngestionService
from shared.models.schemas import DocumentType


# 50 FEVER Documents (10 per domain)
FEVER_DOCUMENTS = [
    # SCIENTISTS (10 documents)
    {"title": "Albert Einstein - Theoretical Physicist", "content": "Albert Einstein was born in Ulm, Germany on March 14, 1879. He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. He developed the theory of relativity and the famous equation E=mc². He died on April 18, 1955 in Princeton, New Jersey."},
    {"title": "Marie Curie - Physicist and Chemist", "content": "Marie Curie was born Maria Skłodowska in Warsaw, Poland on November 7, 1867. She won two Nobel Prizes - in Physics (1903) and Chemistry (1911). She discovered the elements radium and polonium. She died on July 4, 1934."},
    {"title": "Isaac Newton - Mathematician and Physicist", "content": "Isaac Newton was born in Woolsthorpe, England on January 4, 1643. He published Philosophiæ Naturalis Principia Mathematica in 1687, laying the foundations for classical mechanics. He formulated the laws of motion and universal gravitation. He died on March 31, 1727."},
    {"title": "Galileo Galilei - Astronomer", "content": "Galileo Galilei was born in Pisa, Italy on February 15, 1564. He is known as the father of modern science and made fundamental contributions to astronomy and physics. He improved the telescope and made astronomical observations supporting heliocentrism. He died on January 8, 1642."},
    {"title": "Stephen Hawking - Cosmologist", "content": "Stephen Hawking was born in Oxford, England on January 8, 1942. He was a theoretical physicist and cosmologist known for his work on black holes and cosmology. He wrote A Brief History of Time published in 1988. He died on March 14, 2018."},
    {"title": "Charles Darwin - Naturalist", "content": "Charles Darwin was born in Shrewsbury, England on February 12, 1809. He published On the Origin of Species in 1859, establishing evolutionary biology. He proposed the theory of natural selection. He died on April 19, 1882."},
    {"title": "Nikola Tesla - Electrical Engineer", "content": "Nikola Tesla was born in Smiljan, Austrian Empire (modern Croatia) on July 10, 1856. He developed the alternating current (AC) electrical system. He held over 300 patents and made contributions to wireless communication. He died on January 7, 1943."},
    {"title": "Richard Feynman - Physicist", "content": "Richard Feynman was born in New York City on May 11, 1918. He won the Nobel Prize in Physics in 1965 for his work in quantum electrodynamics. He worked on the Manhattan Project and later became known for popularizing physics. He died on February 15, 1988."},
    {"title": "Rosalind Franklin - Chemist", "content": "Rosalind Franklin was born in London, England on July 25, 1920. She made crucial contributions to understanding DNA structure through X-ray crystallography. Her Photo 51 was critical evidence for the double helix structure. She died on April 16, 1958."},
    {"title": "Carl Sagan - Astronomer", "content": "Carl Sagan was born in Brooklyn, New York on November 9, 1934. He was an astronomer and science communicator who wrote Cosmos and popularized science. He contributed to space exploration including the Voyager missions. He died on December 20, 1996."},
    
    # TECHNOLOGY (10 documents)
    {"title": "Apple Inc. - Technology Company", "content": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976 in Cupertino, California. The company produces iPhone, iPad, Mac computers, and other consumer electronics. It became one of the world's most valuable companies."},
    {"title": "Microsoft Corporation", "content": "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 in Albuquerque, New Mexico. The company develops Windows operating system, Office suite, and Azure cloud services. It is headquartered in Redmond, Washington."},
    {"title": "Google - Search Engine Company", "content": "Google was founded by Larry Page and Sergey Brin in September 1998 while they were PhD students at Stanford University. It started as a search engine and expanded to include Android, YouTube, and cloud computing. The parent company Alphabet was created in 2015."},
    {"title": "Python Programming Language", "content": "Python was created by Guido van Rossum and first released in 1991. It is a high-level, interpreted programming language emphasizing code readability. Python has become one of the most popular programming languages for data science and web development."},
    {"title": "Linux Operating System", "content": "Linux was created by Linus Torvalds and first released on September 17, 1991. It is an open-source Unix-like operating system kernel. Linux powers most web servers, Android devices, and supercomputers worldwide."},
    {"title": "Amazon - E-commerce Company", "content": "Amazon was founded by Jeff Bezos on July 5, 1994 in Seattle, Washington. It started as an online bookstore and expanded to become the world's largest e-commerce platform. Amazon Web Services (AWS) is now a major cloud computing provider."},
    {"title": "Facebook - Social Media Platform", "content": "Facebook was founded by Mark Zuckerberg on February 4, 2004 while he was a student at Harvard University. It became the world's largest social networking service. The company rebranded to Meta Platforms in 2021."},
    {"title": "Tesla Inc. - Electric Vehicle Manufacturer", "content": "Tesla was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined as chairman and later became CEO. The company produces electric vehicles including Model S, Model 3, Model X, and Model Y."},
    {"title": "IBM - International Business Machines", "content": "IBM was founded as Computing-Tabulating-Recording Company in 1911 and renamed International Business Machines in 1924. It pioneered computer hardware, mainframes, and enterprise software. IBM is headquartered in Armonk, New York."},
    {"title": "Intel Corporation - Semiconductor Manufacturer", "content": "Intel was founded by Gordon Moore and Robert Noyce on July 18, 1968 in Santa Clara, California. The company is the world's largest semiconductor chip manufacturer. Intel processors power most personal computers worldwide."},
    
    # GEOGRAPHY (10 documents)
    {"title": "Mount Everest - Highest Mountain", "content": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its peak stands at 8,849 meters (29,032 feet). The mountain is located on the border between Nepal and Tibet."},
    {"title": "K2 - Second Highest Mountain", "content": "K2 is the second-highest mountain on Earth at 8,611 meters (28,251 feet) above sea level. It is located on the China-Pakistan border in the Karakoram Range. K2 is considered more challenging to climb than Everest."},
    {"title": "Pacific Ocean - Largest Ocean", "content": "The Pacific Ocean is the largest and deepest ocean on Earth, covering approximately 165 million square kilometers. Its average depth is 4,280 meters (14,040 feet). It extends from the Arctic in the north to Antarctica in the south."},
    {"title": "Atlantic Ocean - Second Largest Ocean", "content": "The Atlantic Ocean is the second-largest ocean, covering about 106 million square kilometers. Its average depth is approximately 3,646 meters (11,962 feet). It separates the Americas from Europe and Africa."},
    {"title": "Amazon Rainforest - Largest Tropical Rainforest", "content": "The Amazon Rainforest is the world's largest tropical rainforest, covering approximately 5.5 million square kilometers. It spans across nine countries in South America, with Brazil containing about 60% of it. The Amazon River flows through the rainforest."},
    {"title": "Sahara Desert - Largest Hot Desert", "content": "The Sahara is the world's largest hot desert, covering about 9 million square kilometers in North Africa. It spans across multiple countries including Egypt, Libya, Algeria, and Morocco. Temperatures can exceed 50°C (122°F) during the day."},
    {"title": "Great Barrier Reef - Largest Coral Reef", "content": "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia. It stretches over 2,300 kilometers and covers approximately 344,400 square kilometers. It is visible from space and is a UNESCO World Heritage Site."},
    {"title": "Nile River - Longest River", "content": "The Nile River is traditionally considered the longest river in the world at approximately 6,650 kilometers (4,130 miles). It flows through northeastern Africa, primarily through Egypt and Sudan. The river has two major tributaries: the White Nile and Blue Nile."},
    {"title": "Antarctica - Southernmost Continent", "content": "Antarctica is Earth's southernmost continent, containing the geographic South Pole. It is the fifth-largest continent, covering 14 million square kilometers. About 98% of Antarctica is covered by ice averaging 1.9 kilometers thick."},
    {"title": "Mount Kilimanjaro - Highest Mountain in Africa", "content": "Mount Kilimanjaro is the highest mountain in Africa at 5,895 meters (19,341 feet) above sea level. It is a dormant volcano located in Tanzania. Kilimanjaro has three volcanic cones: Kibo, Mawenzi, and Shira."},
    
    # ARTS & LITERATURE (10 documents)
    {"title": "William Shakespeare - Playwright", "content": "William Shakespeare was born in Stratford-upon-Avon, England in April 1564. He is widely regarded as the greatest writer in the English language and wrote approximately 39 plays and 154 sonnets. Famous works include Hamlet, Romeo and Juliet, and Macbeth. He died on April 23, 1616."},
    {"title": "J.K. Rowling - Author", "content": "J.K. Rowling was born Joanne Rowling on July 31, 1965 in Yate, England. She wrote the Harry Potter series, which became the best-selling book series in history. The first book, Harry Potter and the Philosopher's Stone, was published in 1997."},
    {"title": "Leonardo da Vinci - Renaissance Artist", "content": "Leonardo da Vinci was born on April 15, 1452 in Vinci, Italy. He was a polymath of the Renaissance era known for painting the Mona Lisa and The Last Supper. He was also a scientist, engineer, and inventor. He died on May 2, 1519."},
    {"title": "Pablo Picasso - Painter", "content": "Pablo Picasso was born on October 25, 1881 in Málaga, Spain. He was a painter and sculptor who co-founded the Cubist movement. His most famous work is Guernica, painted in 1937. He died on April 8, 1973."},
    {"title": "Vincent van Gogh - Post-Impressionist Painter", "content": "Vincent van Gogh was born on March 30, 1853 in the Netherlands. He created about 2,100 artworks including The Starry Night and Sunflowers. Despite his fame today, he sold only one painting during his lifetime. He died on July 29, 1890."},
    {"title": "Jane Austen - Novelist", "content": "Jane Austen was born on December 16, 1775 in Hampshire, England. She wrote six major novels including Pride and Prejudice, Sense and Sensibility, and Emma. Her works critique the British landed gentry of the late 18th century. She died on July 18, 1817."},
    {"title": "Ernest Hemingway - Author", "content": "Ernest Hemingway was born on July 21, 1899 in Oak Park, Illinois. He won the Nobel Prize in Literature in 1954. Famous works include The Old Man and the Sea, A Farewell to Arms, and For Whom the Bell Tolls. He died on July 2, 1961."},
    {"title": "Mark Twain - American Author", "content": "Mark Twain, born Samuel Clemens on November 30, 1835 in Missouri, was an American writer and humorist. He wrote The Adventures of Tom Sawyer and Adventures of Huckleberry Finn, often called the Great American Novel. He died on April 21, 1910."},
    {"title": "Charles Dickens - Victorian Novelist", "content": "Charles Dickens was born on February 7, 1812 in Portsmouth, England. He wrote famous novels including A Tale of Two Cities, Great Expectations, and Oliver Twist. His works criticized Victorian society and poverty. He died on June 9, 1870."},
    {"title": "Agatha Christie - Mystery Writer", "content": "Agatha Christie was born on September 15, 1890 in Torquay, England. She wrote 66 detective novels and 14 short story collections, creating famous characters Hercule Poirot and Miss Marple. She is the best-selling novelist of all time. She died on January 12, 1976."},
    
    # ASTRONOMY (10 documents)
    {"title": "Jupiter - Largest Planet", "content": "Jupiter is the largest planet in our Solar System with a diameter of approximately 139,820 kilometers. It is a gas giant composed primarily of hydrogen and helium. Jupiter has at least 95 known moons including the four large Galilean moons."},
    {"title": "Saturn - Ringed Planet", "content": "Saturn is the sixth planet from the Sun and the second-largest in the Solar System with a diameter of about 116,460 kilometers. It is famous for its prominent ring system made of ice and rock particles. Saturn has at least 146 known moons."},
    {"title": "Mars - The Red Planet", "content": "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface. It has a diameter of approximately 6,779 kilometers, about half the size of Earth. Mars has two small moons: Phobos and Deimos."},
    {"title": "Moon - Earth's Natural Satellite", "content": "The Moon is Earth's only permanent natural satellite with a diameter of 3,474 kilometers. It orbits Earth at an average distance of 384,400 kilometers. The Moon influences Earth's tides and has been visited by humans during the Apollo missions."},
    {"title": "International Space Station", "content": "The International Space Station (ISS) is a modular space station in low Earth orbit. Construction began in 1998 through collaboration between NASA, Roscosmos, ESA, JAXA, and CSA. The ISS orbits Earth at approximately 408 kilometers altitude."},
    {"title": "Hubble Space Telescope", "content": "The Hubble Space Telescope was launched into low Earth orbit on April 24, 1990. It is one of the largest and most versatile space telescopes, operated by NASA and ESA. Hubble has made over 1.5 million observations and revolutionized astronomy."},
    {"title": "Milky Way Galaxy", "content": "The Milky Way is the galaxy containing our Solar System. It is a barred spiral galaxy with a diameter of about 100,000 light-years. The Milky Way contains between 100-400 billion stars and possibly as many planets."},
    {"title": "Big Bang Theory - Origin of Universe", "content": "The Big Bang theory is the prevailing cosmological model explaining the origin of the universe. It proposes that the universe began approximately 13.8 billion years ago from an extremely hot and dense state. The theory is supported by cosmic microwave background radiation."},
    {"title": "Black Holes - Cosmic Phenomena", "content": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their lives. The first image of a black hole was captured in 2019."},
    {"title": "Voyager 1 - Space Probe", "content": "Voyager 1 is a space probe launched by NASA on September 5, 1977. It is the most distant human-made object from Earth, having entered interstellar space in 2012. Voyager 1 carries the Golden Record containing sounds and images from Earth."},
]


async def ingest_fever_data():
    """Ingest 50 FEVER documents with batching to avoid rate limits"""
    
    print("="*80)
    print("FEVER DATA INGESTION - 50 DOCUMENTS")
    print("="*80)
    print()
    print("Documents to ingest:")
    print("  ✓ 10 Scientists (Einstein, Curie, Newton, Galileo, etc.)")
    print("  ✓ 10 Technology (Apple, Microsoft, Google, Python, etc.)")
    print("  ✓ 10 Geography (Everest, Pacific Ocean, Amazon, etc.)")
    print("  ✓ 10 Arts/Literature (Shakespeare, da Vinci, Picasso, etc.)")
    print("  ✓ 10 Astronomy (Jupiter, Mars, Moon, Hubble, etc.)")
    print()
    print("Pipeline: Normalization → Extraction → Validation → Fusion → Embedding")
    print()
    print("Rate Limit Strategy:")
    print("  - Processing in batches of 10 documents")
    print("  - 60-second delay between batches")
    print("  - Estimated time: 15-20 minutes")
    print()
    print("="*80)
    print()
    
    # Create temp directory
    temp_dir = Path(__file__).parent / "temp_ingestion"
    temp_dir.mkdir(exist_ok=True)
    
    service = IngestionService()
    ingested_count = 0
    batch_size = 10  # Process 10 docs at a time
    
    for batch_num in range(0, len(FEVER_DOCUMENTS), batch_size):
        batch = FEVER_DOCUMENTS[batch_num:batch_num + batch_size]
        batch_label = f"BATCH {batch_num//batch_size + 1}/{(len(FEVER_DOCUMENTS)-1)//batch_size + 1}"
        
        print(f"\n{'='*60}")
        print(f"{batch_label} - Processing {len(batch)} documents")
        print(f"{'='*60}")
        
        for i, doc in enumerate(batch, batch_num + 1):
            try:
                # Write to temp file
                temp_file = temp_dir / f"fever_doc_{i}.txt"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(f"{doc['title']}\n\n{doc['content']}")
                
                # Ingest via service (triggers Celery pipeline)
                raw_doc = await service.ingest_from_file(
                    file_path=temp_file,
                    source_type=DocumentType.TEXT,
                )
                
                ingested_count += 1
                print(f"[{i}/50] ✓ {doc['title'][:50]}")
                
                # Clean up temp file
                temp_file.unlink()
                
            except Exception as e:
                print(f"[{i}/50] ✗ Error: {e}")
        
        # Wait between batches to avoid Groq rate limits
        if batch_num + batch_size < len(FEVER_DOCUMENTS):
            print(f"\n⏳ Waiting 60 seconds (Groq rate limit mitigation)...")
            await asyncio.sleep(60)
    
    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass
    
    print()
    print("="*80)
    print(f"✅ Queued {ingested_count}/50 documents for processing")
    print("="*80)
    print()
    print("⏳ Wait 10-15 minutes for Celery to complete the pipeline.")
    print()
    print("Monitor progress:")
    print("  - Check Celery worker logs")
    print("  - Run: python helpers/check_neo4j.py")
    print()
    print("Once complete (~250+ relationships in Neo4j), run Step 2:")
    print("  python tests/benchmarks/fever/2_run_evaluation.py")
    print()


if __name__ == "__main__":
    asyncio.run(ingest_fever_data())
