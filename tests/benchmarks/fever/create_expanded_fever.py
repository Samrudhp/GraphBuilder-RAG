#!/usr/bin/env python3
"""
Create Expanded FEVER-Style Dataset

Generates 20 evidence documents and 5000 claims for comprehensive testing.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Expanded evidence documents (20 total)
EVIDENCE_DOCUMENTS = [
    # Original 8 from FEVER
    "Albert Einstein was born in Ulm, Germany on March 14, 1879.",
    "Marie Curie was the first woman to win a Nobel Prize.",
    "Newton's Principia formulated the three laws of motion.",
    "The telephone was invented by Alexander Graham Bell, not Einstein.",
    "Marie Curie was born in Warsaw, Poland, not France.",
    "Isaac Newton was born in 1642 and died in 1727.",
    "There is no reliable historical record of Einstein's favorite color.",
    "Marie Curie had five siblings: four sisters and one brother.",

    # Additional 12 evidence documents
    "Charles Darwin published The Origin of Species in 1859.",
    "The Wright brothers achieved the first powered flight in 1903.",
    "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
    "The first moon landing occurred in 1969 with Apollo 11.",
    "DNA structure was discovered by Watson and Crick in 1953.",
    "The Great Wall of China was built over several dynasties starting from 221 BC.",
    "Shakespeare wrote Romeo and Juliet around 1595.",
    "The Internet was developed in the late 1960s by DARPA.",
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "The human genome contains approximately 3 billion base pairs.",
    "Mount Everest is the highest mountain on Earth at 8,848 meters.",
    "The first computer bug was found in 1947 by Grace Hopper."
]

def generate_supports_claims(evidence: str, count: int = 50) -> List[Dict]:
    """Generate SUPPORTS claims for an evidence document."""
    claims = []

    # Extract key facts from evidence
    if "Einstein" in evidence and "born" in evidence:
        base_claims = [
            "Albert Einstein was born in Germany.",
            "Einstein's birthplace is Ulm.",
            "Albert Einstein was born on March 14, 1879.",
            "Einstein was born in Ulm, Germany.",
            "The year Einstein was born is 1879.",
            "Einstein's birth month is March.",
            "Albert Einstein was born in the 19th century.",
            "Einstein was a German-born physicist.",
            "Ulm is where Einstein was born.",
            "March 14 is Einstein's birthday.",
            "1879 is the birth year of Einstein.",
            "Einstein's birth occurred in Germany.",
            "The city of Ulm is Einstein's birthplace.",
            "Einstein entered the world in 1879.",
            "Germany claims Einstein as a native son.",
        ]
    elif "Marie Curie" in evidence and "Nobel" in evidence:
        base_claims = [
            "Marie Curie won a Nobel Prize.",
            "Curie was the first woman Nobel laureate.",
            "Marie Curie received the Nobel Prize.",
            "The first female Nobel Prize winner was Marie Curie.",
            "Curie achieved a scientific milestone with Nobel Prize.",
            "Marie Curie earned Nobel recognition.",
            "Curie broke gender barriers in Nobel awards.",
            "The Nobel Prize was awarded to Marie Curie.",
            "Marie Curie holds Nobel Prize distinction.",
            "Curie's Nobel win was historic.",
            "Marie Curie was Nobel Prize recipient.",
            "The Nobel committee honored Marie Curie.",
            "Marie Curie won Nobel Prize in science.",
            "Curie's achievement includes Nobel Prize.",
            "Marie Curie received Nobel laureate status.",
        ]
    elif "Newton" in evidence and "laws of motion" in evidence:
        base_claims = [
            "Isaac Newton formulated the laws of motion.",
            "Newton's Principia contains the laws of motion.",
            "The laws of motion were formulated by Newton.",
            "Newton published the three laws of motion.",
            "The Principia includes Newton's laws of motion.",
            "Newton discovered the laws of motion.",
            "The laws of motion come from Newton.",
            "Newton's work includes motion laws.",
            "The Principia details Newton's laws.",
            "Newton established motion principles.",
            "The laws of motion are Newton's creation.",
            "Newton's scientific work covers motion.",
            "The Principia contains Newton's discoveries.",
            "Newton formulated fundamental motion laws.",
            "The laws of motion were Newton's contribution.",
        ]
    elif "telephone" in evidence and "Bell" in evidence:
        base_claims = [
            "Alexander Graham Bell invented the telephone.",
            "The telephone was invented by Bell.",
            "Bell, not Einstein, invented the telephone.",
            "Alexander Graham Bell is the telephone inventor.",
            "The telephone inventor was Alexander Graham Bell.",
            "Bell created the telephone device.",
            "The telephone invention belongs to Bell.",
            "Bell developed telephone technology.",
            "Alexander Graham Bell made telephone invention.",
            "The telephone was Bell's creation.",
            "Bell is credited with telephone invention.",
            "The telephone inventor is Alexander Graham Bell.",
            "Bell's invention was the telephone.",
            "Alexander Graham Bell invented telephony.",
            "The telephone came from Bell's work.",
        ]
    elif "Marie Curie" in evidence and "born" in evidence:
        base_claims = [
            "Marie Curie was born in Poland.",
            "Curie's birthplace is Warsaw.",
            "Marie Curie was born in Warsaw, Poland.",
            "Poland is Marie Curie's birthplace.",
            "Curie was not born in France.",
            "Warsaw is where Marie Curie was born.",
            "Marie Curie originated from Poland.",
            "Poland was Marie Curie's birth country.",
            "Curie's birth took place in Warsaw.",
            "Marie Curie was Polish by birth.",
            "Warsaw, Poland is Curie's birthplace.",
            "Marie Curie was born Polish.",
            "Curie's origins are in Poland.",
            "Poland claims Marie Curie as native.",
            "Marie Curie was born in Polish territory.",
        ]
    elif "Newton" in evidence and "1642" in evidence:
        base_claims = [
            "Isaac Newton was born in 1642.",
            "Newton died in 1727.",
            "Newton lived for 85 years.",
            "Newton's lifespan was from 1642 to 1727.",
            "Newton was born in the 17th century.",
            "1642 marks Newton's birth year.",
            "Newton's death occurred in 1727.",
            "Newton lived 85 years on Earth.",
            "The years 1642 to 1727 span Newton's life.",
            "Newton was a 17th century figure.",
            "1642 is Newton's birth year.",
            "1727 is Newton's death year.",
            "Newton's life lasted 85 years.",
            "Newton lived through 17th and 18th centuries.",
            "Newton's birth and death years are 1642 and 1727.",
        ]
    elif "Darwin" in evidence:
        base_claims = [
            "Charles Darwin published The Origin of Species.",
            "The Origin of Species was published in 1859.",
            "Darwin wrote The Origin of Species.",
            "The Origin of Species publication year is 1859.",
            "Darwin authored evolutionary theory.",
            "The Origin of Species came from Darwin.",
            "Darwin published in 1859.",
            "The Origin of Species is Darwin's work.",
            "1859 saw Origin of Species publication.",
            "Darwin's book was published in 1859.",
            "The Origin of Species bears Darwin's name.",
            "Darwin wrote about evolution.",
            "The Origin of Species publication date is 1859.",
            "Darwin's publication year was 1859.",
            "The Origin of Species was Darwin's contribution.",
        ]
    elif "Wright brothers" in evidence:
        base_claims = [
            "The Wright brothers achieved powered flight.",
            "First powered flight was in 1903.",
            "The Wright brothers flew in 1903.",
            "Powered flight began in 1903.",
            "Wright brothers pioneered aviation.",
            "The Wrights achieved flight in 1903.",
            "Powered flight started with Wright brothers.",
            "1903 marks first powered flight.",
            "The Wright brothers flew powered aircraft.",
            "First flight was by Wright brothers.",
            "The Wrights accomplished powered flight.",
            "1903 saw Wright brothers' flight.",
            "The Wright brothers pioneered flight.",
            "Powered flight achievement belongs to Wrights.",
            "The Wright brothers flew in 1903.",
        ]
    elif "Mona Lisa" in evidence:
        base_claims = [
            "Leonardo da Vinci painted the Mona Lisa.",
            "The Mona Lisa was painted by da Vinci.",
            "Da Vinci created the Mona Lisa.",
            "The Mona Lisa painting dates to the Renaissance.",
            "Da Vinci's masterpiece is Mona Lisa.",
            "The Mona Lisa comes from da Vinci.",
            "Da Vinci painted the famous portrait.",
            "The Mona Lisa is da Vinci's work.",
            "Da Vinci created Mona Lisa painting.",
            "The Mona Lisa was painted by Leonardo.",
            "Da Vinci's art includes Mona Lisa.",
            "The Mona Lisa is Renaissance art.",
            "Leonardo da Vinci made Mona Lisa.",
            "The Mona Lisa painting is by da Vinci.",
            "Da Vinci painted Mona Lisa between 1503 and 1519.",
        ]
    elif "moon landing" in evidence:
        base_claims = [
            "The first moon landing was in 1969.",
            "Apollo 11 achieved moon landing.",
            "Humans first walked on moon in 1969.",
            "The moon landing occurred in 1969.",
            "Apollo 11 landed on moon.",
            "1969 saw first moon landing.",
            "Humans reached moon in 1969.",
            "The moon landing happened in 1969.",
            "Apollo 11 mission landed on moon.",
            "First lunar landing was 1969.",
            "Moon landing occurred during Apollo 11.",
            "1969 marks moon landing year.",
            "Humans walked on moon first time in 1969.",
            "Apollo 11 accomplished moon landing.",
            "The moon landing took place in 1969.",
        ]
    elif "DNA" in evidence:
        base_claims = [
            "DNA structure was discovered in 1953.",
            "Watson and Crick discovered DNA structure.",
            "DNA double helix was found in 1953.",
            "Watson and Crick made DNA discovery.",
            "DNA structure discovery year is 1953.",
            "Watson and Crick found double helix.",
            "1953 saw DNA structure discovery.",
            "DNA discovery was made by Watson and Crick.",
            "The double helix was discovered in 1953.",
            "Watson and Crick discovered DNA in 1953.",
            "DNA structure was revealed in 1953.",
            "Watson and Crick's discovery was DNA structure.",
            "1953 marks DNA discovery year.",
            "DNA double helix discovery by Watson and Crick.",
            "Watson and Crick discovered DNA structure.",
        ]
    elif "Great Wall" in evidence:
        base_claims = [
            "The Great Wall was built starting from 221 BC.",
            "Great Wall construction began in 221 BC.",
            "The Great Wall spans several dynasties.",
            "Great Wall building started 221 BC.",
            "The Great Wall construction began in 221 BC.",
            "Great Wall was built over dynasties.",
            "221 BC marks Great Wall start.",
            "The Great Wall began construction in 221 BC.",
            "Great Wall spans multiple dynasties.",
            "Construction of Great Wall started 221 BC.",
            "The Great Wall was built starting 221 BC.",
            "Great Wall construction spans dynasties.",
            "221 BC is Great Wall construction start.",
            "The Great Wall was built over several dynasties.",
            "Great Wall building began in 221 BC.",
        ]
    elif "Shakespeare" in evidence:
        base_claims = [
            "Shakespeare wrote Romeo and Juliet.",
            "Romeo and Juliet was written around 1595.",
            "Shakespeare authored Romeo and Juliet.",
            "Romeo and Juliet dates to around 1595.",
            "Shakespeare wrote the play Romeo and Juliet.",
            "Romeo and Juliet is Shakespeare's work.",
            "Shakespeare authored Romeo and Juliet around 1595.",
            "The play Romeo and Juliet was written by Shakespeare.",
            "Shakespeare wrote Romeo and Juliet circa 1595.",
            "Romeo and Juliet comes from Shakespeare.",
            "Shakespeare's play Romeo and Juliet dates to 1595.",
            "Romeo and Juliet was penned by Shakespeare.",
            "Shakespeare wrote Romeo and Juliet in 1595.",
            "The author of Romeo and Juliet is Shakespeare.",
            "Romeo and Juliet was written around 1595 by Shakespeare.",
        ]
    elif "Internet" in evidence:
        base_claims = [
            "The Internet was developed in the late 1960s.",
            "DARPA developed the Internet.",
            "Internet development began in 1960s.",
            "DARPA created the Internet.",
            "The Internet originated in late 1960s.",
            "DARPA developed Internet technology.",
            "Internet was created in 1960s.",
            "DARPA is Internet's developer.",
            "The Internet development started in late 1960s.",
            "DARPA created the Internet network.",
            "Internet origins are in 1960s.",
            "DARPA developed the Internet.",
            "The Internet was developed by DARPA.",
            "Late 1960s saw Internet development.",
            "DARPA created Internet in 1960s.",
        ]
    elif "Penicillin" in evidence:
        base_claims = [
            "Penicillin was discovered in 1928.",
            "Alexander Fleming discovered penicillin.",
            "Fleming found penicillin in 1928.",
            "Penicillin discovery year is 1928.",
            "Fleming discovered penicillin antibiotic.",
            "1928 marks penicillin discovery.",
            "Alexander Fleming found penicillin.",
            "Penicillin was discovered by Fleming in 1928.",
            "Fleming's discovery was penicillin.",
            "1928 saw penicillin discovery.",
            "Alexander Fleming discovered the antibiotic.",
            "Penicillin discovery occurred in 1928.",
            "Fleming found penicillin in 1928.",
            "The discovery of penicillin was in 1928.",
            "Alexander Fleming discovered penicillin.",
        ]
    elif "genome" in evidence:
        base_claims = [
            "The human genome has 3 billion base pairs.",
            "Human genome contains approximately 3 billion base pairs.",
            "Genome size is about 3 billion base pairs.",
            "Human DNA has 3 billion base pairs.",
            "The human genome consists of 3 billion base pairs.",
            "Approximately 3 billion base pairs in human genome.",
            "Human genome size is 3 billion base pairs.",
            "The genome contains about 3 billion base pairs.",
            "Human DNA comprises 3 billion base pairs.",
            "The human genome has roughly 3 billion base pairs.",
            "3 billion base pairs make up human genome.",
            "Human genome contains 3 billion base pairs.",
            "The size of human genome is 3 billion base pairs.",
            "Human DNA has approximately 3 billion base pairs.",
            "The human genome consists of about 3 billion base pairs.",
        ]
    elif "Everest" in evidence:
        base_claims = [
            "Mount Everest is 8,848 meters high.",
            "Everest is the highest mountain.",
            "Mount Everest height is 8,848 meters.",
            "Everest reaches 8,848 meters.",
            "Mount Everest is the tallest mountain.",
            "Everest's height is 8,848 meters.",
            "The highest mountain is Everest.",
            "Mount Everest stands at 8,848 meters.",
            "Everest is 8,848 meters tall.",
            "Mount Everest reaches 8,848 meters.",
            "The tallest peak is Mount Everest.",
            "Everest's elevation is 8,848 meters.",
            "Mount Everest is the highest at 8,848 meters.",
            "Everest reaches a height of 8,848 meters.",
            "Mount Everest is 8,848 meters high.",
        ]
    elif "computer bug" in evidence:
        base_claims = [
            "The first computer bug was found in 1947.",
            "Grace Hopper found the first computer bug.",
            "Computer bug discovered in 1947.",
            "Grace Hopper discovered computer bug.",
            "The first bug was found in 1947.",
            "Grace Hopper discovered the computer bug.",
            "1947 saw first computer bug discovery.",
            "Grace Hopper found the bug in 1947.",
            "The computer bug was discovered in 1947.",
            "Grace Hopper discovered first computer bug.",
            "1947 marks computer bug discovery.",
            "Grace Hopper found computer bug in 1947.",
            "The first computer bug discovery was in 1947.",
            "Grace Hopper discovered the bug.",
            "Computer bug was found by Grace Hopper in 1947.",
        ]
    else:
        # Generic claims for remaining evidence
        base_claims = [
            f"The evidence states: {evidence[:50]}...",
            f"According to records: {evidence[:50]}...",
            f"Historical fact: {evidence[:50]}...",
            f"Scientific record shows: {evidence[:50]}...",
            f"Documented information: {evidence[:50]}...",
            f"The evidence confirms: {evidence[:50]}...",
            f"Records indicate: {evidence[:50]}...",
            f"Established fact: {evidence[:50]}...",
            f"The documentation states: {evidence[:50]}...",
            f"Verified information: {evidence[:50]}...",
            f"The evidence proves: {evidence[:50]}...",
            f"According to sources: {evidence[:50]}...",
            f"Factual record: {evidence[:50]}...",
            f"The evidence demonstrates: {evidence[:50]}...",
            f"Confirmed information: {evidence[:50]}...",
        ]

    # Generate variations
    for i in range(min(count, len(base_claims) * 3)):
        base_idx = i % len(base_claims)
        claim_text = base_claims[base_idx]

        # Add some variations
        if i >= len(base_claims):
            # Create paraphrased versions
            if "was" in claim_text:
                claim_text = claim_text.replace("was", "is", 1)
            elif "is" in claim_text:
                claim_text = claim_text.replace("is", "was", 1)

        claims.append({
            "id": len(claims) + 1,
            "claim": claim_text,
            "label": "SUPPORTS",
            "evidence": evidence
        })

    return claims[:count]

def generate_refutes_claims(evidence: str, count: int = 50) -> List[Dict]:
    """Generate REFUTES claims for an evidence document."""
    claims = []

    if "Einstein" in evidence and "born" in evidence:
        base_claims = [
            "Albert Einstein was born in Austria.",
            "Einstein was born in Berlin.",
            "Albert Einstein was born on March 14, 1880.",
            "Einstein's birthplace is Vienna.",
            "The year Einstein was born is 1880.",
        ]
    elif "Marie Curie" in evidence and "Nobel" in evidence:
        base_claims = [
            "Marie Curie never won a Nobel Prize.",
            "Curie was not the first woman Nobel laureate.",
            "Marie Curie received no Nobel Prize.",
            "A man was the first Nobel Prize winner.",
            "Curie did not achieve Nobel Prize milestone.",
        ]
    elif "Newton" in evidence and "laws of motion" in evidence:
        base_claims = [
            "Galileo formulated the laws of motion.",
            "Einstein formulated the laws of motion.",
            "The laws of motion were formulated by Galileo.",
            "Newton did not publish the laws of motion.",
            "The Principia does not include motion laws.",
        ]
    elif "telephone" in evidence and "Bell" in evidence:
        base_claims = [
            "Einstein invented the telephone.",
            "The telephone was invented by Einstein.",
            "Bell did not invent the telephone.",
            "Thomas Edison invented the telephone.",
            "The telephone inventor was Einstein.",
        ]
    elif "Marie Curie" in evidence and "born" in evidence:
        base_claims = [
            "Marie Curie was born in France.",
            "Curie's birthplace is Paris.",
            "Marie Curie was born in France.",
            "France is Marie Curie's birthplace.",
            "Curie was born in France.",
        ]
    elif "Newton" in evidence and "1642" in evidence:
        base_claims = [
            "Isaac Newton was born in 1643.",
            "Newton died in 1726.",
            "Newton lived for 84 years.",
            "Newton was born in the 18th century.",
            "Newton's lifespan was from 1643 to 1727.",
        ]
    else:
        # Generic false claims
        base_claims = [
            f"This statement contradicts the evidence.",
            f"The evidence proves this wrong.",
            f"This claim is factually incorrect.",
            f"The documented facts show otherwise.",
            f"This assertion is not supported.",
        ]

    for i in range(min(count, len(base_claims) * 2)):
        base_idx = i % len(base_claims)
        claims.append({
            "id": len(claims) + 1,
            "claim": base_claims[base_idx],
            "label": "REFUTES",
            "evidence": evidence
        })

    return claims[:count]

def generate_neutral_claims(evidence: str, count: int = 50) -> List[Dict]:
    """Generate NOT_ENOUGH_INFO claims."""
    claims = []

    neutral_templates = [
        "What was the exact date of this event?",
        "Who were the witnesses to this event?",
        "What was the weather like during this event?",
        "What were the exact circumstances?",
        "What was the motivation behind this?",
        "What were the long-term consequences?",
        "Who funded this project?",
        "What were the challenges faced?",
        "What alternative theories existed?",
        "What was the public reaction?",
        "What was the economic impact?",
        "Who were the key participants?",
        "What were the technological limitations?",
        "What was the social context?",
        "What were the political implications?",
        "Who documented this event?",
        "What were the immediate effects?",
        "What was the cultural significance?",
        "Who were the contemporary observers?",
        "What were the scientific methods used?",
        "What was the historical context?",
        "Who were the primary sources?",
        "What were the conflicting accounts?",
        "What was the timeline of events?",
        "Who were the beneficiaries?",
        "What were the ethical considerations?",
        "What was the scale of the event?",
        "Who were the critics at the time?",
        "What were the technological innovations?",
        "What was the global impact?",
        "Who were the international observers?",
        "What were the diplomatic consequences?",
        "What was the media coverage like?",
        "Who were the eyewitnesses?",
        "What were the statistical measurements?",
        "What was the experimental design?",
        "Who were the peer reviewers?",
        "What were the control conditions?",
        "What was the sample size?",
        "Who were the research assistants?",
        "What were the funding sources?",
        "What was the institutional context?",
        "Who were the collaborating scientists?",
        "What were the preliminary findings?",
        "What was the publication process?",
        "Who were the editors involved?",
        "What were the review comments?",
        "What was the revision history?",
        "Who were the patent holders?",
        "What were the commercial applications?",
    ]

    for i in range(count):
        template = neutral_templates[i % len(neutral_templates)]
        claims.append({
            "id": len(claims) + 1,
            "claim": template,
            "label": "NOT_ENOUGH_INFO",
            "evidence": evidence
        })

    return claims

def create_expanded_fever_dataset():
    """Create expanded dataset with 20 docs and 5000 claims."""

    all_claims = []
    claim_id = 1

    print(f"Creating expanded dataset with {len(EVIDENCE_DOCUMENTS)} evidence documents...")

    for doc_idx, evidence in enumerate(EVIDENCE_DOCUMENTS):
        print(f"Processing document {doc_idx + 1}/{len(EVIDENCE_DOCUMENTS)}")

        # Generate claims for this evidence - increased counts
        supports = generate_supports_claims(evidence, 150)  # Increased from 125
        refutes = generate_refutes_claims(evidence, 150)    # Increased from 125
        neutral = generate_neutral_claims(evidence, 200)    # Increased from 125

        # Update IDs
        for claim in supports + refutes + neutral:
            claim["id"] = claim_id
            claim_id += 1

        all_claims.extend(supports + refutes + neutral)

    # Shuffle to mix claim types
    random.shuffle(all_claims)

    # Save expanded dataset
    output_file = Path("tests/benchmarks/data/fever/fever_expanded_5000.jsonl")

    with open(output_file, "w") as f:
        for claim in all_claims:
            f.write(json.dumps(claim) + "\n")

    print(f"✅ Created {len(all_claims)} claims in {output_file}")
    print(f"📊 Label distribution: {sum(1 for c in all_claims if c['label'] == 'SUPPORTS')} SUPPORTS, "
          f"{sum(1 for c in all_claims if c['label'] == 'REFUTES')} REFUTES, "
          f"{sum(1 for c in all_claims if c['label'] == 'NOT_ENOUGH_INFO')} NOT_ENOUGH_INFO")

    return all_claims

if __name__ == "__main__":
    create_expanded_fever_dataset()