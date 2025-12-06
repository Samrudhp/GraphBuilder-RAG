from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['graphbuilder_rag']

triples = list(db.validated_triples.find({}))
print(f"Found {len(triples)} triples\n")

for i, t in enumerate(triples, 1):
    print(f"Triple {i}:")
    print(f"  Subject: {t['triple']['subject']}")
    print(f"  Predicate: {t['triple']['predicate']}")
    print(f"  Object: {t['triple']['object']}")
    print(f"  Confidence: {t['validation']['confidence_score']}")
    print(f"  Status: {t['status']}")
    print(f"  Wikipedia: {t['validation']['external_verifications']['wikipedia']}")
    print(f"  Wikidata: {t['validation']['external_verifications']['wikidata']}")
    print()
