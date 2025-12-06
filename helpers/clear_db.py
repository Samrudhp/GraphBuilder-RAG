from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['graphbuilder_rag']

print("Clearing all collections...")
db.raw_documents.delete_many({})
db.normalized_docs.delete_many({})
db.candidate_triples.delete_many({})
db.validated_triples.delete_many({})
db.embeddings_meta.delete_many({})
db.chunks.delete_many({})

print("✓ Database cleared")
print(f"Raw docs: {db.raw_documents.count_documents({})}")
print(f"Normalized: {db.normalized_docs.count_documents({})}")
print(f"Candidate triples: {db.candidate_triples.count_documents({})}")
print(f"Validated triples: {db.validated_triples.count_documents({})}")
