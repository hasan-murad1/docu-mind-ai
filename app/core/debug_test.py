from vector_store import search_similar_chunks

query = "I'm a Plus Monthly member and my order is BDT 400 — is delivery free?"
results = search_similar_chunks(query, top_k=2)

for i, r in enumerate(results):
    print(f"\n{'='*60}")
    print(f"Result {i+1}: distance={r['distance']:.4f}")
    print(r["text"])