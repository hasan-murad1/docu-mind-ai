from vector_store import search_similar_chunks

query = "Can I withdraw money from my NovaCart Wallet as cash?"
results = search_similar_chunks(query, top_k=7)

for i, r in enumerate(results):
    print(f"\n--- Result {i+1} (distance: {r['distance']:.4f}) ---")
    print(r["text"][:300])