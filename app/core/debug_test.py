from vector_store import search_similar_chunks

query = "Where does DocuMind AI store metadata and chat history?"
results = search_similar_chunks(query, top_k=5)

print(f"Total results found: {len(results)}\n")

for i, r in enumerate(results):
    print(f"--- Result {i+1} (distance: {r['distance']:.4f}, source: {r['source']}) ---")
    print(r["text"])
    print()