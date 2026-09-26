import requests
from document_processor import extract_text, chunk_text
from vector_store import add_chunks_to_store, search_similar_chunks

# Ollama runs a local server on this address by default
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"
# Safety limit: don't wait forever for a response if Ollama hangs
REQUEST_TIMEOUT_SECONDS = 180

def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build a prompt that forces the LLM to answer ONLY using the
    provided context, instead of relying on its own (possibly wrong) memory.
    """
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are a precise assistant that answers questions using ONLY the context below.

Important rules:
- If the context clearly discusses the topic being asked about, answer using the information and any straightforward comparisons or calculations needed (for example, comparing a number in the question to a number in the context).
- If the question asks about something that is NOT mentioned anywhere in the context at all (a different topic, person, or fact), say "I don't have information about that in the uploaded document(s)."
- Do not write creative content (poems, stories, etc.) — only answer factual questions based on the context.

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt


def ask_llm(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the generated answer."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # low temperature = more factual, less random
            },
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["response"]


# Chunks with a distance above this are considered "not actually relevant"
# and will be filtered out before being sent to the LLM. Tune this if needed.
RELEVANCE_THRESHOLD = 2.0

def answer_question(question: str, top_k: int = 7) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Filter out chunks that aren't actually close enough to be relevant
    3. Build a grounded prompt
    4. Ask the LLM to answer using only that context
    """
    retrieved_chunks = search_similar_chunks(question, top_k=top_k)

    # Keep only chunks that are genuinely close in meaning to the question.
    # This stops leftover/unrelated chunks from confusing the model on
    # greetings, small talk, or questions about things not in the document.
    relevant_chunks = [c for c in retrieved_chunks if c["distance"] <= RELEVANCE_THRESHOLD]

    if not relevant_chunks:
        return {
            "answer": "I don't have information about that in the uploaded document(s).",
            "sources": [],
        }

    prompt = build_prompt(question, relevant_chunks)
    answer = ask_llm(prompt)

    sources = list(set(c["source"] for c in relevant_chunks))

    return {
        "answer": answer,
        "sources": sources,
    }


# Manual test
if __name__ == "__main__":
    question = "What does this project use to store embeddings?"
    print(f"Question: {question}\n")

    result = answer_question(question)

    print("Answer:")
    print(result["answer"])
    print(f"\nSources: {result['sources']}")