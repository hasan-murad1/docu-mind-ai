import requests
from document_processor import extract_text, chunk_text
from vector_store import add_chunks_to_store, search_similar_chunks

# Ollama runs a local server on this address by default
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

# Safety limit: don't wait forever for a response if Ollama hangs
REQUEST_TIMEOUT_SECONDS = 60


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build a prompt that forces the LLM to answer ONLY using the
    provided context, instead of relying on its own (possibly wrong) memory.
    """
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information that isn't in the context.

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
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()  # raises an error if the request failed
    return response.json()["response"]


def answer_question(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a grounded prompt
    3. Ask the LLM to answer using only that context
    """
    retrieved_chunks = search_similar_chunks(question, top_k=top_k)

    if not retrieved_chunks:
        return {
            "answer": "No documents found. Please upload a document first.",
            "sources": [],
        }

    prompt = build_prompt(question, retrieved_chunks)
    answer = ask_llm(prompt)

    sources = list(set(c["source"] for c in retrieved_chunks))

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