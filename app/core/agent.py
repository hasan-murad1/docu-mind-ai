"""Agentic RAG core (Phase 1).

Contains:
- TraceStep / AgentTrace: records every agent step (no secrets, no prompts).
- LLMBudget: hard limit on LLM calls per question.
- run_agent: minimal agent (vector search -> one LLM call) that records a trace.
"""

import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from usage_limiter import MAX_QUESTION_LENGTH

# --- Hard limits (no infinite loops) ---
MAX_RETRIES = 3
MAX_LLM_CALLS_PER_QUESTION = 8
DEFAULT_TOP_K = 7

NO_INFO_ANSWER = "I don't have information about that in the uploaded document(s)."


class LLMBudgetExceeded(Exception):
    """Raised when the agent tries to make more LLM calls than allowed."""


class LLMBudget:
    """Counts LLM calls for one question and enforces a hard maximum."""

    def __init__(self, max_calls: int = MAX_LLM_CALLS_PER_QUESTION):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls
        self.used = 0

    def consume(self) -> None:
        """Count one LLM call, or raise if the budget is used up."""
        if self.used >= self.max_calls:
            raise LLMBudgetExceeded(
                f"LLM call limit reached ({self.max_calls} per question)."
            )
        self.used += 1


@dataclass
class TraceStep:
    """One recorded agent step. Only safe, small values are stored here."""

    step: str
    tool: str
    chunks_retrieved: int
    best_distance: Optional[float]
    retry: int
    time_ms: float
    llm_calls: int  # total LLM calls used so far, at the end of this step


class AgentTrace:
    """Ordered list of TraceStep objects for one question."""

    def __init__(self):
        self.steps: list[TraceStep] = []

    def record(
        self,
        step: str,
        started_at: float,
        tool: str = "none",
        chunks_retrieved: int = 0,
        best_distance: Optional[float] = None,
        retry: int = 0,
        llm_calls: int = 0,
    ) -> None:
        """Add a step. `started_at` comes from time.perf_counter()."""
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self.steps.append(
            TraceStep(
                step=step,
                tool=tool,
                chunks_retrieved=chunks_retrieved,
                best_distance=(
                    round(best_distance, 4) if best_distance is not None else None
                ),
                retry=retry,
                time_ms=round(elapsed_ms, 1),
                llm_calls=llm_calls,
            )
        )

    def to_list(self) -> list[dict]:
        """Return the trace as plain dicts (JSON-friendly)."""
        return [asdict(s) for s in self.steps]


def validate_question(question) -> str:
    """Return a clean question string, or raise ValueError."""
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question too long. Max {MAX_QUESTION_LENGTH} characters.")
    return question


def _default_llm(prompt: str) -> str:
    """Call the provider selected in rag_pipeline (Ollama or Groq)."""
    import rag_pipeline  # imported lazily so tests don't load heavy models

    if rag_pipeline.LLM_PROVIDER == "groq":
        return rag_pipeline.ask_groq(prompt)
    return rag_pipeline.ask_llm(prompt)


def run_agent(
    question: str,
    search_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    build_prompt_fn: Optional[Callable] = None,
    threshold: Optional[float] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Phase 1 agent: vector search -> relevance filter -> one LLM call.
    Returns {"answer", "sources", "trace"}.
    The function arguments can be replaced in tests (dependency injection).
    """
    question = validate_question(question)

    if search_fn is None or build_prompt_fn is None or threshold is None:
        import rag_pipeline  # lazy import

        search_fn = search_fn or rag_pipeline.search_similar_chunks
        build_prompt_fn = build_prompt_fn or rag_pipeline.build_prompt
        threshold = (
            threshold if threshold is not None else rag_pipeline.RELEVANCE_THRESHOLD
        )
    llm_fn = llm_fn or _default_llm

    trace = AgentTrace()
    budget = LLMBudget()

    # Step 1: retrieve with the vector search tool
    t0 = time.perf_counter()
    retrieved = search_fn(question, top_k=top_k)
    relevant = [c for c in retrieved if c["distance"] <= threshold]
    best = min((c["distance"] for c in retrieved), default=None)
    trace.record(
        "retrieve",
        t0,
        tool="vector_search",
        chunks_retrieved=len(relevant),
        best_distance=best,
        llm_calls=budget.used,
    )

    if not relevant:
        trace.record("finish_no_info", time.perf_counter(), llm_calls=budget.used)
        return {"answer": NO_INFO_ANSWER, "sources": [], "trace": trace.to_list()}

    # Step 2: generate the answer (1 LLM call)
    t1 = time.perf_counter()
    budget.consume()
    prompt = build_prompt_fn(question, relevant)
    answer = llm_fn(prompt)
    trace.record("generate", t1, llm_calls=budget.used)

    sources = sorted(set(c["source"] for c in relevant))
    return {"answer": answer, "sources": sources, "trace": trace.to_list()}


# Manual test: run from the project root
if __name__ == "__main__":
    import json

    result = run_agent("What does this project use to store embeddings?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Trace:", json.dumps(result["trace"], indent=2))