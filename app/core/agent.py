"""Agentic RAG core.

Contains:
- TraceStep / AgentTrace: records every agent step (no secrets, no prompts).
- LLMBudget: hard limit on LLM calls per question.
- make_counted_llm: every LLM call counts toward the budget AND the daily cap.
- run_agent: plan -> retrieve -> evaluate -> (rewrite + retry, max 3)
  -> draft -> self-check -> answer + sources.
"""

import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from evaluator import Evaluation, evaluate_context
from planner import plan_question
from self_check import SelfCheck, check_answer
from usage_limiter import MAX_QUESTION_LENGTH, check_and_increment_daily_cap

# --- Hard limits (no infinite loops) ---
MAX_RETRIES = 3                  # query rewrites after the first retrieval
MAX_LLM_CALLS_PER_QUESTION = 8   # worst case: 1 plan + 4 evaluate + 1 draft + 1 self-check = 7
MAX_CONTEXT_CHUNKS = 8           # chunks kept across retries (best distance first)
DEFAULT_TOP_K = 7

# Safe default tool: used when the plan is invalid, the chosen tool finds
# nothing, and for every retry with a rewritten query (broadest search)
FALLBACK_TOOL = "vector_search"

NO_INFO_ANSWER = "I don't have information about that in the uploaded document(s)."
_NO_INFO_PREFIXES = ("i don't have information", "i do not have information")


class LLMBudgetExceeded(Exception):
    """Raised when the agent tries to make more LLM calls than allowed."""


class DailyCapExceeded(Exception):
    """Raised when the daily LLM call cap is reached."""


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


def is_no_info_answer(text) -> bool:
    """True if an answer is empty or starts with the standard 'no information' sentence."""
    if not isinstance(text, str):
        return True
    normalized = " ".join(text.replace("\u2019", "'").lower().split())
    normalized = normalized.lstrip("\"'*` ")
    return not normalized or normalized.startswith(_NO_INFO_PREFIXES)


def _default_llm(prompt: str) -> str:
    """Call the provider selected in rag_pipeline (Ollama or Groq)."""
    import rag_pipeline  # imported lazily so tests don't load heavy models

    return rag_pipeline.call_llm(prompt)


def make_counted_llm(
    budget: LLMBudget,
    llm_fn: Callable[[str], str],
    daily_cap_fn: Callable[[], bool],
) -> Callable[[str], str]:
    """
    Wrap an LLM function so every call counts toward both limits.
    Order: per-question budget first (no side effects), then the daily cap.
    """

    def counted(prompt: str) -> str:
        budget.consume()
        if not daily_cap_fn():
            raise DailyCapExceeded("Daily usage limit reached. Please try again tomorrow.")
        return llm_fn(prompt)

    return counted


# ---------- Tools ----------

def default_tools() -> dict:
    """Real read-only search tools, all with the signature (query, top_k, source)."""
    import vector_store  # lazy import: loads the embedding model

    return {
        "vector_search": lambda query, top_k, source=None: vector_store.search_similar_chunks(
            query, top_k=top_k
        ),
        "keyword_search": lambda query, top_k, source=None: vector_store.keyword_search(
            query, top_k=top_k
        ),
        "metadata_search": lambda query, top_k, source=None: vector_store.metadata_search(
            query, source, top_k=top_k
        ),
    }


def run_tool(
    tools: dict,
    tool_name: str,
    query: str,
    top_k: int,
    source: Optional[str] = None,
) -> list[dict]:
    """Run one tool by name. Unknown tool names raise ValueError."""
    tool_fn = tools.get(tool_name)
    if tool_fn is None:
        raise ValueError(f"Unknown tool: {tool_name!r}")
    return tool_fn(query, top_k, source)


def _best_distance(chunks: list[dict]) -> Optional[float]:
    return min((c["distance"] for c in chunks), default=None)


def _retrieve(
    trace: AgentTrace,
    step: str,
    tools: dict,
    tool_name: str,
    query: str,
    source: Optional[str],
    top_k: int,
    threshold: float,
    llm_calls: int,
    retry: int = 0,
) -> list[dict]:
    """Run one tool, keep only chunks within the threshold, and record a trace step."""
    started = time.perf_counter()
    retrieved = run_tool(tools, tool_name, query, top_k, source)
    relevant = [c for c in retrieved if c["distance"] <= threshold]
    trace.record(
        step,
        started,
        tool=tool_name,
        chunks_retrieved=len(relevant),
        best_distance=_best_distance(retrieved),
        retry=retry,
        llm_calls=llm_calls,
    )
    return relevant


def merge_chunks(
    existing: list[dict],
    new: list[dict],
    limit: int = MAX_CONTEXT_CHUNKS,
) -> list[dict]:
    """
    Combine chunks from several retrievals: remove duplicates (same source + text,
    keeping the best distance), sort by distance, and keep at most `limit`.
    """
    best: dict = {}
    for chunk in list(existing) + list(new):
        key = (chunk["source"], chunk["text"])
        if key not in best or chunk["distance"] < best[key]["distance"]:
            best[key] = chunk
    return sorted(best.values(), key=lambda c: c["distance"])[:limit]


def _normalize_query(query: str) -> str:
    """Lowercase and collapse spaces, so 'Refund  Policy' == 'refund policy'."""
    return " ".join(query.lower().split())


def _evaluation_step_name(evaluation: Evaluation) -> str:
    if evaluation.is_fallback:
        return "evaluate_fallback"
    return "evaluate_enough" if evaluation.enough else "evaluate_not_enough"


def _self_check_step_name(check: SelfCheck) -> str:
    if check.is_fallback:
        return "self_check_fallback"
    if check.supported:
        return "self_check_ok"
    return "self_check_revised" if check.answer is not None else "self_check_unsupported"


# ---------- Agent ----------

def run_agent(
    question: str,
    tools: Optional[dict] = None,
    sources_fn: Optional[Callable[[], list]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    build_prompt_fn: Optional[Callable] = None,
    threshold: Optional[float] = None,
    top_k: int = DEFAULT_TOP_K,
    daily_cap_fn: Optional[Callable[[], bool]] = None,
) -> dict:
    """
    Phase 6 agent: plan -> retrieve -> evaluate -> (rewrite + retry)
    -> draft -> self-check -> answer + sources.

    LLM calls: 1 plan + 1 evaluate per attempt (max MAX_RETRIES + 1)
    + 1 draft + 1 self-check. Every call counts toward the per-question
    budget AND the daily cap. "No info" answers never show sources.
    Returns {"answer", "sources", "trace"}.
    All dependencies can be replaced in tests (dependency injection).
    """
    question = validate_question(question)

    if tools is None:
        tools = default_tools()
    if sources_fn is None:
        import vector_store  # lazy import

        sources_fn = vector_store.list_sources
    if build_prompt_fn is None or threshold is None:
        import rag_pipeline  # lazy import

        build_prompt_fn = build_prompt_fn or rag_pipeline.build_prompt
        threshold = (
            threshold if threshold is not None else rag_pipeline.RELEVANCE_THRESHOLD
        )

    trace = AgentTrace()
    budget = LLMBudget()
    counted_llm = make_counted_llm(
        budget,
        llm_fn or _default_llm,
        daily_cap_fn or check_and_increment_daily_cap,
    )

    def finish_no_info(step: str) -> dict:
        trace.record(step, time.perf_counter(), llm_calls=budget.used)
        return {"answer": NO_INFO_ANSWER, "sources": [], "trace": trace.to_list()}

    # Step 1: no documents at all -> answer right away, no LLM call spent
    available_sources = sources_fn()
    if not available_sources:
        return finish_no_info("finish_no_docs")

    # Step 2: plan (1 counted LLM call). Invalid output -> safe fallback plan
    t0 = time.perf_counter()
    plan = plan_question(question, counted_llm, available_sources)
    trace.record(
        "plan_fallback" if plan.is_fallback else "plan",
        t0,
        tool="llm",
        llm_calls=budget.used,
    )

    # Step 3: retrieve -> evaluate -> rewrite, with a hard limit of MAX_RETRIES retries
    context: list[dict] = []
    evaluation: Optional[Evaluation] = None
    tool, query, source = plan.tool, plan.query, plan.source
    tried_queries = {_normalize_query(query)}

    for attempt in range(MAX_RETRIES + 1):
        found = _retrieve(
            trace, "retrieve", tools, tool, query, source,
            top_k, threshold, budget.used, retry=attempt,
        )

        # Free safety net (no LLM call): the planner's specialised tool found nothing
        if attempt == 0 and not found and tool != FALLBACK_TOOL:
            found = _retrieve(
                trace, "retrieve_fallback", tools, FALLBACK_TOOL, question, None,
                top_k, threshold, budget.used, retry=attempt,
            )

        context = merge_chunks(context, found)
        is_last_attempt = attempt == MAX_RETRIES

        # Nothing found and no retries left: stop without spending an LLM call
        if not context and is_last_attempt:
            break

        # Evaluate the context (1 counted LLM call); may suggest a rewritten query
        t_eval = time.perf_counter()
        evaluation = evaluate_context(question, context, counted_llm)
        trace.record(
            _evaluation_step_name(evaluation),
            t_eval,
            tool="llm",
            chunks_retrieved=len(context),
            best_distance=_best_distance(context),
            retry=attempt,
            llm_calls=budget.used,
        )

        if evaluation.enough and context:
            break

        next_query = evaluation.rewrite
        if is_last_attempt or next_query is None or _normalize_query(next_query) in tried_queries:
            break  # no retries left, no new idea, or the same search again

        # Retry with the rewritten query, using the broadest tool
        tried_queries.add(_normalize_query(next_query))
        tool, query, source = FALLBACK_TOOL, next_query, None

    if not context or evaluation is None or not evaluation.enough:
        return finish_no_info("finish_no_info")

    # Step 4: draft an answer to the user's ORIGINAL question (1 counted LLM call)
    t1 = time.perf_counter()
    draft = counted_llm(build_prompt_fn(question, context))
    trace.record("draft", t1, tool="llm", chunks_retrieved=len(context), llm_calls=budget.used)

    # The model itself said "no info": skip the self-check and hide sources
    if is_no_info_answer(draft):
        return finish_no_info("finish_no_info")

    # Step 5: self-check the draft against the same context (1 counted LLM call)
    t2 = time.perf_counter()
    check = check_answer(question, draft, context, counted_llm)
    trace.record(
        _self_check_step_name(check),
        t2,
        tool="llm",
        chunks_retrieved=len(context),
        llm_calls=budget.used,
    )

    if check.supported:  # also covers the fallback (invalid self-check output)
        answer = draft
    elif check.answer is not None and not is_no_info_answer(check.answer):
        answer = check.answer
    else:
        return finish_no_info("finish_no_info")

    sources_used = sorted(set(c["source"] for c in context))
    return {"answer": answer, "sources": sources_used, "trace": trace.to_list()}


# Manual test: run from the project root
#   python app/core/agent.py
#   python app/core/agent.py "Your own question here"
if __name__ == "__main__":
    import json
    import sys

    question = (
        sys.argv[1] if len(sys.argv) > 1
        else "What does this project use to store embeddings?"
    )
    result = run_agent(question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Trace:", json.dumps(result["trace"], indent=2))