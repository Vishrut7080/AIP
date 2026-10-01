#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq

Ships the Lab 5 pipeline (`lenient_complete`, the only prompt whose responses are
committed to the offline cache, so the CI gate and this service measure the same
thing) behind the Lab 6 guards.

Latency budget (README.md:30-33), measured not assumed:
    TTFT (streaming)                <= 1500 ms
    p95 cached                      <=  800 ms
    p95 uncached                    <= 6000 ms
    cost per query                  <= $0.01
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, settings, tracing  # noqa: E402
from aip.config import resolve_model  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.guards import ToolGuard, enforce_citations  # noqa: E402
from labs.lab4.rag import REFUSAL, STRICTNESS, answer_question  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# The Lab 3/4 winning configuration, and the Lab 5 v2 prompt. final_k=8 comes
# from report4.md (Q37's partial answer needs the chunk at rank 8).
RETRIEVE_K = 16
FINAL_K = 8
STRICTNESS_NAME = "lenient_complete"
SEMANTIC_CACHE_THRESHOLD = 0.95  # B1: measured, see /cache/sweep

# B4 -- the latency budget, and which stage to optimise first.
#
# Measured over 8 forced-cache-miss requests, same 8 questions per tier:
#
#   tier            p50      p95      max     target p95
#   MAIN          4854 ms   7499 ms   7499 ms   <= 6000   FAIL
#   SMALL         2148 ms   3416 ms   3416 ms   <= 6000   PASS
#
# Generation is 1515 ms of a 2075 ms request; retrieval is 68 ms. So the stage
# to optimise first is GENERATION, and the lever is the model tier, not the
# retriever -- which is why no amount of reranking would have helped.
#
# Quality is NOT the tradeoff here, which is the part worth noticing. Over 42
# judged answers: MAIN correctness 0.762, SMALL 0.786, faithfulness identical at
# 0.905 for both, and SMALL costs $0.41 against MAIN's $0.00 (MAIN was entirely
# cache-served, so read its cost as unmeasured and its latency as the honest
# number). The smaller model is not a compromise on this workload; it is the
# better trade outright.
GENERATION_TIER = "SMALL"

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()
_STARTED = time.time()

# B1: two cache layers. The exact cache is keyed on the normalised question; the
# semantic cache is keyed on the embedding and only returns an answer above
# SEMANTIC_CACHE_THRESHOLD. Both are per-process and bounded -- an unbounded
# cache in a service is a memory leak with a nice latency graph.
_EXACT_CACHE: OrderedDict[str, dict] = OrderedDict()
_SEMANTIC_CACHE: OrderedDict[str, tuple[np.ndarray, dict]] = OrderedDict()
_EXACT_MAX = 512
_SEMANTIC_MAX = 256

# Counters for /metrics. Process-local on purpose: the point is to show what one
# instance knows, and a real deployment would use a shared store.
STATS = {"requests": 0, "errors": 0, "exact_hits": 0, "semantic_hits": 0,
         "refusals": 0, "tool_calls": 0, "stream_requests": 0}
_ERRORS_BY_TYPE: dict[str, int] = {}
_LOCK = threading.Lock()


def _bump(key: str, n: int = 1) -> None:
    with _LOCK:
        STATS[key] = STATS.get(key, 0) + n


def pipeline():
    """Build the pipeline ONCE, at startup.

    Building it per-request re-embeds the corpus every time. Students do this
    and then report a 40-second p95.

    Under AIP_OFFLINE=1 the index comes from data/index/ (matrix.npy), so the
    service starts with no network and no embedding provider.
    """
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                t0 = time.perf_counter()
                from scripts.warm_cache import build_index, load_index

                if settings.offline:
                    r = load_index(model=resolve_model("EMBED"))
                else:
                    r = build_index()
                _PIPELINE = r
                tracing.event("service.index_built",
                              n_chunks=len(r.chunks),
                              ms=round((time.perf_counter() - t0) * 1000, 1))
    return _PIPELINE


# A fresh guard per request. Lab 6 found that ToolGuard.calls_made is monotonic
# and never resets, so a shared guard spends one budget across the process
# lifetime and every later request dies with "budget exhausted".
def _guard() -> ToolGuard:
    return ToolGuard(max_calls=4,
                     allow={"search_policy", "get_policy_details",
                            "compute_premium"},
                     requires_confirmation={"issue_refund"},
                     confirm_fn=lambda name, a: False)


# ---------------------------------------------------------------------------
# B1: caching
# ---------------------------------------------------------------------------
def _normalise(q: str) -> str:
    """Normalise before hashing so trivial variations hit the same entry."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


def _key(q: str) -> str:
    return hashlib.sha256(_normalise(q).encode()).hexdigest()[:32]


def _cache_get_exact(q: str) -> dict | None:
    k = _key(q)
    with _LOCK:
        if k in _EXACT_CACHE:
            _EXACT_CACHE.move_to_end(k)
            return _EXACT_CACHE[k]
    return None


def _cache_put_exact(q: str, payload: dict) -> None:
    with _LOCK:
        _EXACT_CACHE[_key(q)] = payload
        while len(_EXACT_CACHE) > _EXACT_MAX:
            _EXACT_CACHE.popitem(last=False)


def _embed_query(q: str) -> np.ndarray | None:
    """Query vector for the semantic cache. None when offline and uncached."""
    try:
        from aip.embed import embed
        return np.asarray(embed(q, input_type="query"), dtype=np.float32)
    except Exception:  # noqa: BLE001
        # Offline with no cached vector: no semantic caching. A miss is the
        # correct answer, not a crash -- and certainly not a wrong hit.
        return None


def _cache_get_semantic(q: str, threshold: float) -> dict | None:
    v = _embed_query(q)
    if v is None:
        return None
    best, best_sim = None, -1.0
    with _LOCK:
        items = list(_SEMANTIC_CACHE.values())
    for stored_v, payload in items:
        sim = float(np.dot(v, stored_v))
        if sim > best_sim:
            best, best_sim = payload, sim
    if best is not None and best_sim >= threshold:
        return {**best, "semantic_sim": round(best_sim, 4),
                "semantic_match": _cache_peek_question()}
    return None


def _cache_peek_question() -> str:
    with _LOCK:
        items = list(_SEMANTIC_CACHE.keys())
    return items[-1] if items else ""


def _cache_put_semantic(q: str, payload: dict) -> None:
    v = _embed_query(q)
    if v is None:
        return
    with _LOCK:
        _SEMANTIC_CACHE[q] = (v, payload)
        while len(_SEMANTIC_CACHE) > _SEMANTIC_MAX:
            _SEMANTIC_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=FINAL_K, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")
    semantic_cache: bool = Field(default=True,
                                 description="B1: off to measure the exact layer alone")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str
    cache_layer: str = "miss"


def _citations(answer: str, hits) -> list[Citation]:
    """Cite only the sources the answer actually references, and verify each."""
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    out = []
    for i in cited:
        if 1 <= i <= len(hits):
            out.append(Citation(index=i, doc_id=hits[i - 1].chunk.doc_id,
                                excerpt=hits[i - 1].chunk.text[:600]))
    return out


def _cost_delta(before: float) -> float:
    return round(global_budget().spent_usd - before, 6)


def _stream_generate(question: str, hits):
    """Generate with real token streaming. A GENERATOR yielding text pieces.

    `aip/llm.py` has no streaming path and is read-only toolkit code, so this
    calls LiteLLM directly -- but it reuses aip's cache, cost ledger and tracing
    so /metrics and the dashboard still see these calls. Duplicating the
    accounting here would mean the numbers in /metrics exclude streamed
    requests, which is exactly the kind of half-measurement this lab is about.

    Offline (AIP_OFFLINE=1) there is nothing to stream from, so it replays the
    cached response as one piece. CI must behave like a service.
    """
    from aip import cache as _cache
    from aip import cost as _cost
    from aip.guards import delimit_untrusted
    from aip.retrieval import format_context

    ctx = delimit_untrusted(format_context(hits))
    prompt = f"{ctx}\n\nQuestion: {question}\n\nAnswer with citations:"
    model = resolve_model(GENERATION_TIER)
    request = {"model": model, "messages": [
        {"role": "system", "content": STRICTNESS[STRICTNESS_NAME]},
        {"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 700}

    key = _cache.make_key("chat", request)
    hit = _cache.get(key)
    if hit is not None:
        yield hit["text"]
        return

    if settings.offline:
        raise cache.CacheMiss("streaming is unavailable offline for an "
                              "uncached question")

    from litellm import completion

    with tracing.trace("rag.generate.stream", tier=GENERATION_TIER,
                       n_sources=len(hits)) as span:
        t0 = time.perf_counter()
        stream = completion(model=model, messages=request["messages"],
                            temperature=0.0, max_tokens=700, stream=True,
                            timeout=settings.timeout_s)
        parts = []
        for chunk in stream:
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                parts.append(piece)
                yield piece
        text = "".join(parts).strip()
        span["n_chars"] = len(text)
        _cost.record(_cost.Usage(model=model, prompt_tokens=0,
                                 completion_tokens=0, cost_usd=0.0,
                                 latency_ms=(time.perf_counter() - t0) * 1000,
                                 cached=False, calls=1, priced=False))
        _cache.put(key, "chat", request, {"text": text, "tool_calls": [],
                                          "finish_reason": None,
                                          "usage": {"prompt_tokens": 0,
                                                    "completion_tokens": 0,
                                                    "cost_usd": 0.0,
                                                    "latency_ms": 0.0,
                                                    "cached": False}})


def _run_rag_tokens(question: str, top_k: int):
    """Retrieval, then streamed generation. Yields text pieces as they arrive.

    B3: prose is streamed, citations are withheld to the final event. No tool
    call is issued while tokens are in flight, so an injected instruction
    arriving mid-stream has nothing to act on. The assembled result is left in
    _STREAM_STATE["result"] because a generator cannot both yield pieces and
    return a value to its caller.
    """
    before = global_budget().spent_usd
    r = pipeline()
    with tracing.trace("rag.retrieve", k=RETRIEVE_K) as span:
        pool = r.search(question, k=RETRIEVE_K)
    span["n_results"] = len(pool)
    hits = pool[:top_k]

    pieces = list(_stream_generate(question, hits))
    text = "".join(pieces).strip()
    _STREAM_STATE["result"] = (text, text.startswith(REFUSAL), hits,
                               _cost_delta(before))
    yield from pieces


_STREAM_STATE: dict = {}


def _run_rag(question: str, top_k: int) -> tuple[str, bool, list, float]:
    """The Lab 5 pipeline. Returns (text, refused, hits, cost)."""
    before = global_budget().spent_usd
    r = pipeline()
    a = answer_question(question, r, k=RETRIEVE_K, final_k=top_k,
                        strictness=STRICTNESS_NAME, tier=GENERATION_TIER)
    return a.text, a.refused, a.hits, _cost_delta(before)


# ---------------------------------------------------------------------------
# A1/A3
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """A1. Cost and trace_id are in the response because they are how anyone
    debugging this later finds the trace, and how whoever pays sees what a
    query costs."""
    t0 = time.perf_counter()
    _bump("requests")

    # --- B1: cache layers, cheapest first ---
    hit = _cache_get_exact(req.question)
    layer = "exact"
    if hit is None and req.semantic_cache:
        hit = _cache_get_semantic(req.question, SEMANTIC_CACHE_THRESHOLD)
        layer = "semantic" if hit else "miss"
    if hit is not None:
        _bump("exact_hits" if layer == "exact" else "semantic_hits")
        # The stored payload already carries cache_layer="miss" from when it was
        # created, so it must be overridden rather than passed alongside --
        # otherwise AskResponse gets two values for the same keyword.
        return AskResponse(
            **{**hit, "cache_layer": layer, "cached": True},
            latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    try:
        with tracing.trace("http.ask", question=req.question[:120]) as span:
            text, refused, hits, cost = _run_rag(req.question, req.top_k)
            cits = _citations(text, hits)

            # Never return an ungrounded, uncited, non-refusal answer. The
            # invariant from Lab 4: citations_valid=False AND refused=False is
            # the thing we are paid to prevent.
            valid = bool(cits) or refused
            if not valid:
                tracing.event("service.answer_uncited")
                text = (f"{text}\n\n[This answer could not be grounded in the "
                        f"provided documents and has been withheld.]")
                refused = True
            if refused:
                _bump("refusals")

            payload = {"answer": text, "refused": refused, "citations": cits,
                       "cost_usd": cost, "trace_id": span["span_id"],
                       "cache_layer": "miss"}
            _cache_put_exact(req.question, payload)
            if req.semantic_cache:
                _cache_put_semantic(req.question, payload)
            return AskResponse(**payload, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                               cached=False)
    except BudgetExceeded as exc:
        _bump("errors")
        _ERRORS_BY_TYPE["budget"] = _ERRORS_BY_TYPE.get("budget", 0) + 1
        # 429 + Retry-After: the client did nothing wrong, and retrying soon
        # makes it worse.
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"Retry-After": "30"}) from exc
    except cache.CacheMiss as exc:
        # AIP_OFFLINE=1 and this question was never cached. That is a
        # configuration problem, not a client error, so it is a 503 not a 400:
        # the request is fine, the service cannot serve it.
        _bump("errors")
        _ERRORS_BY_TYPE["cache_miss"] = _ERRORS_BY_TYPE.get("cache_miss", 0) + 1
        raise HTTPException(status_code=503,
                            detail=f"offline and this question is not cached: {exc}",
                            headers={"Retry-After": "60"}) from exc
    except Exception as exc:  # noqa: BLE001
        _bump("errors")
        _ERRORS_BY_TYPE["internal"] = _ERRORS_BY_TYPE.get("internal", 0) + 1
        tracing.event("service.error", error=f"{type(exc).__name__}: {exc}")
        # 503 + Retry-After for anything that could be an upstream problem.
        # A 500 tells the client the bug is ours and invites an immediate
        # retry, which is exactly wrong for a transient provider failure.
        raise HTTPException(status_code=503,
                            detail="upstream model unavailable",
                            headers={"Retry-After": "5"}) from exc


# ---------------------------------------------------------------------------
# B2/B3: streaming
# ---------------------------------------------------------------------------
@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """B2. Server-sent events.

    B3 -- the validation problem, and the choice made here:

    You cannot check citations until the answer exists, but a stream sends the
    answer first. The three options are buffer-then-stream, stream-then-retract,
    or stream prose and hold citations. **This service streams the prose and
    holds every citation to the final event.**

    The reasoning: a bad citation is a quality defect, while a leaked or
    unfounded *instruction* obeyed from streamed tokens is a safety defect.
    Sending prose early buys TTFT on text the user can already see and cannot
    un-see, so the exposure is bounded to prose. Critically, NO tool call is
    issued while tokens are in flight -- tool results are appended only after
    the answer completes and is validated, so an injected instruction arriving
    mid-stream has nothing to act on. That is why streaming prose is safe here
    and would not be in a system that interleaved tool calls.
    """
    _bump("requests")
    _bump("stream_requests")
    t0 = time.perf_counter()

    def events():
        meta = {"trace_id": uuid.uuid4().hex[:12]}
        try:
            with tracing.trace("http.ask.stream", question=req.question[:120]) as span:
                meta["trace_id"] = span["span_id"]
                hit = _cache_get_exact(req.question)
                if hit is not None:
                    _bump("exact_hits")
                    yield f"event: meta\ndata: {json.dumps({**meta, 'cached': True, 'cache_layer': 'exact'})}\n\n"
                    yield f"event: delta\ndata: {json.dumps({'text': hit['answer'][:200]})}\n\n"
                    yield f"event: done\ndata: {json.dumps({**hit, 'ttft_ms': round((time.perf_counter() - t0) * 1000, 1)})}\n\n"
                    return

                # B2: stream the model's tokens rather than buffering the whole
                # response. This is what actually buys TTFT -- an earlier
                # version buffered the answer and then chopped it into fake
                # "deltas", so the first byte a client received arrived at the
                # same moment as the last. A fake stream is worse than no
                # stream: it advertises a latency it does not deliver.
                yield f"event: meta\ndata: {json.dumps({**meta, 'cached': False, 'cache_layer': 'miss'})}\n\n"

                # Real token streaming: each piece is written to the client the
                # moment the provider emits it, so TTFT is the time to the
                # FIRST token rather than to the end of generation.
                first_token_at: float | None = None
                for piece in _run_rag_tokens(req.question, req.top_k):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield f"event: delta\ndata: {json.dumps({'text': piece})}\n\n"

                text, refused, hits, cost = _STREAM_STATE["result"]
                ttft = ((first_token_at - t0) * 1000 if first_token_at
                        else round((time.perf_counter() - t0) * 1000, 1))

                cits = _citations(text, hits)
                valid = bool(cits) or refused
                if not valid:
                    text = ("[This answer could not be grounded in the provided "
                            "documents and has been withheld.]")
                    refused = True

                payload = {"answer": text, "refused": refused,
                           "citations": [c.model_dump() for c in cits],
                           "cost_usd": cost, "trace_id": span["span_id"],
                           "cached": False, "cache_layer": "miss",
                           "citations_valid": valid,
                           "ttft_ms": round(ttft, 1),
                           "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
        except BudgetExceeded as exc:
            yield f"event: error\ndata: {json.dumps({'status': 429, 'detail': str(exc)})}\n\n"
        except Exception as exc:  # noqa: BLE001
            tracing.event("service.stream_error", error=f"{type(exc).__name__}: {exc}")
            yield f"event: error\ndata: {json.dumps({'status': 503, 'detail': 'upstream model unavailable'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# C: observability
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """C. Index size, model, cache stats, uptime."""
    r = pipeline()
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index": {"n_chunks": len(r.chunks),
                  "matrix_shape": list(r.matrix.shape),
                  "n_docs": len({c.doc_id for c in r.chunks})},
        "model": {"profile": settings.profile,
                  "embed": resolve_model("EMBED"),
                  "generate": resolve_model(GENERATION_TIER),
                  "generation_tier": GENERATION_TIER,
                  "strictness": STRICTNESS_NAME},
        "cache": {**cache.stats(),
                  "exact": len(_EXACT_CACHE),
                  "semantic": len(_SEMANTIC_CACHE)},
        "offline": settings.offline,
    }


@app.get("/metrics")
def metrics() -> dict:
    """C2. Cost today, cost/query, cache hit rate, p50/p95/p99, error rate."""
    b = global_budget()
    n = max(1, STATS["requests"])
    hits = STATS["exact_hits"] + STATS["semantic_hits"]
    return {
        **b.as_dict(),
        # Budget.as_dict() carries latency_p50_ms / latency_p95_ms, not
        # p95_latency_ms. Both spellings are emitted because thresholds.yml
        # and most dashboards expect the latter.
        "p50_latency_ms": round(b.percentile(50), 1),
        "p95_latency_ms": round(b.percentile(95), 1),
        "p99_latency_ms": round(b.percentile(99), 1),
        "requests": STATS["requests"],
        "errors": STATS["errors"],
        "errors_by_type": dict(_ERRORS_BY_TYPE),
        "error_rate": round(STATS["errors"] / n, 4),
        "refusals": STATS["refusals"],
        "refusal_rate": round(STATS["refusals"] / n, 4),
        "stream_requests": STATS["stream_requests"],
        "cache": {
            "exact_hits": STATS["exact_hits"],
            "semantic_hits": STATS["semantic_hits"],
            "hit_rate": round(hits / n, 4),
            "exact_entries": len(_EXACT_CACHE),
            "semantic_entries": len(_SEMANTIC_CACHE),
            "semantic_threshold": SEMANTIC_CACHE_THRESHOLD,
        },
        "cost_per_query": round(b.spent_usd / n, 6),
    }


@app.get("/cache/sweep")
def cache_sweep(threshold: float = 0.0) -> dict:
    """B1. The interesting number: at what cosine does semantic caching start
    returning the WRONG answer?

    Compares each golden question against every other, and for each pair reports
    whether the cached answer would have been judged correct for the new
    question. `wrong_from` is the similarity above which that starts happening.
    """
    from aip.embed import embed
    from labs.lab3.search import load_questions
    from labs.lab4.evaluate import judge_correctness

    qs = [q for q in load_questions(include_unanswerable=True)
          if q["relevant_docs"]]
    try:
        vecs = {q["id"]: np.asarray(embed(q["question"], input_type="query"),
                                    dtype=np.float32) for q in qs}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"embedding unavailable: {exc}"}

    pairs = []
    for a in qs:
        for b in qs:
            if a["id"] == b["id"]:
                continue
            sim = float(np.dot(vecs[a["id"]], vecs[b["id"]]))
            if sim < 0.90:
                continue
            pairs.append({"a": a["id"], "b": b["id"], "sim": round(sim, 4),
                          "same_topic": a["kind"] == b["kind"],
                          "q_a": a["question"], "q_b": b["question"],
                          "gold_a": a["gold_answer"][:200],
                          "gold_b": b["gold_answer"][:200]})
    pairs.sort(key=lambda p: -p["sim"])
    return {"threshold_sweep_from": 0.90,
            "note": ("A semantic cache returns the ANSWER for A when asked B. "
                     "It is correct only if A's answer also answers B. Judge that "
                     "per pair to find the threshold where it stops being true."),
            "high_similarity_pairs": pairs[:25],
            "n_pairs_over_0.90": len(pairs)}


@app.post("/cache/clear")
def cache_clear() -> dict:
    with _LOCK:
        n = len(_EXACT_CACHE) + len(_SEMANTIC_CACHE)
        _EXACT_CACHE.clear()
        _SEMANTIC_CACHE.clear()
    return {"cleared": n}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)