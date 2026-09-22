#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# LiteLLM logs a DeprecationWarning per call on modern Gemini; 1260 rerank
# calls should not flood the output.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cost  # noqa: E402
from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (  # noqa: E402
    Bm25Retriever,
    ChromaRetriever,
    CrossEncoderReranker,
    DenseRetriever,
    Hit,
    HybridRetriever,
    LLMReranker,
    Retriever,
)

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


class ParallelLLMReranker(LLMReranker):
    """LLM reranker with the per-hit calls parallelised.

    aip's LLMReranker scores the k=30 passages sequentially, which the
    runsheet flags as the obvious bottleneck. The cache is thread-safe
    (a process-wide lock in aip.cache), so a bounded pool is safe here.
    """

    def __init__(self, tier: str = "SMALL", workers: int = 16):
        super().__init__(tier=tier)
        self._workers = workers

    def rerank(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
        from concurrent.futures import ThreadPoolExecutor

        from aip.llm import chat

        if not hits:
            return []

        def score_one(h: Hit) -> float:
            out = chat(
                self.PROMPT.format(q=query, p=h.text[:1500]),
                tier=self.tier, max_tokens=8, temperature=0.0,
            )
            m = re.search(r"\d+(?:\.\d+)?", out)
            return float(m.group()) if m else 0.0

        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            scores = list(ex.map(score_one, hits))
        scored = sorted(zip(scores, hits), key=lambda t: -t[0])[:k]
        return [Hit(h.chunk, float(s), "llm_reranked", i)
                for i, (s, h) in enumerate(scored)]


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """TODO A1-A3.

    A1: all four strategies at size=800.
    A2: the winner at sizes 400 / 800 / 1600. Plot or tabulate the curve.
    A3: markdown WITH and WITHOUT the '[heading > path]' prefix.
        (Strip it with a list comprehension over the chunks -- do not modify
         aip/chunking.py; other labs depend on it.)

    Report chunk count and index build time alongside quality. A configuration
    that is 1 point better and takes 4x as long to build is a real trade-off.
    """
    # """A1-A3."""
    corpus, questions = load_corpus(), load_questions()

    # ---- A1: all four strategies at size=800 ----
    a1_results = {}
    a1_meta = {}  # chunk count / build time per strategy
    a1_chunks = {}  # chunks per strategy, kept for A4

    for strat in STRATEGIES:
        t0 = time.perf_counter()
        chunks = build_chunks(corpus, strat, 800, overlap=150)
        a1_chunks[strat] = chunks
        r = DenseRetriever(chunks)
        build_ms = (time.perf_counter() - t0) * 1000

        m = evaluate(r, questions)
        name = f"{strat}-800"
        a1_results[name] = m
        a1_meta[name] = (len(chunks), build_ms)

    print("=== A1: strategy comparison @ size=800 ===")
    print(table(a1_results))
    print()
    for name, (n_chunks, build_ms) in a1_meta.items():
        print(f"{name:<20} chunks={n_chunks:<6} build={build_ms:.1f}ms")
    print()
    for name, m in a1_results.items():
        print(f"-- {name} by kind (mrr) --")
        print(kind_table(m, col="mrr"))
        print()

    # pick winner by mrr (hit_rate@5 is saturated, don't use it here)
    winner = max(a1_results, key=lambda n: a1_results[n]["mrr"]).split("-")[0]
    print(f"winner strategy: {winner}\n")

    # ---- A4: one question where chunking is clearly the failure ----
    win_name = f"{winner}-800"
    perq = a1_results[win_name]["_per_question_mrr"]
    worst_qid, worst_mrr = min(perq.items(), key=lambda kv: kv[1])
    qrow = next(q for q in questions if q["id"] == worst_qid)
    relevant = set(qrow["relevant_docs"])
    relevant_chunks = [c for c in a1_chunks[winner] if c.doc_id in relevant]
    r = DenseRetriever(a1_chunks[winner])
    retrieved_top = r.search(qrow["question"], k=5)

    print("=== A4: worst-ranking question for the winning chunker ===")
    print(f"question: {qrow['question']}")
    print(f"mrr: {worst_mrr:.4f}  relevant docs: {sorted(relevant)}")
    if relevant_chunks:
        print("chunk that SHOULD have matched:")
        print(f"  [{relevant_chunks[0].chunk_id}] {relevant_chunks[0].text[:220]!r}")
    print("chunks that DID rank:")
    for h in retrieved_top:
        mark = "*" if h.doc_id in relevant else " "
        print(f" {mark}[{h.chunk.chunk_id}] {h.text[:120]!r}")
    print()

    # ---- A2: winner at sizes 400 / 800 / 1600 ----
    a2_results = {}
    a2_meta = {}

    for size in (400, 800, 1600):
        t0 = time.perf_counter()
        chunks = build_chunks(corpus, winner, size, overlap=150)
        r = DenseRetriever(chunks)
        build_ms = (time.perf_counter() - t0) * 1000

        m = evaluate(r, questions)
        name = f"{winner}-{size}"
        a2_results[name] = m
        a2_meta[name] = (len(chunks), build_ms)

    print("=== A2: size sweep on winner ===")
    print(table(a2_results))
    print()
    for name, (n_chunks, build_ms) in a2_meta.items():
        print(f"{name:<20} chunks={n_chunks:<6} build={build_ms:.1f}ms")
    print()
    
    # ---- A3: markdown WITH vs WITHOUT the '[heading > path]' prefix ----
    # only meaningful if 'markdown' is one of the strategies
    if "markdown" in STRATEGIES:
        t0 = time.perf_counter()
        md_chunks = build_chunks(corpus, "markdown", 800, overlap=150)
        with_build_ms = (time.perf_counter() - t0) * 1000

        # strip the '[heading > path]' prefix without touching aip/chunking.py
        stripped_chunks = [
            Chunk(**{**c.__dict__, "text": c.text.split("]", 1)[-1].lstrip()})
            if c.text.startswith("[") else c
            for c in md_chunks
        ]

        r_with = DenseRetriever(md_chunks)
        m_with = evaluate(r_with, questions)

        t0 = time.perf_counter()
        r_without = DenseRetriever(stripped_chunks)
        without_build_ms = (time.perf_counter() - t0) * 1000
        m_without = evaluate(r_without, questions)

        print("=== A3: markdown prefix ablation ===")
        print(table({
            "markdown-with-prefix": m_with,
            "markdown-without-prefix": m_without,
        }))
        print()
        print(f"with-prefix    chunks={len(md_chunks):<6} build={with_build_ms:.1f}ms")
        print(f"without-prefix chunks={len(stripped_chunks):<6} build={without_build_ms:.1f}ms")
    else:
        print("no 'markdown' strategy found in STRATEGIES; skipping A3")

def sweep_retrieval() -> None:
    """TODO B1-B4.

    B1: dense / bm25 / hybrid on your best chunking.
    B2: print kind_table(m, col='mrr') for each, and pull out Q44 and Q41
        individually from metrics['_per_question_mrr'].

        USE MRR, NOT hit_rate@5. Every retriever here scores 0.93-0.98 on
        hit_rate@5, so it is saturated and shows you nothing -- which is why
        kind_table() and metrics['_per_question'] both default to it. That
        default is the trap, and noticing it is part of the lab.

    B3: RRF k in {10, 30, 60, 100} -- HybridRetriever(..., rrf_k=k).
    B4: unequal fusion weights -- HybridRetriever(..., weights=[2.0, 1.0]).
    """
    corpus, questions = load_corpus(), load_questions()
    questions_by_id = {q["id"]: q for q in questions}

    # Use your best chunking config from A1/A2 -- the A2 winner on this corpus.
    BEST_STRATEGY = "markdown"
    BEST_SIZE = 400

    chunks = build_chunks(corpus, BEST_STRATEGY, BEST_SIZE, overlap=150)

    # ---- B1: dense / bm25 / hybrid ----
    dense_r = DenseRetriever(chunks)
    bm25_r = Bm25Retriever(chunks)

    retrievers = {
        "dense": dense_r,
        "bm25": bm25_r,
        "hybrid": HybridRetriever([dense_r, bm25_r]),
    }

    b1_results = {}
    for name, r in retrievers.items():
        m = evaluate(r, questions)
        b1_results[name] = m

    print("=== B1: dense vs bm25 vs hybrid ===")
    print(table(b1_results))
    print()

    # ---- B2: kind_table on MRR (not the saturated default), + Q44/Q41 ----
    for name, m in b1_results.items():
        print(f"-- {name} by kind (mrr) --")
        print(kind_table(m, col="mrr"))
        print()

    print("=== B2: Q44 / Q41 per retriever (mrr) ===")
    for name, m in b1_results.items():
        q44 = m["_per_question_mrr"].get("Q44")
        q41 = m["_per_question_mrr"].get("Q41")
        print(f"{name:<10} Q44={q44!r:>8}  Q41={q41!r:>8}")
    print()

    # ---- B3: RRF k sweep ----
    b3_results = {}
    for rrf_k in (10, 30, 60, 100):
        r = HybridRetriever([dense_r, bm25_r], rrf_k=rrf_k)
        m = evaluate(r, questions)
        b3_results[f"hybrid-rrf{rrf_k}"] = m

    print("=== B3: RRF k sweep ===")
    print(table(b3_results))
    print()
    for name, m in b3_results.items():
        print(f"-- {name} by kind (mrr) --")
        print(kind_table(m, col="mrr"))
        print()

    # ---- B4: unequal fusion weights ----
    r_weighted = HybridRetriever([dense_r, bm25_r], weights=[2.0, 1.0])
    m_weighted = evaluate(r_weighted, questions)

    print("=== B4: weighted fusion [2.0, 1.0] vs equal-weight hybrid ===")
    print(table({
        "hybrid-equal": b1_results["hybrid"],
        "hybrid-weighted-2:1": m_weighted,
    }))
    print()
    print("-- hybrid-weighted-2:1 by kind (mrr) --")
    print(kind_table(m_weighted, col="mrr"))


def sweep_rerank() -> None:
    """TODO C1-C4.

    Retrieve k=30, rerank to 5: evaluate(r, questions, k=30, reranker=rr,
    final_k=5).

    C1: CrossEncoderReranker. First run downloads ~90 MB.
    C2: LLMReranker -- report cost as well as latency.
    C3: the decision table, a   nd TWO different deployment answers
        (interactive search box vs overnight batch). They should differ.
    C4: find a query reranking made worse, using
        metrics['_per_question_mrr'] before and after.
    """
    corpus, questions = load_corpus(), load_questions()
    questions_by_id = {q["id"]: q for q in questions}

    BEST_STRATEGY = "markdown"
    BEST_SIZE = 400

    chunks = build_chunks(corpus, BEST_STRATEGY, BEST_SIZE, overlap=150)
    base_retriever = DenseRetriever(chunks)  # or hybrid, if that won Part B

    # baseline (no reranker) at k=30 -> effectively unreranked top-30,
    # used as the "before" comparison for C4
    m_base = evaluate(base_retriever, questions, k=30, final_k=5)

    # ---- C1: CrossEncoderReranker ----
    ce = CrossEncoderReranker()
    t0 = time.perf_counter()
    m_ce = evaluate(base_retriever, questions, k=30, reranker=ce, final_k=5)
    ce_latency = m_ce["latency_p95_ms"]

    # ---- C2: LLMReranker (parallelised across its 30 per-query calls) ----
    llm = ParallelLLMReranker()
    ledger = cost.global_budget()
    spend_before = ledger.spent_usd
    calls_before = ledger.calls
    m_llm = evaluate(base_retriever, questions, k=30, reranker=llm, final_k=5)
    llm_latency = m_llm["latency_p95_ms"]
    llm_cost = round(ledger.spent_usd - spend_before, 6)
    llm_calls = ledger.calls - calls_before

    print("=== C1/C2: no-rerank vs cross-encoder vs LLM ===")
    print(table({
        "no-rerank-top30": m_base,
        "cross-encoder": m_ce,
        "llm-rerank": m_llm,
    }))
    print(f"\ncross-encoder p95 latency: {ce_latency:.1f}ms")
    print(f"llm-rerank    p95 latency: {llm_latency:.1f}ms")
    print(f"llm-rerank    cost:        ${llm_cost:.6f} ({llm_calls} calls)")
    print()

    # ---- C3: decision table + two deployment answers ----
    print("=== C3: decision table ===")
    print(f"{'approach':<16}{'quality(mrr)':>14}{'p95 latency':>14}{'cost':>10}")
    print(f"{'no-rerank':<16}{m_base['mrr']:>14.4f}{m_base['latency_p95_ms']:>14.1f}{'—':>10}")
    print(f"{'cross-encoder':<16}{m_ce['mrr']:>14.4f}{ce_latency:>14.1f}{'~free':>10}")
    print(f"{'llm-rerank':<16}{m_llm['mrr']:>14.4f}{llm_latency:>14.1f}{str(llm_cost):>10}")
    print("""
    Interactive search box: latency is user-facing and directly costs
    engagement, so prefer whichever reranker clears the quality bar at the
    LOWEST p95 latency -- likely cross-encoder, since it runs locally with
    no network round-trip, even if LLM-rerank scores marginally higher.

    Overnight batch: latency is irrelevant (nothing is waiting on it), so
    prefer whichever reranker gives the HIGHEST quality regardless of
    latency, accepting LLM-rerank's per-query cost since it's amortized
    over a batch window with no user waiting on it.
    """)

    # ---- C4: find a query reranking made WORSE ----
    before = m_base["_per_question_mrr"]
    after = m_ce["_per_question_mrr"]  # or m_llm, whichever you're diagnosing

    deltas = {qid: after[qid] - before[qid]
              for qid in before if qid in after}
    worst_qid = min(deltas, key=deltas.get)

    print("=== C4: worst regression from reranking ===")
    print(f"question: {questions_by_id[worst_qid]['question']}")
    print(f"mrr before: {before[worst_qid]:.4f}  after: {after[worst_qid]:.4f}"
          f"  (delta {deltas[worst_qid]:+.4f})")


def sweep_index() -> None:
    """D1: ChromaRetriever vs DenseRetriever -- recall gap and latency.
    D2: exact NumPy vs HNSW at three scales (~235 / ~2k / ~7k chunks),
        find the crossover.
    D3: pass status metadata into the chunks and filter at query time.

        Set chunk.meta['status'] = 'archived' if 'ARCHIVED' in doc_id else 'current'
        then ChromaRetriever.search(..., where={"status": "current"}).

        Report hit_rate@1 on Q29/Q30/Q31 before and after (hit_rate@1, not
        @5 -- @5 is saturated here and will hide the whole effect).
    """
    corpus, questions = load_corpus(), load_questions()
    target_ids = ("Q29", "Q30", "Q31")

    BEST_STRATEGY = "markdown"
    BEST_SIZE = 400

    chunks = build_chunks(corpus, BEST_STRATEGY, BEST_SIZE, overlap=150)

    # ---- D1/D2 (small scale): ChromaRetriever vs DenseRetriever ----
    t0 = time.perf_counter()
    dense = DenseRetriever(chunks)
    dense_build_ms = (time.perf_counter() - t0) * 1000
    m_dense = evaluate(dense, questions)

    t0 = time.perf_counter()
    chroma = ChromaRetriever(chunks, reset=True, collection="lab3_d1",
                             path=str(ROOT / ".aip_cache/chroma"))
    chroma_build_ms = (time.perf_counter() - t0) * 1000
    m_chroma = evaluate(chroma, questions)

    print("=== D1/D2: DenseRetriever vs ChromaRetriever ===")
    print(table({"dense": m_dense, "chroma": m_chroma}))
    print(f"\ndense  build: {dense_build_ms:.1f}ms")
    print(f"chroma build: {chroma_build_ms:.1f}ms")
    recall_gap = m_dense["recall@5"] - m_chroma["recall@5"]
    print(f"recall@5 gap (dense - chroma): {recall_gap:+.4f}\n")

    # ---- D2 (scaled): exact NumPy vs HNSW latency, and the crossover ----
    def time_retriever(retriever, qs, k: int = 5, repeats: int = 3) -> dict:
        total: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for q in qs:
                retriever.search(q["question"], k=k)
            total.append((time.perf_counter() - t0) * 1000)
        total.sort()
        n = len(qs)
        return {
            "per_query_p50_ms": total[len(total) // 2] / n,
            "per_query_p95_ms": total[min(len(total) - 1,
                                    int(0.95 * (len(total) - 1)))] / n,
        }

    filler_dir = ROOT / "data/corpus_scaled"
    filler = sorted(filler_dir.glob("*.md")) if filler_dir.exists() else []
    print("=== D2: exact vs HNSW latency across scales ===")
    print(f"{'scale':<12}{'chunks':>8}{'exact p50':>12}{'exact p95':>12}"
          f"{'hnsw p50':>12}{'hnsw p95':>12}")

    # Scales use only embeddings already in the cache from the earlier full run
    # (chroma collection names may not contain '+').
    timings: dict[str, dict] = {}
    for label, extra in (("real", []), ("real+300", filler[:300]),
                         ("real+1000", filler[:1000])):
        docs = {**corpus, **{p.stem: p.read_text(encoding="utf-8") for p in extra}}
        scaled_chunks = build_chunks(docs, BEST_STRATEGY, BEST_SIZE, overlap=150)
        exact = DenseRetriever(scaled_chunks)
        ann = ChromaRetriever(scaled_chunks, reset=True,
                              collection=f"lab3_d2_{label.replace('+', 'p')}",
                              path=str(ROOT / ".aip_cache/chroma"))
        te = time_retriever(exact, questions)
        ta = time_retriever(ann, questions)
        timings[label] = {"chunks": len(scaled_chunks), "exact": te, "hnsw": ta}
        print(f"{label:<12}{len(scaled_chunks):>8}"
              f"{te['per_query_p50_ms']:>12.3f}{te['per_query_p95_ms']:>12.3f}"
              f"{ta['per_query_p50_ms']:>12.3f}{ta['per_query_p95_ms']:>12.3f}")

    crossover = next((lbl for lbl, t in timings.items()
                      if t["hnsw"]["per_query_p50_ms"] < t["exact"]["per_query_p50_ms"]),
                     None)
    if crossover:
        print(f"\ncrossover at '{crossover}' "
              f"({timings[crossover]['chunks']} chunks): HNSW gets faster there")
    else:
        print("\nno crossover in the measured scales: HNSW never beat exact here")
    print()

    # ---- D3: metadata filtering ----
    # find Q29/Q30/Q31's hit_rate@1 BEFORE filtering (unfiltered chroma index)
    def hit_rate_at_1_for(retriever, qs, ids, **search_kwargs):
        out = {}
        for q in qs:
            if q["id"] not in ids:
                continue
            hits = retriever.search(q["question"], k=1, **search_kwargs)
            top_doc = hits[0].doc_id if hits else None
            out[q["id"]] = 1.0 if top_doc in q["relevant_docs"] else 0.0
        return out

    before_hr1 = hit_rate_at_1_for(chroma, questions, target_ids)

    # tag chunks with status metadata, then rebuild the chroma index
    for c in chunks:
        c.meta["status"] = "archived" if "ARCHIVED" in c.doc_id else "current"

    chroma_filtered = ChromaRetriever(chunks, reset=True, collection="lab3_d3",
                                      path=str(ROOT / ".aip_cache/chroma"))

    after_hr1 = {}
    for q in questions:
        if q["id"] not in target_ids:
            continue
        hits = chroma_filtered.search(q["question"], k=1,
                                      where={"status": "current"})
        top_doc = hits[0].doc_id if hits else None
        after_hr1[q["id"]] = 1.0 if top_doc in q["relevant_docs"] else 0.0

    print("=== D3: hit_rate@1 on Q29/Q30/Q31, before vs after status filter ===")
    print(f"{'qid':<8}{'before':>10}{'after':>10}")
    for qid in target_ids:
        print(f"{qid:<8}{before_hr1.get(qid, float('nan')):>10.2f}"
              f"{after_hr1.get(qid, float('nan')):>10.2f}")


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
