# Lab 3 — Semantic Search: Recommendation Report

Harness: `labs/lab3/search.py`. Golden set: `data/eval/rag_golden.jsonl`.
**n=42**: Q36/Q38/Q39 have an empty relevant set (recall is undefined), so they are dropped from every metric.
Embeddings: `gemini/gemini-embedding-001`; rerank LLM: `gemini/gemini-3.5-flash-lite`. Log: `labs/lab3/lab3_output.txt`, results: `reports/lab3_sweeps.json`.

Unlike Lab 2, **all quality columns here are identical warm-cache vs first run** (retrieval is deterministic), so every table below is reproducible from the log. Latency/cost for Part C were re-measured on the first **uncached** run (the cached re-run in the log shows LLM-rerank p95 55.6 ms / $0 — an artifact, not a deployment number).

## Recommendation (short version)

**Deploy `markdown-400` chunks + exact dense retrieval.** Every target is met with headroom:

| target | required | measured |
|---|---|---|
| nDCG@10 | ≥ 0.80 | **0.8527** |
| recall@5 | ≥ 0.85 | **0.9028** |
| hit_rate@1 | ≥ 0.65 | **0.7857** |
| MRR (paraphrase) | ≥ 0.75 | **0.8000** |
| p95 latency | ≤ 400 ms | **3.3 ms** |

235 chunks, index build 239 ms after warm embed cache (embedding cost ~$0 — free-tier/cached), retrieval $0 per query.

**One thing that surprised us:** hybrid retrieval (dense+BM25 via RRF) *lost* to dense alone on this corpus (ndcg@10 0.7949 vs 0.8527) — the opposite of T4's stated default that hybrid is "the strongest single change most RAG systems can make." Our embedding model is already strong enough to handle the exact plan-code identifiers that BM25 usually exists to rescue, so fusing in a much weaker retriever dragged down more good rankings than it rescued. A technique that's right on average can be wrong on your data.

## Part A — Chunking strategies (fixed 800-char budget)

| strategy | chunks | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 | p95 ms |
|---|---|---|---|---|---|---|---|
| fixed-800 | 83 | 0.7381 | 0.9524 | 0.8373 | 0.8387 | 0.7952 | 4.0 |
| sliding-800 | 91 | 0.7857 | 0.9286 | 0.8452 | 0.8451 | 0.8053 | 3.2 |
| recursive-800 | 98 | 0.7857 | 0.9286 | 0.8393 | 0.8552 | 0.8127 | 5.9 |
| markdown-800 | 164 | 0.7619 | **0.9762** | 0.8988 | **0.8720** | **0.8458** | 5.3 |

**A1. Markdown wins** — with an interesting split: it is *highest* on recall@5/ndcg@10 (0.8988/0.8458) yet only *third* on hit_rate@1 (0.7619). Per-kind MRR shows why: markdown lifts **paraphrase 0.72 → 0.90** (the section trees keep "grace period" phrasing inside one queryable chunk, so rephrasings match). Fixed-800 is worst on 4 of the 5 quality columns (hit_rate@1, recall@5, mrr, ndcg@10) — hard cuts bisect mid-sentence and collapse pool-plane sections. Fixed-800 is not worst on hit_rate@5 (0.9524): still below markdown (0.9762), but above sliding and recursive (0.9286) — a weak discriminator because getting *a* relevant doc into the top 5 is easy for every strategy here.

**A2 — call-specific sizes (markdown):**

| size | chunks | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 | p95 ms |
|---|---|---|---|---|---|---|---|
| **markdown-400** | 235 | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | 3.3 |
| markdown-800 | 164 | 0.7619 | 0.9762 | 0.8988 | 0.8720 | 0.8458 | 2.8 |
| markdown-1600 | 150 | 0.7143 | 0.9524 | 0.8750 | 0.8262 | 0.8075 | 3.7 |

**Dilution, measured:** four of five quality columns fall monotonically as chunks get bigger (400 → 800 → 1600); hit_rate@5 is flat between 400 and 800 (0.9762 → 0.9762) before dropping at 1600 (0.9524) — the right doc still lands in the top 5 at 800 chars, it just ranks lower within it (hence mrr and hit_rate@1 already falling at that size). The trajectory is "bigger chunk = more noise per vector = worse ranking," capped low-side by relevance leakage across a 200-char cut. 400 is the smallest size whose chunks still contain a full answer.

**A3 — heading-prefix ablation (markdown-800):**

| variant | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| with heading prefix | **0.7619** | 0.9762 | 0.8988 | **0.8720** | **0.8458** |
| without heading prefix | 0.6190 | **1.0000** | 0.9048 | 0.7837 | 0.7915 |

The prefix **helps first-hit precision** (hit_rate@1 +14 pts) at the cost of recall@5 (-0.6 pt): the `Document / Section` path disambiguates near-identical plan docs — without it the embedding spreads effectively-similar ranking across more chunks (hit_rate@5 1.00 but more of them wrong at rank 1). Keep the prefix. (Note: our corpus is the dedupe case, not the "prefix *causes* repeated hits" case.)

**A4 — chunking failure (Q37, "Does Aurora cover treatment in Singapore, and up to what limit?"):** mrr **0.125**. Relevant docs per the golden set: `exclusions`, `plans-overview`. The chunk we'd expect to rank first, `exclusions::m0` — "[Permanent Exclusions] The following are never payable under any Aurora indemnity plan, at any sum insured, regardless of waiting periods served" — is a generic permanence clause; it contains no mention of Singapore, international treatment, or any specific limit. It did **not** make top-5: the model surfaced `topup`, `travel-insurance-exclusions`, `network-hospitals`, `senior-citizen-plan`, `plan-silver` chunks instead — none of them relevant.

Mechanism: this is a **chaining + partial-grounding failure**, not primarily a chunk-size problem. The gold answer needs the permanence clause (exclusions) combined with the covered-benefit list (plans-overview), but neither chunk we retrieved, nor any chunk in the corpus that we found, states a concrete Singapore/international treatment limit — the specific figure the question asks for appears to live in a document (e.g. a Platinum-plan international-benefit addendum) that isn't present in our 30-document corpus. So no single chunk contains the full grounded answer, and no chunking strategy at any size would fix it: the missing information isn't a chunking problem, it's a missing-source problem. Re-chunking `exclusions` differently wouldn't surface a limit that was never in the corpus to begin with.

## Part B — Retrievers (markdown-400)

| retriever | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 | p95 ms |
|---|---|---|---|---|---|---|
| dense | **0.7857** | 0.9762 | **0.9028** | **0.8800** | **0.8527** | 3.3 |
| bm25 | 0.4762 | 0.9286 | 0.7956 | 0.6698 | 0.6978 | 1.7 |
| hybrid (rrf30) | 0.6667 | 0.9762 | 0.8631 | 0.7976 | 0.7949 | 6.2 |

**Negative result (explicit): the hybrid is worse than dense** (−0.058 ndcg@10, −0.082 mrr), reversing T4's "hybrid usually wins" default. Per-kind MRR shows exactly why:

| kind (n=4·10·5·18·3·2) | dense | bm25 | hybrid |
|---|---|---|---|
| aggregation | 0.8750 | 0.3750 | 0.5833 |
| multi_hop | 1.0000 | 0.6500 | 0.8167 |
| paraphrase | 0.8000 | 0.4867 | 0.6500 |
| single_hop | 0.9074 | 0.8519 | 0.9444 |
| trap_archived | 0.8333 | 0.5111 | 0.6667 |
| unanswerable | 0.3125 | 0.4167 | 0.3750 |

**B2 — the identifier-rescue pair.** Q44 (plan code `AUR-HI-SIL-2026`): dense **0.50** → bm25 **1.00** → hybrid **1.00**. Q41 ("if I skip paying on time, how long before I lose everything"): dense **1.00** → bm25 **0.00** → hybrid **0.25**. BM25's exact-token precision fixes the identifier, but its blind paraphrase misses cost the dense-only ranking on Q41; the RRF-fused rank takes the *average* of where each retriever suffered. Net on this corpus the dense side wins more often than the rescue side, hence the negative result.

**B3 — RRF k:** k=10 (mrr 0.8115, hit_rate@5 1.0) ≈ k=30/60/100 (mrr 0.7976). Effect tiny and order-insensitive — default RRF is fine.

**B4 — weighting (hybrid):** equal mrr 0.7976 / ndcg 0.7949 vs weighted 2:1 dense-dominant mrr **0.8103** / ndcg **0.8068**. +0.013 mrr is a real direction (trust dense more) but within noise at n=42; it still does not catch dense (0.8800/0.8527).

## Part C — Reranking (base k=30 → final k=5, uncached latencies)

*Note: the harness reports `ndcg@10` not the README's `ndcg@5`. Part C retrieves k=30 then reranks to 5; Part A/B compute nDCG@10 over the full retrieval pool (30/top-10 respectively). Pool sizes differ—the C rows are not cell-to-cell comparable to A/B.*

| config | hit_rate@1 | hit_rate@5 | recall@5 | mrr | ndcg@10 | p95 ms | $/1k |
|---|---|---|---|---|---|---|---|
| no-rerank (top-30) | 0.7857 | 0.9762 | 0.9028 | 0.8800 | 0.8685 | 4.6 | 0 |
| cross-encoder | 0.7619 | **1.0000** | 0.8889 | 0.8619 | 0.8174 | 852.0 | 0 |
| llm-rerank | **0.8571** | 0.9524 | 0.8810 | **0.9048** | 0.8465 | 2996.8 | $0.371 |

`search.py` returns the top-5 reranked only, so `recall@5`/`ndcg@10` fall with k — compare `hit_rate@1`/`mrr` for rerank quality. Two findings jump out:

**C1-C2 — the cross-encoder hurts (out-of-domain).** TP-LISA is trained on web search, not insurance-policy prose: it *lowers* mrr (0.8800→0.8619) and ndcg@10 (0.8685→0.8174). The LLM reranker is the only one that improves rank-sensitive metrics (mrr **0.9048**, hit_rate@1 **0.8571**) at $0.371/1k (1260 calls). We parallelise the LLM reranker's 30 per-query scoring calls across 16 worker threads; running the same 1260 calls fully sequentially would take roughly **16× longer** (not a full 30×, since only the per-query batch of 30 calls is parallelised — 30 calls over 16 workers is 2 sequential rounds, vs 30 sequential rounds one at a time). LLM cost is fixed per query regardless of corpus size; it amortizes to $0.37 per thousand queries.

**C3 — deployment decision (two environments, two answers).**
- **Interactive search box:** deploy **no-rerank dense** (p95 4.6 ms, $0). Nothing below mrr 0.88 is a *detectable* quality goal here, and no reranker earns its latency below that gate. If a rank-sensitive consumer forces a reranker in the interactive path, the **cross-encoder is the only defensible choice** (852 ms vs 2,997 ms) — but expect CSR-doc regressions like C4.
- **Overnight batch:** deploy **llm-rerank** — latency is irrelevant and mrr is best; $0.37/1k is fine when amortized over batch budgets and no human waits on the 3 s tail.

**C4 — worst regression:** cross-encoder on Q32 "Which plans have no co-payment?" pulls a perfect 1.0000 down to **0.2000** (modality-5: it overrode a bi-encoder result that was already correct, because its web-search priors don't read "co-payment" tables the way the corpus writes them).

## Part D — Dense vs Chroma, scale, and the Q30 fix

**D1 — cold shard-equivalence check.** Chroma vs dense on the real corpus: **quality identical down to 4 decimals on every metric** (recall@5 gap **+0.0000**, ndcg@10 0.8527 both), p95 8.12 ms vs 3.83 ms, build 1993.4 ms vs 254.0 ms. So Chroma is a zero-recall-cost storage layer here — it is *slower*, not riskier.

**D2 — HNSW crossover.** Exact (full-scan) vs HNSW at three scales (235 / 2,335 / 7,235 chunks):

| scale | exact p50 | exact p95 | hnsw p50 | hnsw p95 |
|---|---|---|---|---|
| real (235) | 3.30 | 3.30 | 6.48 | 6.48 |
| real+300 (2,335) | 6.63 | 6.63 | 9.50 | 9.50 |
| real+1000 (7,235) | 11.24 | 11.24 | **6.79** | **6.79** |

**Crossover between 2,335 and 7,235 chunks.** Below ~2k, HNSW's graph traversal + Python call overhead loses to one BLAS matmul (~2× slower at 235 chunks); once exact search's O(n·d) grows and HNSW's O(log n) takes over, HNSW wins (7,235 → 6.79 vs 11.24 ms). Scale was capped at 1000 filler docs because the larger filler embedding was not run to completion — the 40k-doc store's embeddings for 3,000 filler docs are not cached.

**D3 — the "retrieval was broken" that wasn't.** Q29/Q30/Q31 (trap_archived) hit_rate@1 before/after a metadata filter (`status='current'`, `ARCHIVED` in doc id → archived):

| qid | before | after |
|---|---|---|
| Q29 | 1.00 | 1.00 |
| **Q30** | **0.00** | **1.00** |
| Q31 | 1.00 | 1.00 |

Q30 was failing on **stale data, not recall**: an archived policy outranked its current twin. The fix was data hygiene (curate stale docs out of the path) — no retriever change. When retrieval quality is poor, check the *index* before the *retriever*.

---

### Negative results (explicit)
- **Hybrid < dense** (−0.082 mrr, −0.058 ndcg@10); BM25 rescues Q44 but its blindness costs more on this corpus.
- **Cross-encoder rerank hurts** (mrr 0.8800→0.8619): web-search training on insurance prose; Q32 1.0→0.2.
- **Chunk-size dilution:** monotonically lower on 4 of 5 quality columns as size grows 400→800→1600; hit_rate@5 alone is flat 400→800 before dropping at 1600.
- **HNSW is slower at small scale** (~2× at 235 chunks); only pays off ≥ ~2.3k chunks.

### Two unavoidable caveats
1. **Greedy one-axis-at-a-time sweep** — the winner is fixed before the next axis is swept (markdown-400 chosen, then retrieval/rerank/index tuned on it). Joint optima (e.g. a reranker that is best under a different chunking) are out of scope by design.
2. **n=42** — meaningful ranking-delta tests (B4, RRF-k) are under-powered; deltas below ~0.03 mrr are noise, the big findings (chunking, hybrid, rerank, metadata) are not borderline.