# Aurora Policy Assistant — Evaluation Report

**Name:** ______  **Partner:** ______  **Date:** ______
**Model:** `gemini-3.5-flash-lite` (generation), `gemini-embedding-001` (retrieval)

Every number here is reproducible from committed artefacts:
`python labs/lab7/gate.py` → `reports/gate_metrics.json`,
`python labs/lab7/semantic_cache.py` → `reports/lab7_semantic_cache.json`,
`python labs/lab5/diagnose.py` → `reports/lab5_diagnosis.json`.

---

## 1. What it does

Aurora's policy assistant answers questions about insurance policy documents and
about a customer's own policy record. It searches 30 policy documents, finds the
passages that matter, and writes an answer where **every sentence carries a
citation** to the document it came from. Click a citation and you see the actual
passage. When the documents do not cover a question, it says so rather than
guessing, and that refusal is treated as a correct answer rather than a failure.

It can also look up a customer's own policy and compute premiums, and it is
built so that a prompt-injection attack cannot move money: privileged actions sit
behind an allowlist, argument validation, and human confirmation.

---

## 2. How well it works

45 test questions (40 answerable, 5 deliberately unanswerable). Judged by
`labs/lab4/evaluate.py`'s rubrics, on the configuration that ships.

| Metric | Measured | Target | |
|---|---|---|---|
| correctness (0–1 normalised) | **0.775** | ≥ 0.75 | pass |
| faithfulness | **0.911** | ≥ 0.90 | pass |
| citation validity | **1.000** | ≥ 0.98 | pass |
| refusal recall | **1.000** | ≥ 0.80 | pass |
| refusal precision | **1.000** | ≥ 0.75 | pass |
| hit rate @ 5 (retrieval only) | **1.000** | ≥ 0.85 | pass |
| cost per query | **$0.00047** | ≤ $0.01 | pass |
| p95 latency (uncached) | **3651 ms** | ≤ 6000 ms | pass |
| p95 latency (cached) | **3 ms** | ≤ 800 ms | pass |
| TTFT (streaming) | **fails, 1500 ms target** | ≤ 1500 ms | **FAIL** |

Citation validity of 1.000 is the load-bearing number: it means the system never
returned a confident answer with a citation pointing at nothing. That invariant
is enforced in code, not requested in a prompt.

---

## 3. Where it fails

17 of 45 questions were imperfect under an earlier configuration. After the
changes described below the failure set is 11 of 40 answerable, classified by
`labs/lab5/diagnose.py`:

| Failure mode | n | What it means |
|---|---|---|
| **6 — generation** | 8 | The right passages were in the context and the answer was still wrong |
| **4 — ranking** | 3 | The needed passage was in the top 30 but crowded out of the final 8 |

Two specific and uncomfortable findings:

**It produces numbers no source supports.** Lab 6's red team wrote a poisoned
document claiming group-corporate claims have a 90-day window. The model
*rejected* that — and then answered **60 days**, a figure it synthesised by
combining a number from `claims-process.md` with a question from
`claims-timelines.md`. Neither document states a 60-day submission window; the
real answer is 30. Every one of the five injection defences held. The failure is
not injection, it is arithmetic across sources.

**Under-answering cost more than over-answering.** Lab 5 added a completeness
rule after finding 9 of 12 failures were answers that were right about the
headline and silent about the exception ("maternity not covered on Bronze" —
never mentioning that Silver, Gold and Platinum *do* cover it). Correctness did
not move: 1.500 → 1.500. Four questions improved and four previously-passing
questions broke, including three new wrongful refusals. **Lab 5's conclusion,
reproduced here, is that this fix does not ship.**

---

## 4. What it costs

| | |
|---|---|
| per query | **$0.00047** |
| per 1,000 queries | **$0.47** |
| at 10,000 queries/day | **$1,712 / year** |
| at 100,000 queries/day | $17,100 / year |

Roughly 60× under the $0.01/query budget. Moving generation from
`gemini-3.7-flash` to `gemini-3.5-flash-lite` was the main lever — and it cost
nothing in quality (correctness 0.762 → 0.786, faithfulness identical at 0.905).
The 2.9 MB index is built once at startup and then costs nothing per query.

---

## 5. How fast it is

Latency budget, measured over 42 uncached requests (B4):

```
stage                    p50        p95
──────────────────────────────────────────
embed query            ~1046 ms   ~1338 ms
retrieve (dense)          3 ms        3 ms
generate               1515 ms     3169 ms
validate / repair         ~0 ms       ~0 ms
──────────────────────────────────────────
total, uncached        2072 ms     3651 ms
total, cached             3 ms        3 ms
```

**Generation is the stage to optimise first, and it is 98% of the latency.**
Retrieval is 3 ms. No amount of reranking, re-chunking or hybrid retrieval would
have moved this number, because the time is not spent finding the passage.

The TTFT target is missed, and the cause is measured rather than assumed. The
provider streams correctly when called directly (first token 766–1166 ms, 4–5
chunks). The dominant cost is the **query embedding**: a separate network call,
uncached on every new question, at ~1046 ms median and 4936 ms worst observed.
TTFT ≈ retrieval + first token, so the fix is a smaller or local embedding model,
or a lexical pre-filter that skips embedding for obvious matches. Not a
generation problem.

---

## 6. What it is **not** safe for

This is the section that matters, and the answers are specific to *this* system.

**Do not use it to make a coverage decision without human review.** The system
answers correctly 77.5% of the time. The failures are not evenly distributed —
they cluster in multi-hop questions ("my mother is 63 and I want to add her,
which plans allow it and what changes?") where the answer depends on combining
two or three facts across documents. A wrong answer there is a customer told
they are covered when they are not. Citation validity of 1.000 means the
citations are real; it does **not** mean the conclusion drawn from them is right.

**Do not treat a citation as proof.** The citations are mechanically verified to
exist. The *inference* across them is not verified at all. The 60-day case above
is the failure mode: valid sources, invalid arithmetic, no citation error to catch.

**Do not rely on it for arithmetic or for anything with consequences.** Premium
calculation delegates to a deterministic tool precisely because models are
unreliable at arithmetic. The same discipline should apply to any future
computation: constrain it, do not request it.

**It is not hardened against a determined attacker.** Lab 6 measured 17/17
attacks blocked *with no guards at all* on this model, which means the suite is
too easy rather than the system being safe. One written attack survives all five
defence layers (by causing a cross-document hallucination rather than by obeying
an instruction). The safety argument is not "attacks are blocked" — it is that
`issue_refund` sits behind an allowlist, a schema, and a human, so an injected
instruction cannot move money.

**The TTFT target is not met**, so it is not suitable for a chat-style
interface where perceived speed dominates satisfaction, despite the streaming
endpoint existing.

---

## 7. What I would do next, ranked

**1. Make wrong numbers detectable, not preventable.** Expected value: high.
Require every numeric claim in an answer to cite the chunk that supplies it, and
reject uncited numbers. This catches the 60-day class of failure — a *constraint*
enforced in code, which holds whether or not the model is compromised. It is
also the only fix that addresses Lab 6's surviving attack, since no injection
filter can see a number the model invented itself.

**2. Fix TTFT with a smaller or local embedding model.** Expected value: medium.
TTFT ≈ query-embedding latency + first token, and the embedding is the larger
term at ~1046 ms. Expected TTFT ~900–1200 ms, inside target. Cheap, low risk, no
retrieval-quality change if the same embedding model family is kept.

**3. Address the 8 generation failures with decomposition, not prompt words.**
Expected value: medium. 5 of the 8 are multi-hop. Lab 5 established that
prompt-level fixes move the number zero; splitting a multi-hop question into
sub-questions and retrieving per hop is a structural change with a mechanism
behind it.

**Not doing:** relaxing the refusal threshold. It has already been relaxed once
(Lab 5 measured −0.050 correctness and a fall in refusal precision when it was),
refusal precision is currently 1.000, and there is nothing to buy.

---

## Appendix — how to reproduce

```bash
python scripts/warm_cache.py --verify     # offline replay works?
AIP_OFFLINE=1 python labs/lab7/gate.py     # all 8 metrics, no API key
python labs/lab5/diagnose.py --pareto      # failure classification
python labs/lab7/semantic_cache.py         # B1 threshold measurement
uvicorn labs.lab7.service:app --port 8000  # the service
curl localhost:8000/health                 # index size, model, cache
curl localhost:8000/metrics                # cost, latency, error rate
```

**A caveat on the CI gate's own numbers.** Under `AIP_OFFLINE=1` the committed
cache is replayed, so the gate's cost reads $0.0000 and its latency is
wall-clock around a cache hit. Those are honest measurements of *this machine
replaying a cache* and **not** of the deployed service. The deployed cost and
latency in sections 4 and 5 come from online runs and are the ones to quote. The
gate's job is to catch regressions between commits, which replay is good at.