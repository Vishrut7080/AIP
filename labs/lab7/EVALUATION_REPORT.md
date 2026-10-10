# Aurora Policy Assistant — Evaluation Report

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
| refusal precision | **0.833** (5/6) | ≥ 0.75 | pass |
| hit rate @ 5 (retrieval only) | **1.000** | ≥ 0.85 | pass |
| cost per query | **$0.00047** | ≤ $0.01 | pass |
| p95 latency (uncached) | **3651 ms** | ≤ 6000 ms | pass |
| p95 latency (exact cache) | **3 ms** | ≤ 800 ms | pass |
| TTFT (streaming) | **fails, 1500 ms target** | ≤ 1500 ms | **FAIL** |

Citation validity of 1.000 is the load-bearing number: it means the system never
returned a confident answer with a citation pointing at nothing. That invariant
is enforced in code, not requested in a prompt.

**The refusal figures are one run, not a stable measurement.** Refusal precision
is 0.833 (5/6 — in the gate's run, **Q23** is the single wrongful refusal) and
recall is 1.000, but both rest on 5 unanswerable questions, and 3 independent
online repeats of the identical shipped config put precision in [0.833, 0.833]
and recall in [0.800, 1.000]. Treat any single-case difference here as noise.
The three questions this report names as deterministic failures — Q23 and Q17
among them — come from different experiments, and §3 and §7 say which.

**These numbers are also not what Labs 4 and 5 measured.** Those labs ran
`gemini-3.7-flash` (tier `MAIN`); the service ships `gemini-3.5-flash-lite`
(tier `SMALL`). The table above is the shipping configuration. See
`reports/lab7_tier2x2.json` for the full comparison.

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

**Under-answering was the biggest failure class — and Lab 5's fix for it does
replicate, once measured on the model that actually ships.** Lab 5 added a
completeness rule after finding 9 of 12 failures were answers that were right
about the headline and silent about the exception ("maternity not covered on
Bronze" — never mentioning that Silver, Gold and Platinum *do* cover it).

Lab 5 measured this on `gemini-3.7-flash` and concluded the fix did not ship:
correctness was flat at 1.500 → 1.500, faithfulness −0.044, and refusal
precision fell 0.571 → 0.429. **That conclusion does not hold at tier `SMALL`,
which is what the service runs.** Re-running the same 2×2 on the shipping model,
3 independent repeats per cell (`reports/lab7_tier2x2.json`):

| | `lenient` (no rule 7) | `+ rule 7` (shipped) | Δ |
|---|---|---|---|
| correctness | 0.754 ± 0.026 | **0.783 ± 0.007** | **+0.029** |
| refusal precision | 0.698 ± 0.028 | **0.833 ± 0.000** | **+0.135** |
| refusal recall | 0.933 ± 0.116 | **1.000 ± 0.000** | **+0.067** |
| faithfulness | **0.904 ± 0.034** | 0.882 ± 0.026 | −0.022 |

So the rule *is* shipped, and on the shipping model it trades 0.022 of
faithfulness for 0.029 correctness and 0.135 refusal precision. The single
remaining cost is one deterministic wrongful refusal — **Q17** — which appears
in all six online runs of both cells at the shipping tier, so it is a cost of
rule 7 rather than run-to-run variance.

**The general lesson is the one worth keeping:** a fix rejected on the wrong
model is not a rejected fix. Lab 5's experiment was internally valid and its
conclusion was still wrong, because "MAIN" and "SMALL" are different models and
only one of them was ever going to be deployed.

**The semantic cache is unsafe at every threshold tested, so it ships switched
off.** B1 asked for a similarity threshold that makes answering B from A's
cached answer safe. There isn't one. `reports/lab7_semantic_cache.json` scores
30 question pairs by similarity band, asking of each pair whether the answer
cached for A would *also* have been judged correct for B:

| similarity band | pairs | correct for B |
|---|---|---|
| 0.70 – 0.80 | 24 | **0** |
| 0.80 – 0.86 | 3 | **0** |
| 0.86 – 0.90 | 2 | **0** |
| 0.90 – 0.95 | 1 | **0** |
| **total** | **30** | **0** |

The failure is structural rather than a tuning problem. A semantic hit on B
returns A's answer verbatim, so it is correct only when A's answer happens to
answer B as well — and on this corpus it never did. A concrete pair in the
0.80–0.86 band: Q28 ("I want cashless at a hospital that turns out to be an
excluded provider. What are my options?") and Q42 ("The hospital said they
won't do cashless. Am I finished?"). Both are cashless-network questions, so
0.84 similarity is defensible by the embedding — but the difference between
"here is what you can still do" and "it is finished" is the entire answer, on
the topic where a customer is already stuck.

The outcome is bimodal. At the handout's suggested 0.95 the cache never fires at
all, because 24 of the 30 pairs sit below 0.80; at any threshold where it *does*
fire, it returned a wrong answer in every measured case. So
`SEMANTIC_CACHE_ON = False` ships, and the threshold remains recorded at 0.95
for anyone who wants to re-measure it. Thirty pairs is a small sample and
establishes nothing about the rate; but the decision to leave a mechanism *off*
does not need a precise rate. It needs one counterexample, and there were 30.

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
total, exact cache         3 ms        3 ms
```

The cached row is the **exact-key** layer only. The semantic layer contributes
nothing to it, because it ships disabled (§3) — there is no measurement of
semantic-cache hit latency here, because there is no such hit to measure.

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

**3. Attack the 8 generation failures with decomposition, not prompt words.**
Expected value: medium. Most are multi-hop. One prompt-level fix was tried
(rule 7) and it did move the target metric on the shipping model, which
weakens the "prompt words do nothing" claim — the honest version is that
prompt words move correctness only at some tiers, so each one has to be
measured on the deployed model. Splitting a multi-hop question into
sub-questions and retrieving per hop is a structural change with a mechanism
behind it, independent of which model is deployed.

**4. Fix Q23 and Q17 — the two refusals that are wrong for a structural reason.**
Expected value: medium, cost low. Neither is variance: Q23 is refused when it
should be answered (the single wrongful refusal in the gate run), and Q17 is
refused in all six online repeats because rule 7 causes it. They come from
different experiments, so this is two independent bugs rather than one
question seen twice. Both are worth reading by hand before anything above,
because two known-bad answers are a better use of an afternoon than a
structural change.

**Not doing:** relaxing the refusal threshold. It was relaxed once and, on the
shipping model, that relaxation *raised* refusal precision (0.698 → 0.833) and
recall (0.933 → 1.000). Current precision is 0.833, comfortably above the 0.75
gate, and the one remaining cost is a single question. There is nothing to buy.

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

**Correctness is replayed too, and that one is subtler.** A replayed judgement
is a real judgement of a real answer — it is not fabricated — but it is the
judgement from *one recorded run*, so the gate reports 0.775 correctness with a
±0.000 that it does not deserve. The live figure for the same configuration,
3 online repeats, is **0.783 ± 0.007** (§3). Where the two disagree, believe
the online number; where the gate is silent, believe it, because replay detects
regressions between commits that live sampling would only catch at the cost of
running the provider on every push.

**TTFT is reported as a miss and is deliberately not in the gate.** It is the one
graded target with no threshold, and the reason is mechanical rather than
convenient: a replayed stream arrives as a single delta, so time-to-first-token
under `AIP_OFFLINE=1` measures the replay, not the service. Gating it would
produce a green tick on a number that means nothing. It stays an un-gated,
openly-missed target in section 2, measured online, and it is first on the fix
list in section 7 for that reason.