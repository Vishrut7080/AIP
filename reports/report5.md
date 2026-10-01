# Lab 5 — RAG v2: Diagnose, Fix, Prove

**Name:** ______  **Partner:** ______  **Date:** ______

> **This file is written incrementally on purpose.** Part B's prediction is
> recorded *before* the fix is implemented, so it cannot be retrofitted to
> whatever the number turns out to be. Later sections are appended as they are
> measured. `no number, no claim` (T3 §1.1).

---

## Part A — the failure tally

**Input:** `reports/lab4.json` (45 questions; 5 unanswerable). **Command:**
`python labs/lab5/diagnose.py --precompute` then `--input reports/lab4.json --pareto`

Lab 4 baseline: correctness **1.500** / 2 (n = 40 answerable), faithfulness
**0.956**, citation validity **1.000**, refusal recall **0.800** (4/5), refusal
precision **0.571** (4 of 7 refusals were correct).

```
failure mode          n    share   cumulative
generation            12   70.6%     70.6%  █████████████████████
ranking                5   29.4%    100.0%  █████████
```

**17 failures, 0 needing a hand check, concentrated in two modes** — the shape
`README.md:98` says to expect. Modes 1, 2, 3 and 7 are empty; mode 5 is empty
for a structural reason (below).

### A1/A2 — the two clusters

| Cluster | n | Question kinds |
|---|---|---|
| **6 generation** | 12 | 5 single_hop, 5 multi_hop, 2 paraphrase |
| **4 ranking** | 5 | 2 multi_hop, 2 aggregation, 1 trap_archived |

Per-case evidence is in `reports/lab5_diagnosis.json`; the measurements behind
it are in `reports/lab5_signals.json` (17 records).

**Two things the classifier had to get right.**

*Refusals are not exempt from the mode-6 test.* Three of the failures are
refusals (Q20, Q41, Q43). Q20 refuses **on gold context too**, so no retriever
could have saved it. Q41 and Q43 answer correctly on gold context but refuse on
the real context, with every relevant document already in it — a refusal
threshold, not a retrieval failure. An early `if row["refused"]` guard hid all
three; the tree has to reach mode 6 first.

*Retrieval lost something when EITHER the answer chunk was crowded out OR a
relevant document never arrived.* Q22 and Q35 lost a whole document (a second
hop). Q23, Q29 and Q32 have **every** relevant document present but the specific
chunk holding the answer was crowded out of k=8. Presence of the document is
not presence of the passage; testing only one of the two misclassifies half of
mode 4.

### A3 — mode 5 is empty for a structural reason

Lab 4 ran with `reranker=None` (`evaluate.py:163`), so `answer_question` takes
the `hits[:final_k]` path and **no reranker existed to drop anything**. This is
not a diagnosis; it is an absence of machinery. `diagnose.py` prints this so the
zero is not mistaken for a finding.

### A4 — evidence the clusters are not the same shape

Within mode 6, **9 of 12 are partial credit (correctness 1), not wrong answers.**
The system states the correct headline and drops a clause:

| | Question | Answer given | What was dropped |
|---|---|---|---|
| Q03 | maternity on Bronze? | "No, not covered on Bronze" | that Silver/Gold/Platinum **do** cover it |
| Q05 | grace period, annual? | "30 days" | the 15-day instalment variant; that cover does not operate |
| Q25 | caesarean cost on Gold? | "₹60,000 out of pocket" ✓ | "Gold has no co-payment" |

The remaining 3 (Q20, Q41, Q43) are the wrongful refusals above.

This matters for Part B: the dominant cluster reads as **under-answering**, not
incapacity. Correctness by kind, for context:

| kind | n | mean (0–2) | partial | zero |
|---|---|---|---|---|
| single_hop | 18 | 1.722 | 5 | 0 |
| multi_hop | 10 | 1.200 | 6 | 1 |
| paraphrase | 5 | 1.200 | 0 | 2 |
| aggregation | 4 | 1.500 | 2 | 0 |
| trap_archived | 3 | 1.667 | 1 | 0 |

### A5 — an out-of-scope finding

**Q40 is unanswerable, the system answered it anyway, and the judge scored it
2/2.** It is therefore invisible to the classifier and absent from the 17. Real
refusal recall is 4/5 and the judge is rewarding an ungrounded answer. This is a
*judge* problem, not a RAG problem, and it is not fixed in this lab — recorded
here so it is not mistaken for a clean result.

---

## Part B — rank by expected value — **20 min**

| Cluster | n | Fix | Est. recovery | Cost Δ | Latency Δ | Effort |
|---|---|---|---|---|---|---|
| **6 generation** (9 partial) | 9 | Completeness rule: name the second clause | 4–6 | 0 | ~0 | low |
| 6 generation (3 refusals) | 3 | Relax refusal threshold | 2–3 | 0 | ~0 | low |
| **4 ranking** | 5 | `final_k` 8 → 12 | 2–3 | ~0 | slight ↑ | low |

### The pick, and why not the others

**I pick the completeness rule on the 9 partial-credit cases.** It is the
largest sub-cluster, it costs nothing (no extra model call), and it is the
diagnosis-driven choice: those answers are right about the headline and silent
about the exception.

**I am deliberately not touching the refusal threshold,** and this is the
important decision in Part B. `OVERVIEW.md:126` records that the reference
solution relaxed the answer-length rule and measured **correctness *and* refusal
precision both falling** (−0.050). That threshold has *already been relaxed once*
— `lenient` is the deployed config (`evaluate.py:154`) — and correctness is still
1.500. Relaxing it again is the move with the weakest prior support in this
dataset, and it is the one most likely to re-run someone else's published
failure. Refusal precision is already the worst metric on the board at 0.571, so
there is little room to trade it down.

**Mode 4 is the cheaper fix but not the better one.** `final_k` 8 → 12 is nearly
free, but its 5 cases are aggregation and archived-trap questions. Those need a
*specific* fact that a crowded-out chunk held; more context mostly adds
distractors. Expected value is lower than the count suggests.

### B3 — the prediction, written before implementation

> **I expect the completeness rule to recover 4 to 6 of the 9 partial-credit
> mode-6 cases, taking correctness from 1.500 to roughly 1.60–1.70, with refusal
> precision unchanged at ~0.57.**

Recorded 2026-10-01, before the fix was implemented. Being wrong is informative
and is not penalised; not predicting is.

**What would falsify my model of the system:** if correctness moves by less than
+0.05, then "the answers are incomplete" is the wrong diagnosis and the limit is
the model's reasoning rather than the prompt's brevity — in which case the next
thing to try is decomposition for multi-hop, not more prompt words.

---

## Part C — the fix — **70 min**

Rule 7, appended to the deployed `lenient` prompt (`rag.py`):

> 7. If the sources state a second condition, limit, exception or variant that
> changes the answer — a different plan, tier, term, or instalment case — state
> it explicitly. Answering the general case alone is an incomplete answer, not a
> correct one. Do not pad with anything the sources do not say.

**One variable, verified.** Rules 1–6 are byte-identical to v1; the diff is the
two lines of rule 7 and a blank. I first re-listed the whole rule set in the new
variant and that introduced whitespace changes *inside the refusal sentence* —
which would have made this a two-variable experiment. Fixed by appending to the
shipped string instead.

**Cost: $0.0114/query, no extra model call.** Inside the ≤2× rule.

---

## Part D — prove it

### D1 — before/after, every Lab 4 metric

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| **correctness (0–2)** | **1.500** | **1.500** | **0.000** |
| faithfulness | 0.956 | 0.911 | **−0.044** |
| citation validity | 1.000 | 1.000 | 0.000 |
| refusal recall | 0.800 | 0.600 | **−0.200** |
| refusal precision | 0.571 | 0.429 | **−0.143** |
| repair rate | 0.089 | 0.089 | 0.000 |
| mean answer length | 202 ch | 227 ch | +25 |
| cost / query | — | $0.0114 | ≤2× ✓ |

**The prediction was wrong.** I predicted 1.60–1.70; correctness did not move.

### D2 — the regression check — **three metrics got worse**

The headline 0.000 is a **cancellation, not an absence of effect**: 4 questions
improved and 4 previously-passing questions broke.

**Recovered (4):**

| | v1 → v2 | what happened |
|---|---|---|
| Q03 | 1 → 2 | added *"Maternity benefits are only available on Silver, Gold and Platinum"* — exactly the clause Part A predicted |
| Q23 | 1 → 2 | named the crowded-out second hop |
| Q41 | 0 → 2 | wrongful refusal resolved — gave the 30-day / 15-day instalment split |
| Q43 | 0 → 2 | wrongful refusal resolved — 200 points, 1000 = 5% capped at 15% |

**Regressed (4):**

| | v1 → v2 | what happened |
|---|---|---|
| Q01 | 2 → 1 | added the **archived-2024 15-day policy** the gold answer deliberately omits. Faithful to the sources, wrong against the reference — the archived-document trap leaking into a single_hop question |
| Q13 | 2 → 0 | became a wrongful refusal |
| Q21 | 1 → 0 | became a wrongful refusal |
| Q42 | 2 → 0 | became a wrongful refusal |

Plus **Q06 and Q45 lost faithfulness (1 → 0)** with correctness unchanged. Rule 7
asks the model to name a second condition, and it sometimes names one the sources
do not support. That is the rule's failure mode, and it is visible in the
faithfulness column rather than the correctness column.

**The mechanism.** Rule 7 made the model *more willing to speak*. On the 9
partial-credit cases that was right. But speaking more also made it speak when it
should not — 3 new refusals, 2 unsupported additions. The gain and the cost come
from the same instruction. This is the same shape as the reference's failure
(`OVERVIEW.md:126`), arrived at by a different route: I avoided touching the
refusal *threshold* and reintroduced the problem through the *completeness*
requirement.

**Worth noting which kind absorbed the damage.** `single_hop` was the strongest
kind in v1 (1.722) and took the worst hit (1.611). `paraphrase` went the other
way (1.200 → 1.600). So the rule helped exactly the cluster it was aimed at and
damaged the cluster that was already working.

### D3 — re-classifying what remains

16 failures (v1: 17). **The distribution did not shift.** All 5 mode-4 ranking
failures persist untouched — a prompt change cannot fix retrieval. Of the 12
mode-6 failures, 3 recovered and 3 became new refusals, leaving 8.

This is *not* the masked-problem case the handout describes. Nothing new was
revealed; mode 6 simply traded three failures for three others.

### D4 — the next fix

**`final_k` 8 → 12, aimed at the 5 mode-4 ranking failures.** It is the only
untouched cluster and nearly free. Expected 2–3 of 5 — lower than the count
implies, because Q22 and Q35 need a specific *document* rather than more chunks.
Refusal behaviour must be re-measured, not assumed.

---

## Verdict

**Do not ship this.** Correctness is flat, three metrics regressed, and the
target metric did not move at all.

But the diagnosis was sound — Q03 and Q41/Q43 recovered *exactly* as Part A
predicted, which is evidence the classifier is right. The failure was in the
fix's blast radius, not its target. Rule 7's second half is the problem:
*"Do not pad with anything the sources do not say"* is a prohibition, and
prohibitions in a system prompt are weaker than the affirmative instruction
preceding them. A fix that asks for completeness without a hard citation-enforced
guard on the added clause is the shape that regressed.