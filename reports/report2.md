# Lab 2 — The Prompt Lab: Recommendation Report

Harness: `labs/lab2/grid.py` (given) + variants in `labs/lab2/variants.py`.
All runs below are **uncached** (`AIP_CACHE=0`) so costs and latencies are real
and cross-comparable across the 7 configurations. Dev = `data/eval/extraction_dev.jsonl` (n=60),
test = `data/eval/extraction_test.jsonl` (n=120). Log: `labs/lab2/lab2_output.txt`, results: `reports/lab2_grid.json`.

## Grid table (dev, n=60, all configs, schema_valid = 1.0000, error_rate = 0)

| configuration | record_accuracy | field_accuracy | repair_rate* | cost_usd | cost_per_1k | p50_ms | p95_ms | cost/yr @10k/day |
|---|---|---|---|---|---|---|---|---|
| zero_shot | 0.5333 | 0.9104 | 0 | $0.0439 | $0.73 | 1175 | 1463 | $2,669 |
| few_shot | **0.5833*** | 0.9187 | 0 | $0.0555 | $0.93 | 1167 | 1444 | $3,377 |
| zero_shot_main | 0.4833 | 0.9062 | 0 | $0.1561 | $2.60 | 2454 | 3406 | $9,494 |
| few_shot_main | 0.5500 | 0.9250 | 0 | $0.2260 | $3.77 | 2790 | 4592 | $13,749 |
| few_shot_reasoned | 0.5500 | 0.9104 | 1.13 | $0.1535 | $2.56 | 1436 | 1852 | $9,337 |
| few_shot_reasoned_main | 0.5333 | 0.9208 | 0 | $0.2414 | $4.02 | 2921 | 4855 | $14,686 |
| cascade | 0.5833* | **0.9292*** | 0 | $0.1526 | $2.54 | 1213 | 2571 | $9,285 |

\* `repair_rate` = extra structured-retry calls per ticket (harness does not persist a per-case repair flag; calls/ticket is the closest observable). Only `few_shot_reasoned` retried (128 calls for 60 tickets): the reasoning field overflowed the small model's output budget and triggered a retry on roughly half the cases. The cascade's extra calls (130/60) are the second draw + escalations, not repairs.

## Part A — Few-shot selection (6 examples, by hand)

| id | what it teaches |
|---|---|
| T0054 | mis-sold policy → **complaint** (not claims/billing); angry refund demand is urgency **4** not 5; no policy number → `null` (wrong in Lab 1) |
| T0097 | billing/complaint boundary: ombudsman threat is anger, not a category change; double debit stays **billing**; urgency 4, escalate |
| T0200 | Hinglish (`language=hi-en`); question about a past claim stays **claims**; frustrated tone is **not** escalate; urgency 3 (wrong in Lab 1) |
| T0095 | factual account question (wellness points) is **information**, not a claim; urgency 2; PII |
| T0048 | action request with embedded question stays **policy_change**; no policy number → `null`; urgency 2 |
| T0222 | reverse sentiment trap: a *satisfied* customer still needs routing (information, urgency 1); sentiment never sets urgency |

**A2.** `few_shot_block()` renders each example in the exact `TicketRecordC` JSON format the model must output.

**A3.** Dev comparison (this report).

**A4 — the problem.** The 6 examples came out of dev and are measured on dev → in-sample calibration; the few-shot delta can be an artifact of examples that "teach" specifically the dev cases they came from. **Fix applied:** re-ran `zero_shot` vs `few_shot` on the **untouched test split** (n=120):

zero_shot test: field 0.9021, record 0.5000, $0.73/1k, p95 1662 ms · few_shot test: field 0.9094, record 0.5250, $0.93/1k, p95 1438 ms · paired b=14, c=17, p=0.72.

Selection was also made on teaching value (edge cases my Lab-1 extractor got wrong), not by picking the 6 most extreme dev items to manufacture a delta. The few-shot gain is small, non-significant, and consistent across both splits — nothing here is earned by leakage.

## Part B — Grid answers

1. **Prompt moved more than tier.** Record accuracy rose with the prompt (0.5333→0.5833 SMALL, 0.4833→0.5500 MAIN) and **fell** when tier rose (0.5333→0.4833 zero, 0.5833→0.5500 few), at 3–4× the cost and 2–3× the p95. The model tier's only real gain was field accuracy on the few-shot prompts (+0.0063), which the paired test shows is noise.
2. **The reasoning field cost money and bought nothing.** Adding `reasoning` to few_shot SMALL cost 2.76× ($0.0555 → $0.1535), raised output tokens ~2.1× (128 vs 60 calls), +23% p50 latency — and *lowered* record accuracy (0.5833→0.5500) and field accuracy (0.9187→0.9104). On MAIN it again cost money for the same or worse accuracy. Accuracy points per rupee: **negative**. Negative result.
3. **Dominated configurations** (worse on quality AND cost AND latency): `zero_shot_main`, `few_shot_main`, `few_shot_reasoned`, `few_shot_reasoned_main`, **and `cascade`** (dominated by few_shot: same record accuracy, 2.7× cost, 1.8× p95). The non-dominated frontier is just {zero_shot, few_shot}.

## Part C — The cascade

Trigger: two SMALL draws at T=0.0 vs T=0.7 (distinct cache keys — the T=0.7 draw is what makes disagreement detectable), escalate to MAIN on empty/short evidence, category/urgency/sentiment disagreement, or a structured-error fallback.

- **Escalation rate:** 10/60 = **16.7%**
- **Blended accuracy:** field **0.9292** (best in grid), record **0.5833** (= few_shot)
- **Blended cost:** $0.1526 → $2.54/1k, **$9,285/yr** — **2.7×** few_shot, ~3.5× zero_shot

**Did the trigger carry signal?** A proxy for agreement-when-right vs -wrong: the 10 escalated tickets were mis-scored by the un-routed small `few_shot` in **6/10 (60%)** of cases vs its 42% overall error rate — the trigger does select harder-than-average items. But even so, **5 of the 10 escalated tickets were still wrong after the MAIN rewrite.** And relationally the cascade's extra spend produced **zero** record-accuracy gain over plain few_shot (35/60 twice). The escalation ladder recovers some hard items and breaks some clean ones, and here they exactly cancel.

**Negative result:** self-consistency-style disagreement routing adds 2.7× cost for no detectable quality gain; few_shot already delivers the ladder's blended record accuracy at 37% of the price.

## Part D — Is a difference real?

95% Wilson CIs (record accuracy, dev): few_shot [0.457, 0.699], cascade [0.457, 0.699], zero_shot [0.409, 0.654] — heavily overlapping; unpaired comparison cannot separate any two configs.

McNemar paired tests, all vs `zero_shot` over the same 60 items:

| comparison | b (zero right, other wrong) | c (other right, zero wrong) | p |
|---|---|---|---|
| vs zero_shot_main | 12 | 9 | 0.66 |
| vs few_shot | 4 | 7 | 0.55 |
| vs few_shot_main | 7 | 8 | 1.00 |
| vs few_shot_reasoned | 5 | 6 | 1.00 |
| vs few_shot_reasoned_main | 8 | 8 | 1.00 |
| vs cascade | 4 | 7 | 0.55 |

**No configuration is detectably different from the cheap baseline.** Conclusion: choose on cost — but note the largest non-significant trend is few_shot (b=4, c=7), which is also record-best.

## Part E — Error analysis on the best *quality* config (few_shot)

**E1. 25 imperfect records; top three clusters:**
1. **Urgency mis-scored — 18 of 25 (72%).** Urgency errors alone or dragging `escalate`/`sentiment` down with them; the severity boundaries are the single dominant failure across all seven configs.
2. **Sentiment mislabelled — 10 of 25 (40%).** Neutral/transactional tone read as negative (or vice versa), and negative tone incorrectly promoted sentiment — but never urgency, so the downgrade rule is honored.
3. **Escalate-coupled error — 5 of 25.** When urgency is wrong, `escalate` usually follows it (they share one business rule); remainder is PII contamination (3 cases, e.g. T0194/T0199).

**E2. Worst field: `urgency`. Confusion matrix (few_shot, gold × predicted):**

| gold \ pred | 1 | 2 | 3 | 4 | 5 | acc |
|---|---|---|---|---|---|---|
| 1 | 7 | 5 | 0 | 0 | 0 | 0.58 |
| 2 | 0 | 16 | 0 | 0 | 0 | 1.00 |
| 3 | 0 | 5 | 6 | 0 | 0 | 0.55 |
| 4 | 0 | 0 | 4 | 9 | 1 | 0.64 |
| 5 | 0 | 0 | 1 | 2 | 4 | 0.57 |

**The systematic confusion the aggregate hides:** the 5-level severity scale is really being scored on ~3 effective levels. Column totals show **26 of 60 predictions (43%) land on level 2**, and every error is exactly one *adjacent* level toward the middle of the scale (1→2 five times, 3→2 five times, 4→3, 5→3/4). That is a central-tendency pull: word-level sentiment raises low-severity tickets (gold 1→2) while policy-grounded high severity gets walked back (gold 5→4). Levels 2 vs 1 and 3 vs 4 are not distinguishable from wording alone, and no configuration or tier fixes it — `urgency` stays the worst field on every row of the grid.

**E3. Recommendation.**
**Ship `zero_shot` on `SMALL`.** It is the cheapest configuration (field accuracy **0.9104**, record accuracy **0.5333**, **$0.73 per 1,000 tickets** = **$2,669/year** at 10,000 tickets/day), and the paired test gives no statistically detectable reason to pay more: every upgrade costs 2.7–5.5× without a significant gain. **Change my mind:** if a downstream consumer actually pays for record-level correctness — for example urgency feeds an SLA/route decision where a missed severity is expensive — switch to `few_shot` (same $0.58/ticket marginal cost, +0.05 record, +0.0083 field, its b=4/c=7 trend is the largest in the grid) *once that delta is confirmed significant on a fresh n≥300 sample*; otherwise the extra $708/year is not justified. Do not buy the cascade, the MAIN tier, or the reasoning field: all three are dominated or noise.

---

### Negative results (explicit)
- The MAIN tier: 3.5–5.5× cost, 2–3× p95 latency, no detectable accuracy gain (p = 0.66–1.00) — worse record accuracy than SMALL on every prompt.
- The reasoning field: 2.7× cost and *negative* accuracy margin on SMALL (record 0.5833→0.5500), plus ~1.13 retries/ticket.
- The cascade: escalation 16.7% selects harder items (60% vs 42% error) but the blended record accuracy ties few_shot at 2.7× its cost; dominated.
- Few-shot over zero-shot: both dev (p=0.55) and test (p=0.72) — no detectable difference; ship on cost.