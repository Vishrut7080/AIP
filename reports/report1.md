# Lab 1 — Report (Parts A–D)

*Measured on the `SMALL` tier. Dev split = 60, test split = 120 (test run once). Source: `labs/lab1/lab1_output.txt` and the Part A run.*

---

# Part A — v0 failure characterisation (n=40)

## 1.2 — Failure table

| Failure mode | Count in 40 | Example ticket id |
|---|---|---|
| Not valid JSON at all | 31 | T0054 |
| JSON wrapped in a markdown fence | 31 | T0054 |
| Extra prose before or after the JSON | not distinguished by this run (folded into fence bucket) | — |
| Valid JSON, missing a required field | 0 | — |
| Category outside the allowed set | 31 | T0054 |
| Urgency as a string instead of an int | 31 | T0054 |
| Policy number invented (not in the text) | 0 | — |
| Unhandled exception | 9 | T0078 |

**Notes on the table:**
- The 31 count comes directly from Part 1's `markdown_fence` bucket. Part 2 confirms all 31 were recovered by stripping the wrapper, and 0 were unrecoverable — so this is a real, not estimated, number.
- The script does not separately log a "prose preamble" bucket distinct from `markdown_fence`, so that row can't be filled from this output without re-running with more verbose logging.
- "Missing field" and "invented policy number" are genuine zeros — Part 2 explicitly lists only `urgency_is_string` and `category_out_of_set` as the defects found inside the recovered JSON, nothing else.
- 9 + 31 = 40: every ticket accounted for, either via the exception path or the parse path.

## 1.3 — Mapping to the T1 §3 taxonomy

| Row | Taxonomy # | Failure name |
|---|---|---|
| Unhandled exception (RateLimitError) | #2 | Rate limit |
| JSON wrapped in markdown fence | #5 | Malformed output |
| Category outside allowed set | #6 | Schema violation |
| Urgency as string instead of int | #6 | Schema violation |

## 1.4 — The two rows that don't map cleanly

The two are the **fence-wrapping failure** and the **rate-limit exception** — for different reasons:

- **Markdown-fence wrapping** sits awkwardly under "malformed output" (#5) because the JSON itself is perfectly well-formed; the only defect is three characters of wrapper around it. It's a *formatting* problem, not a *content* problem, and arguably deserves its own bucket separate from things like trailing commas or truncated braces that are actually broken JSON.
- **RateLimitError** technically maps to #2, but on a first read it looks like it doesn't belong in a *data quality* taxonomy at all — it feels like pure infrastructure noise. That's the trap: those 9 tickets produced zero signal about extraction quality, and a naive "31/40 parsed" framing quietly erases them from the denominator.

## Script's closing questions

**1. Which of these are in the T1 §3 taxonomy, and which two are not?**
Fence-wrapping (#5) and schema violations (#6) map straightforwardly. RateLimitError maps to #2 but doesn't feel like it belongs at first glance. The one that resists mapping entirely is the fence-wrapping case argued as a distinct "formatting" failure rather than true malformed output — see above.

**2. Fixing ONE line takes you from 0/40 parsed to 31/40 parsed — but only 0/40 clean. Why is that second number the entire justification for Part B?**
Because stripping the fence only gets you *parseable* JSON, not *correct* JSON. All 31 recovered records still failed on content — every single one had a stringified `urgency` and an out-of-set `category`. That means the one-line fix (tolerant parsing) fixes failure mode #5 completely and does nothing for #6. If the lab stopped here, you'd report "31/40 success" while actually having 0 usable records. Part B exists because parsing tolerance and schema correctness are separate problems requiring separate solutions (structured output + validation + repair), and this 0/40 is the proof that one without the other is worthless.

**3. Which of these would a human reviewer even notice in production?**
The RateLimitError failures — those 9 tickets simply wouldn't produce a ticket record at all, which is visible immediately (a support ticket silently vanishing from the queue). The fence-wrapping and schema-violation failures would likely go *unnoticed* by a human spot-checking output, because the visible symptom (a JSON blob with a category string and urgency field) looks plausible at a glance. A downstream `int(urgency)` call crashing, or a category value falling outside a dropdown's allowed options, is the kind of failure that surfaces much later — in a broken dashboard or a rejected database write — not in a manual review of the model's raw output.

---

# Parts B/C — variant comparison (dev split, n=60)

| Metric | v0 (n=40) | B | C |
|---|---|---|---|
| schema_valid | 0/40 parsed (Part A) | 1.000 | 1.000 |
| field_accuracy | — | 0.871 | **0.902** |
| record_accuracy | — | 0.383 | **0.517** |
| cost_usd | — | 0.0529 | **0.0439** |
| latency p95 (ms) | — | 1674 | 1697 |
| errors | — | 0 | 0 |

(Both B and C ran with 0 cached calls, so the cost comparison is a clean same-cache-state measurement.)

**Reading — we deviate from the reference's "accuracy holds, it does not rise" story.** On the reference, all four deterministic fields were already at 1.000 in Part B, so Part C only bought lower cost. Ours *rose*: field accuracy +0.031 and record accuracy +0.133 from B→C, alongside a −17% cost cut ($0.0529 → $0.0439; reference −20–25%). The reason is visible in the per-field numbers: in our B, `contains_pii` was being decided by the model and leaking errors; moving it into auditable code (plus fixing the email logic) removed roughly a third of all imperfect records while the model's own fields stayed put. So on this baseline Part C bought accuracy *and* cost, not cost alone — the same favour, one row further right on the bargain curve.

**Latency** was flat (p95 1674 → 1697 ms), as expected: the same model behind the same call volume. **Validity** is 1.000 on both variants — the repair loop and the never-crash path earn their keep.

---

# Part D — test split triple (variant C, n=120, run once)

- **Quality:** schema_valid **1.000**, field_accuracy **0.899**, record_accuracy **0.475**, errors **0**, needs_review **0.0083** (exactly 1 record flagged for human review, not a crash).
- **Cost:** **$0.0878** for the full 120-ticket run (≤ $0.15 target ✓; **$0.00073/ticket**).
- **Latency:** p50 1392 ms / **p95 1585 ms** (≤ 4,000 ms target ✓).

**Honest note:** field accuracy 0.899 sits just *under* the ≥0.90 target and below the reference 0.930; record accuracy 0.475 is below the ≥0.55 target and the reference 0.608. The gap is concentrated in `urgency` and `sentiment` — see below. The dev→test gap is visible (`record` 0.517 → 0.475); we ran the test split exactly once.

## Per-field accuracy (test, worst first)

| Field | Accuracy | Reference |
|---|---|---|
| urgency | 0.658 | 0.75 |
| sentiment | 0.750 | 0.83 |
| category | 0.908 | 0.92 |
| escalate | 0.908 | — |
| contains_pii | 0.967 | 1.00 |
| language | 1.000 | 1.00 |
| policy_number | 1.000 | 1.00 |
| product | 1.000 | 1.00 |

**`contains_pii` recovered.** After fixing the deterministic email logic, `contains_pii` went **0.708 → 0.967** on test (0.950 on dev). The remaining 3.3% (4/120) is the one edge case the current rule still misses: a ticket carrying *both* a customer email in the live body and Aurora's own address in the quoted reply footer — the `any(e in excluded_emails)` test is true as soon as the Aurora address is seen, so the customer email is skipped. It's a small, named, and code-fixable residual, which is exactly why this field now belongs in code.

**`urgency` is the worst judge field** (0.658 vs reference 0.75). On dev it was 0.600; on test the errors cluster at the 3/4 anchor (repeated failure vs money-at-risk now). Not scattered — off-by-one at a boundary.

**`sentiment` is the surprise** (0.750 test vs 0.850 dev). The dev/test gap is real and underspecified: 5 of the 8 imperfect test records shown carry a wrong `sentiment`, and the failure mode is the "frustrated requires a prior failure" boundary — first-contact complaints with firm tone landing as `frustrated` instead of `neutral`.

## Category confusion matrix (test)

```
(rows = gold, cols = predicted)
                       billing   claims complaint information policy_change technical
billing                    16        .         .            .             .          .
claims                      .       20         .            1             .          .
complaint                   .        5        11            .             .          .
information                 .        2         .           20             .          .
policy_change               .        .         .            .            22          .
technical                   .        .         .            3             .         20
```

Patterns: `complaint` bleeds into `claims` (5) — the boundary the annotation guidelines warn about; and `information` absorbs mislabels from `technical` (3) and `claims` (2) — an over-cautious default when no pending transaction is obvious.

## Error clusters (from the 8 imperfect test records shown)

| Cluster | Example ids | Count | Fix | Approx. worth |
|---|---|---|---|---|
| **urgency off-by-one at anchors 3/4** | T0047, T0009, T0109, T0232 | 4 of 8 | Sharpen the 3v4 wording ("waiting vs money-at-risk-now"), or stretch 3: temperature-0.7 self-consistency taking the median | ~15–20 records |
| **sentiment: frustrated vs neutral** | T0049, T0208, T0177, T0093, T0232 | 5 of 8 | Reward evidence of a *prior* failure in the span; downgrade first-contact firm tone to neutral | ~12 records |
| **category: complaint→claims** | from matrix (5 on test) | 5 | Force the evidence span to answer "what do they want processed?" — conduct is a complaint only when Aurora's conduct is the subject | ~5 records |

A *closest-to-gold* aside: the deterministic move plus the `contains_pii` fix took imperfect records from 71 (first test run) down to **63**, with `contains_pii` absent from every row shown above.

## Targets vs measured (test, variant C)

| Metric | Target | Measured | Status |
|---|---|---|---|
| Schema validity | 1.00 | 1.000 | ✓ |
| Field accuracy | ≥ 0.90 | 0.899 | ≈ (0.002 under; below ref 0.930) |
| Record accuracy | ≥ 0.55 | 0.475 | ✗ (ref 0.608) |
| Cost, full run | ≤ $0.15 | $0.0878 | ✓ (ref $0.080) |
| p95 latency | ≤ 4,000 ms | 1,585 ms | ✓ (ref 2,276 ms) |
| Unhandled exceptions | 0 | 0 | ✓ |

---

# D5 — the economic argument

**Per-ticket economics (measured cost, test run):**
- System: $0.0878 ÷ 120 = **$0.00073/ticket**.
- Agent: 40 seconds at ₹300/hour = ₹3.33/ticket (≈ $0.040 at ₹83/$). One agent hour handles 90 tickets; the system handles ~1.36M for the same money.

**Annual cost at 10,000 tickets/day (365 days = 3.65M tickets):**
- System: 3,650,000 × $0.00073 ≈ **$2,671/yr** (~₹222k).
- Agent: 3,650,000 × ₹3.33 ≈ **₹12.2M/yr (~$147k)**.
- Gross saving ≈ **$144k/yr (~55×)**.

**Break-even record accuracy:** the system pays as long as the expected cost of *wrong* routings stays below the per-ticket saving. Saving/ticket ≈ ₹3.27 (₹3.33 − ₹0.06). If a misrouted ticket costs one extra agent touch (₹3.33), break-even accuracy = **1 − 3.27/3.33 ≈ 2%**. Our measured 0.475 is an order of magnitude above break-even — deployment economics are set by the cost of a wrong routing, not by raw record accuracy. Even at a pessimistic ₹6.23 misroute cost, our measured accuracy is exactly at parity; below that per-misroute cost, the system is worth deploying as-is.