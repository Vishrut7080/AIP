# Lab 1 — Part A Answers (v0 naive extractor, n=40)

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