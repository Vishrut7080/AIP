#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml

Runs the full 45-question golden set and returns every metric thresholds.yml
names. Under AIP_OFFLINE=1 it replays the committed cache: no API key, no cost,
deterministic. That is what lets CI run this on every push.

WHY THE SHIPPING CONFIG IS FIXED HERE
-------------------------------------
The gate is only meaningful if it measures the system that actually ships. That
system is labs/lab7/service.py's: `lenient_complete` prompt, final_k=8,
GENERATION_TIER=SMALL. Those are imported, not re-declared, so the gate cannot
drift away from the service -- a gate measuring a different pipeline than the one
in production is worse than no gate, because it is green and wrong.

MEASUREMENT NOTES
-----------------
* `correctness` is normalised to 0-1 (judge returns 0-2) so it is comparable
  with thresholds.yml's {min: 0.75}.
* `hit_rate_at_5` is retrieval-only and measures the STAGE, so a retrieval
  regression is visible even when the generator happens to paper over it.
* Judge parse failures return None and are EXCLUDED, not scored as 0. Missing
  data is not a failing answer.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cache import CacheMiss  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.retrieval import format_context  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402
from labs.lab4.evaluate import judge_correctness, judge_faithfulness  # noqa: E402
from labs.lab4.rag import answer_question  # noqa: E402
from labs.lab7.service import (  # noqa: E402
    FINAL_K,
    GENERATION_TIER,
    RETRIEVE_K,
    STRICTNESS_NAME,
    pipeline,
)

# B4's stage breakdown, recomputed per span name for the report.
HIT_K = 5


class GateDataError(RuntimeError):
    """The gate could not measure enough of the golden set to judge it.

    Raised rather than returned, because a metric computed from half the data
    looks exactly like a metric computed from all of it. main() turns this into
    a non-zero exit with the reason attached.
    """


# Wall-clock fields, split out of the committed artefact.
#
# Everything else in the metric dict is deterministic under AIP_OFFLINE=1 -- same
# cache, same machine-independent arithmetic, same answer every run. Wall clock
# is not: replaying 45 questions measures this machine's Python, and it came out
# at 6.7, 6.9 and 7.2 ms on three consecutive runs. Committing a number that
# moves on every push trains you to ignore diffs in the file that actually
# matter, which is the opposite of what a regression artefact is for.
#
# So the fine-grained timings go to a gitignored sidecar, and the one wall-clock
# figure that IS gated -- p95 -- stays but is bucketed to 10 ms. The threshold is
# 6000 ms, so a 10 ms bucket is 0.17% of it: it cannot hide a regression by any
# margin worth gating on, and it makes the committed number stable.
_TIMING_FIELDS = ("_p50_wallclock_ms", "_p95_wallclock_ms",
                  "_p99_wallclock_ms", "_latency_caveat")


def _round_to(x: float, step: int) -> float:
    return float(int(round(x / step)) * step)


def refusal_metrics(rows: list[tuple[str, bool]], una_ids: set[str]) -> dict:
    """Refusal recall and precision from (id, refused) pairs.

    Extracted from `measure()` so the formula is unit-testable without an API
    key -- see tests/test_gate_refusals.py. That test exists because the
    original inline version was a tautology: it averaged a list already
    filtered down to refusals, so every element was True and the value was
    1.0 for ANY behaviour, including a system that refuses every answerable
    question. It passed its own 0.75 threshold. A gate metric that cannot fail
    is worse than no metric, because it reads as coverage.

    recall    = of the unanswerable questions, the share that were refused
    precision = of everything refused, the share that SHOULD have been refused
    """
    unanswerable = [v for i, v in rows if i in una_ids]
    refused_ids = [i for i, v in rows if v]
    true_pos = [i for i in refused_ids if i in una_ids]

    def mean(xs: list[float], default: float = 0.0) -> float:
        return statistics.fmean(xs) if xs else default

    return {
        "refusal_recall": round(mean([1.0 if v else 0.0 for v in unanswerable]), 4),
        "refusal_precision": round(
            len(true_pos) / len(refused_ids) if refused_ids else 0.0, 4),
        "_n_refusals": len(refused_ids),
        "_n_rightful_refusals": len(true_pos),
        "_wrongful_refusals": [i for i in refused_ids if i not in una_ids],
    }


def measure() -> dict[str, float]:
    """TODO D1: run your golden set and return the metric dict.

    Keys must match thresholds.yml. Run under AIP_OFFLINE=1 so CI replays the
    committed cache and costs nothing.
    """
    questions = load_questions(include_unanswerable=True)
    una_ids = {q["id"] for q in questions
               if q["kind"] == "unanswerable" or not q["relevant_docs"]}
    ans = [q for q in questions if q["id"] not in una_ids]
    r = pipeline()

    def ask(q):
        """Run one question, turning a cache miss into a measured failure.

        D3 found this the hard way: deliberately setting final_k=1 changes the
        context, so the exact chat request is no longer in the cache, and under
        AIP_OFFLINE=1 that raised CacheMiss and CRASHED the gate. CI would have
        reported a traceback, not "quality dropped" -- which is a worse outcome,
        because it looks like an infrastructure problem and gets ignored.

        A cache miss here means the configuration under test is not the one the
        cache was recorded against. That is a REAL regression signal, so it is
        reported as a zero-quality answer rather than an exception.
        """
        try:
            return answer_question(q["question"], r, k=RETRIEVE_K,
                                   final_k=FINAL_K, strictness=STRICTNESS_NAME,
                                   tier=GENERATION_TIER)
        except CacheMiss:
            cache_misses.append(q["id"])
            from labs.lab4.rag import Answer
            return Answer(question=q["question"], text="", hits=[], refused=False,
                          citations_valid=False, invalid_citations=[1])

    rows, latencies, cache_misses = [], [], []
    with Budget(limit_usd=3.00, label="lab7-gate") as budget:
        for q in questions:
            rows.append((q, ask(q)))

    # Wall-clock per question, measured here rather than taken from the Budget.
    #
    # The Budget's own p95 is 0.0 ms under AIP_OFFLINE=1, because a cache hit
    # records latency_ms=0.0 by design (aip/llm.py raw_call) and CI replays
    # entirely from cache. A latency gate fed 0.0 can NEVER fail, so it would be
    # a gate that looks armed and is not -- and a green build would be evidence
    # of nothing. Measuring wall clock at least measures THIS machine: real
    # retrieval, real code, a cache lookup that genuinely resolves in ~0ms.
    #
    # It is still not the deployed p95, which depends on provider latency. So
    # the gate reports the measured value, names the caveat, and the deployed
    # figure is quoted separately in EVALUATION_REPORT.md from an online run.
    # Under CI the honest answer is "this build did not regress", not "p95 is 0".
    from time import perf_counter
    with Budget(limit_usd=3.00, label="lab7-gate-wallclock"):
        for q in questions:
            t0 = perf_counter()
            a = ask(q)
            latencies.append((perf_counter() - t0) * 1000)

    def wall_p(pct: float) -> float:
        xs = sorted(latencies)
        return xs[min(len(xs) - 1, int(round(pct / 100 * (len(xs) - 1))))]

    correctness, faithfulness = [], []
    for q, a in rows:
        # The judges are model calls too, so a changed configuration makes them
        # cache misses as well. Under AIP_OFFLINE=1 that raised CacheMiss and
        # crashed the gate one level up from the generator. A judge that cannot
        # run is missing data, and missing data is NOT a passing answer.
        try:
            f = judge_faithfulness(a.text, format_context(a.hits))
        except CacheMiss:
            f = None
        if f is not None:
            faithfulness.append(f)
        if q["id"] not in una_ids:
            try:
                c = judge_correctness(q["question"], a.text, q["gold_answer"])
            except CacheMiss:
                c = None
            if c is not None:
                correctness.append(c)

    # An unmeasurable judge would otherwise silently drop the metric and let the
    # gate pass on absent data. Fail loudly instead.
    if len(correctness) < 0.5 * len(ans):
        raise GateDataError(
            f"only {len(correctness)}/{len(ans)} correctness judgements could be "
            f"made (judge cache misses). Refusing to report a gate on partial "
            f"data -- this is what a broken configuration looks like, and it must "
            f"fail rather than pass quietly.")

    # Retrieval-only: does a relevant document appear in the top HIT_K?
    hits_at_k = []
    for q in ans:
        got = {h.chunk.doc_id for h in r.search(q["question"], k=HIT_K)}
        hits_at_k.append(bool(got & set(q["relevant_docs"])))

    # Refusal metrics. See refusal_metrics() for why the precision formula is
    # not a mean over a pre-filtered list of booleans.
    refus = refusal_metrics([(q["id"], a.refused) for q, a in rows], una_ids)

    def mean(xs: list[float], default: float = 0.0) -> float:
        return statistics.fmean(xs) if xs else default

    cost_per_query = budget.spent_usd / max(1, len(questions))

    return {
        "correctness": round(mean(correctness) / 2, 4),
        "faithfulness": round(mean(faithfulness), 4),
        "citation_validity": round(
            mean([1.0 if a.citations_valid else 0.0 for _, a in rows]), 4),
        "refusal_recall": refus["refusal_recall"],
        "refusal_precision": refus["refusal_precision"],
        "hit_rate_at_5": round(mean([1.0 if v else 0.0 for v in hits_at_k]), 4),
        # Cost under AIP_OFFLINE=1 is legitimately 0 -- nothing left the
        # machine. The deployed figure comes from an online run.
        "cost_per_query_usd": round(cost_per_query, 6),
        # Wall clock, not Budget.percentile(95), which is 0 under replay.
        "p95_latency_ms": _round_to(wall_p(95), 10),
        # Extras. Not gated, but written to reports/gate_metrics.json so the
        # evaluation report can quote them without re-running anything.
        "_n_questions": len(questions),
        "_n_answerable": len(ans),
        "_n_refusals": refus["_n_refusals"],
        "_n_rightful_refusals": refus["_n_rightful_refusals"],
        "_wrongful_refusals": refus["_wrongful_refusals"],
        "_n_judged_correctness": len(correctness),
        "_n_cache_misses": len(cache_misses),
        "_cache_miss_ids": cache_misses[:20],
        "_cache_miss_note": (
            "A cache miss under AIP_OFFLINE=1 means the config under test is "
            "not the one the cache was recorded against. Treated as a zero-"
            "quality answer, not an exception."),
        "_p50_wallclock_ms": round(wall_p(50), 1),
        "_p99_wallclock_ms": round(wall_p(99), 1),
        "_latency_caveat": (
            "Wall clock on a cache-replaying machine. Under AIP_OFFLINE=1 a "
            "cache hit costs 0 ms, so this measures code + retrieval + cache "
            "lookup, NOT provider latency. The deployed p95 is in "
            "EVALUATION_REPORT.md from an online run."),
        "_config": {
            "strictness": STRICTNESS_NAME, "final_k": FINAL_K,
            "tier": GENERATION_TIER, "retrieve_k": RETRIEVE_K,
            "hit_k": HIT_K,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    ap.add_argument("--save", default="reports/gate_metrics.json")
    ap.add_argument("--save-timing", default="reports/gate_timing.local.json",
                    help="where to write machine-specific wall clock (gitignored)")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    try:
        metrics = measure()
    except GateDataError as exc:
        # A gate that crashes is worse than one that fails: CI reports a
        # traceback, which reads as an infrastructure problem and gets ignored.
        # A gate that fails with a reason gets acted on.
        print(f"\nGATE FAILED (could not measure):\n  {exc}")
        return 1

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    cfg = metrics.get("_config", {})
    print(f"\nmeasured: {metrics['_n_questions']} questions "
          f"({metrics['_n_answerable']} answerable), "
          f"{metrics['_n_refusals']} refusals, "
          f"{metrics['_n_judged_correctness']} correctness judgements")
    print(f"config:   {cfg}")
    n_miss = metrics.get("_n_cache_misses", 0)
    if n_miss:
        print(f"\nWARNING: {n_miss} question(s) were cache misses under "
              f"AIP_OFFLINE=1 -- the config under test differs from the one the "
              f"cache was recorded against. Each counts as a zero-quality answer.")
        print(f"         first: {', '.join(metrics['_cache_miss_ids'][:5])}")

    if args.save:
        p = ROOT / args.save
        p.parent.mkdir(parents=True, exist_ok=True)
        # The committed artefact holds only what is reproducible. Stripping the
        # wall-clock fields here rather than in measure() keeps them available to
        # this process -- the printed table still shows a real p95.
        p.write_text(json.dumps(
            {k: v for k, v in metrics.items() if k not in _TIMING_FIELDS},
            indent=2), encoding="utf-8")

    if args.save_timing:
        # Machine-specific, deliberately untracked. EVALUATION_REPORT.md quotes
        # the deployed p95 from an online run, not from here.
        t = ROOT / args.save_timing
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(json.dumps(
            {k: metrics[k] for k in _TIMING_FIELDS if k in metrics}, indent=2),
            encoding="utf-8")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())