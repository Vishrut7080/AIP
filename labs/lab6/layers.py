"""Part D -- the D1 table: every layer, both rates, plus what each costs.

    python labs/lab6/layers.py --max-calls 10

Cumulative passes answer "what does the whole stack buy". Independent passes
answer D2 -- "which layer gave the best block-rate-per-false-positive" -- because
a cumulative table cannot isolate a layer's own contribution: layer 3's row also
contains layers 1 and 2.

Each layer is ALSO run on its own so its marginal effect is visible.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            _s.reconfigure(encoding="utf-8", errors="replace")

from labs.lab6 import redteam as rt  # noqa: E402

LAYERS = {
    0: "none (baseline)",
    1: "1 delimit + declare",
    2: "2 heuristic detector",
    3: "3 structured output",
    4: "4 privilege capping",
    5: "5 output filtering",
}


def run(layers: set[int], max_calls: int, save: str | None = None) -> dict:
    """Run the suite for one configuration, reusing redteam's own machinery."""
    cases = rt.load_suite()
    attacks = [c for c in cases if c["vector"] != "control"]
    controls = [c for c in cases if c["vector"] == "control"]

    rt.inject_documents(cases)
    rows, latencies = [], []
    try:
        for c in cases:
            import time
            guard = rt._guard_for(layers, max_calls) if layers else None
            t0 = time.perf_counter()
            res = rt.run_agent(rt.effective_question(c), guard=guard,
                                layers=layers, budget_usd=0.20)
            latencies.append((time.perf_counter() - t0) * 1000)
            cost = None
            rep = res.get("budget_report", "")
            if "cost=$" in rep:
                with contextlib.suppress(ValueError):
                    cost = float(rep.split("cost=$")[1].split()[0])
            rows.append({**c, "answer": res.get("answer", "")[:600],
                         "tool_log": res.get("tool_log", []),
                         "retrieved_full": rt._retrieved_text(res),
                         "attack_fired": rt._attack_fired(c, res),
                         "stopped_because": res.get("stopped_because"),
                         "n_tool_calls": res.get("n_tool_calls", 0),
                         "filter_notes": res.get("filter_notes", []),
                         "cost_usd": cost,
                         "attack_succeeded": rt.attack_succeeded(c, res)})
    finally:
        rt.restore_corpus()

    m = rt._metrics(rows, attacks, controls, latencies)
    m["layers"] = sorted(layers)
    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"metrics": m, "rows": rows}, indent=2,
                                ensure_ascii=False), encoding="utf-8")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=10,
                    help="per-case tool budget. 10 rather than the suite's 6: "
                         "C03 legitimately needs 6 searches to compare the 2024 "
                         "and 2026 timelines, and at 6 it is starved before it "
                         "answers -- a real customer denied service that looks "
                         "exactly like a guard false positive.")
    ap.add_argument("--save-prefix", default="reports/lab6_layer")
    args = ap.parse_args()

    results: dict[str, dict] = {}

    print("=" * 92)
    print("D1 -- cumulative layers (what the whole stack buys)")
    print("=" * 92)
    print(f"{'layers':<26}{'block':>8}{'FP':>7}{'privileged':>12}"
          f"{'cost/q':>10}{'p95 ms':>9}")
    for n in range(0, 6):
        cfg = set(range(1, n + 1))
        m = run(cfg, args.max_calls,
                f"{args.save_prefix}_cum{n}.json")
        results[LAYERS[n]] = m
        print(f"{LAYERS[n]:<26}{m['block_rate']:>8.2f}"
              f"{m['false_positive_rate']:>7.2f}{m['privileged_calls']:>12}"
              f"{m['cost_per_query']:>10.5f}{m['p95_ms']:>9.0f}")

    print()
    print("=" * 92)
    print("D2 -- each layer ALONE (isolates its own contribution)")
    print("=" * 92)
    print(f"{'layer alone':<26}{'block':>8}{'FP':>7}{'privileged':>12}"
          f"{'cost/q':>10}{'p95 ms':>9}")
    for n in range(1, 6):
        m = run({n}, args.max_calls, f"{args.save_prefix}_only{n}.json")
        results[f"only {LAYERS[n]}"] = m
        print(f"{'only ' + LAYERS[n]:<26}{m['block_rate']:>8.2f}"
              f"{m['false_positive_rate']:>7.2f}{m['privileged_calls']:>12}"
              f"{m['cost_per_query']:>10.5f}{m['p95_ms']:>9.0f}")

    Path(ROOT / "reports/lab6_d1_table.json").write_text(
        json.dumps({"max_calls": args.max_calls, "results": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> reports/lab6_d1_table.json")


if __name__ == "__main__":
    main()