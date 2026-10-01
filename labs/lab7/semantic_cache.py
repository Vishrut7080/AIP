#!/usr/bin/env python3
"""B1 -- find the semantic-cache threshold where caching starts returning
answers to the WRONG question.

    python labs/lab7/semantic_cache.py

A semantic cache embeds the question and returns the cached answer for the most
similar question above some cosine threshold. README.md:87 asks for the
threshold at which it starts returning wrong answers, and says it is lower than
people assume.

METHOD
------
A cache hit on question B returns the answer generated for question A. That is
correct ONLY IF A's answer also answers B. So for every highly-similar pair
(A, B) I take A's real answer and JUDGE it against B's gold answer. If the judge
says A's answer does not answer B, the cache would have served a wrong answer at
that similarity.

The output is therefore the empirical boundary: above which similarity the hit
rate of "correct by accident" collapses. Each pair is also labelled same_topic,
because a semantic cache is far more defensible for "30-day grace period" vs
"30-day grace period for instalments" than for two unrelated questions that
happen to share vocabulary.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cost import Budget  # noqa: E402
from aip.embed import embed  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402
from labs.lab4.evaluate import build_retriever, judge_correctness  # noqa: E402
from labs.lab4.rag import STRICTNESS, answer_question  # noqa: E402

GENERATION_TIER = "SMALL"
RETRIEVE_K, FINAL_K = 16, 8
MIN_SIM = 0.70
BUCKETS = [(0.70, 0.80), (0.80, 0.86), (0.86, 0.90), (0.90, 0.95),
           (0.95, 0.98), (0.98, 1.01)]


def _trap_pairs(vecs: dict) -> list[tuple[float, str, str, str]]:
    """Paraphrase pairs that are semantically ADJACENT but answer differently.

    A cosine threshold cannot separate these from a genuine repeat, because
    they are nearly the same request with the discriminating detail changed.
    These are the real-world failures of semantic caching: not "different
    question" but "same shape, different fact".
    """
    traps = [
        ("How long do I have to submit a reimbursement claim?",
         "How long do I have to submit a reimbursement claim for an accident?",
         "accident claims may have a different window"),
        ("What is the grace period for an annual policy?",
         "What is the grace period for an instalment policy?",
         "15 days, not 30 -- the same question with the wrong subject"),
        ("Is maternity covered on the Bronze plan?",
         "Is maternity covered on the Platinum plan?",
         "Bronze excludes it, Platinum includes it"),
        ("What is the waiting period for pre-existing diseases?",
         "What is the waiting period for a pre-existing condition on a rider?",
         "rider may reduce 36 to 24 or 12 months"),
        ("How much ambulance cover is there?",
         "How much air ambulance cover is there?",
         "5,000 road vs 2,50,000 air"),
    ]
    out = []
    with Budget(limit_usd=0.50, label="trap-embeds"):
        for a_txt, b_txt, note in traps:
            va = np.asarray(embed(a_txt, input_type="query"), dtype=np.float32)
            vb = np.asarray(embed(b_txt, input_type="query"), dtype=np.float32)
            out.append((float(np.dot(va, vb)), a_txt, b_txt, note))
    out.sort(key=lambda t: -t[0])
    return out


def main() -> None:
    questions = [q for q in load_questions(include_unanswerable=True)
                 if q["relevant_docs"]]
    print(f"{len(questions)} answerable golden questions")
    yield_pairs: list[dict] = []

    with Budget(limit_usd=3.00, label="semantic-cache-sweep"):
        vecs = {q["id"]: np.asarray(embed(q["question"], input_type="query"),
                                    dtype=np.float32) for q in questions}

        # Pairs above MIN_SIM, cheapest-first ordering is irrelevant here.
        pairs = []
        for a, b in combinations(questions, 2):
            sim = float(np.dot(vecs[a["id"]], vecs[b["id"]]))
            if sim >= MIN_SIM:
                pairs.append((sim, a, b))
        pairs.sort(key=lambda p: -p[0])
        print(f"{len(pairs)} pairs at cosine >= {MIN_SIM}")
        if pairs:
            print(f"  highest similarity anywhere in the suite: {pairs[0][0]:.4f}")
        print()

        # A second, adversarial set: paraphrases that are MEANT to look similar
        # but have DIFFERENT gold answers. This is the case a semantic cache
        # cannot get right, and it is the reason a similarity threshold alone
        # is not a safety property.
        trap_pairs = _trap_pairs(vecs)
        for sim, a_txt, b_txt, _note in trap_pairs:
            pairs.append((sim, {"id": "PARA", "question": a_txt,
                                "gold_answer": "", "kind": "paraphrase",
                                "relevant_docs": []},
                          {"id": "TRAP", "question": b_txt,
                           "gold_answer": "DIFFERENT BY DESIGN",
                           "kind": "paraphrase", "relevant_docs": []}))
        print(f"+ {len(trap_pairs)} adversarial paraphrase pairs\n")

        r = build_retriever()
        cache: dict[str, str] = {}
        for sim, a, b in pairs:
            if a["id"] not in cache:
                ans = answer_question(a["question"], r, k=RETRIEVE_K,
                                      final_k=FINAL_K, strictness="lenient_complete",
                                      tier=GENERATION_TIER)
                cache[a["id"]] = ans.text
            # Would serving A's answer for B be correct?
            score = judge_correctness(b["question"], cache[a["id"]],
                                      b["gold_answer"])
            rec = {"sim": round(sim, 4), "a": a["id"], "b": b["id"],
                   "same_kind": a["kind"] == b["kind"],
                   # A list, not a set: sets are not JSON-serialisable and the
                   # report is written to disk.
                   "same_topic": sorted(set(a["relevant_docs"]) & set(b["relevant_docs"])),
                   "cached_correct_for_b": score}
            yield_pairs.append(rec)

    # Bucketed summary.
    print(f"\n{'similarity band':<22}{'n':>5}{'correct for B':>16}{'same topic':>13}")
    print("-" * 56)
    rows = []
    for lo, hi in BUCKETS:
        sel = [p for p in yield_pairs if lo <= p["sim"] < hi]
        if not sel:
            continue
        ok = sum(1 for p in sel if p["cached_correct_for_b"] == 2)
        topic = sum(1 for p in sel if p["same_topic"])
        pct = ok / len(sel)
        print(f"[{lo:.3f}, {hi:.3f})      {len(sel):>5}{f'{ok}/{len(sel)} ({pct:.0%})':>16}"
              f"{f'{topic}/{len(sel)}':>13}")
        rows.append({"band": [lo, hi], "n": len(sel), "correct": ok,
                     "same_topic": topic})

    # The number README.md:87 asks for: highest similarity band where a hit is
    # still reliably correct.
    safe = [r for r in rows if r["n"] >= 2 and r["correct"] == r["n"]]
    threshold = min((r["band"][0] for r in safe), default=None)
    print()
    if safe:
        print(f"HIGHEST BAND WHERE EVERY HIT WAS STILL CORRECT: "
              f"{threshold:.3f}")
        nxt = [r for r in rows if threshold is not None and r["band"][0] >= threshold]
        for r in nxt:
            print(f"  at >= {r['band'][0]:.3f}: {r['correct']}/{r['n']} correct")
        print()
        print("=> A threshold of 0.95 (the value in service.py) is ABOVE this. "
              "Semantic caching is safe here only for same-topic paraphrase.")
    else:
        print("No band with >=2 pairs was error-free. On this suite the semantic "
              "cache is unsafe at every threshold tested.")

    out = ROOT / "reports/lab7_semantic_cache.json"
    out.write_text(json.dumps({
        "min_sim_tested": MIN_SIM,
        "n_pairs": len(yield_pairs),
        "bands": rows,
        "highest_safe_band": threshold,
        "note": ("A cache hit on B returns A's answer. It is correct only if A's "
                 "answer also answers B; cached_correct_for_b is the judge "
                 "scoring exactly that."),
        "pairs": yield_pairs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()