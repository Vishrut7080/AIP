#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json   # deployed strictness = lenient (C4)
    python labs/lab4/evaluate.py --full --strictness strict --save reports/lab4_strict.json
    python labs/lab4/evaluate.py --full --strictness default --save reports/lab4_default.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --single Q37 --strictness lenient   # cheap spot-check
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import judge_agreement, llm_judge  # noqa: E402
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"

# My Lab 3 winning configuration (reports/report3.md, reports/lab3_sweeps.json):
#   chunking  -> markdown-aware, size=400, no overlap arg (markdown has none)
#   retriever -> exact dense (DenseRetriever)
# The reference (aip.rag.RAG) used final_k=5. Lab 4 widens the context to
# final_k=8 because Q37's partial answer needs the chunk at retrieval rank 8
# (plans-overview; see the C2 section of reports/report4.md). This is still
# "your retriever and your final_k" -- it just is no longer Lab 3's final_k.
RETRIEVE_K = 16
FINAL_K = 8

# ---------------------------------------------------------------------------
# D1: judges (yours). The shipped rubrics in aip.evals are deliberately
# improvable; using them verbatim scores the cases below wrong.
# ---------------------------------------------------------------------------
# Faithfulness must additionally handle three cases the shipped rubric
# misses:
#   - a PARTIAL refusal (answers the grounded part, refuses the rest)
#   - a claim that cites correctly but is paraphrased into a STRONGER claim
#     than the context supports
#   - an answer right about the world but wrong about the context
RUBRIC_FAITHFULNESS = """\
You are grading whether an ANSWER is fully supported by the provided CONTEXT.

Rules:
- Judge support only. Do NOT judge whether the answer is helpful, well written,
  or matches your own knowledge.
- An answer is unsupported if it states anything the context does not contain,
  even if that statement is true in the real world.
- A valid citation does NOT excuse an unsupported claim: a paraphrase that
  strengthens the source (adds a limit, a scope, a number, or certainty the
  source does not state) is unsupported.
- A partial refusal is SUPPORTED when the part that is answered is grounded in
  the context and the part that is refused is genuinely absent from it.
- A full refusal is SUPPORTED when the context genuinely lacks the answer.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON: {{"score": 0 or 1, "unsupported_claims": [..], "reason": "one sentence"}}
"""

# Correctness must handle refusal explicitly. The shipped rubric scores
# "refuses when the reference answers" as 0 but says nothing about the other
# direction: on an unanswerable question a refusal is full marks and an
# invented confident answer is zero.
RUBRIC_CORRECTNESS = """\
Compare a CANDIDATE answer to a REFERENCE answer for the same question.

Refusal handling (takes priority):
- If the REFERENCE refuses, a candidate that refuses (in whole or the unsupported
  part) scores 2; a candidate that answers confidently where it should refuse
  scores 0.
- If the REFERENCE answers, a candidate that refuses scores 0.

Score 2 = same substantive content as the reference (wording may differ).
Score 1 = partially correct: some correct content, but omits something the
          reference states, or adds something the reference contradicts.
Score 0 = wrong, or an answer that should have refused.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def _judge_score(verdict: dict, default: int = 0) -> int | None:
    """A parse failure is missing data, not a failing answer (the war story).
    Return None so the metric is excluded rather than biased down to 0."""
    if verdict.get("parse_error"):
        return None
    return int(verdict.get("score", default))


def judge_faithfulness(answer_text: str, context: str) -> int | None:
    """Improve on JUDGE_RUBRIC_FAITHFULNESS (see RUBRIC_FAITHFULNESS)."""
    verdict = llm_judge(RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text), tier="LARGE")
    return _judge_score(verdict)


def judge_correctness(question: str, candidate: str, reference: str) -> int | None:
    """0, 1 or 2, with refusals scored explicitly (see RUBRIC_CORRECTNESS)."""
    verdict = llm_judge(RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE")
    return _judge_score(verdict)


def build_retriever():
    """My Lab 3 winning configuration, not the placeholder.

    markdown-aware chunking at 400 chars (235 chunks) + exact dense retrieval.
    `markdown_chunks` takes no overlap parameter, so size=400 alone is the
    configuration that won Part A (report3.md: hit_rate@1 0.7857, ndcg@10 0.8527).
    """
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks)


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else 0.0


# ---------------------------------------------------------------------------
def run_full(save: str = "", strictness: str = "lenient") -> None:
    """C4 product decision: `lenient` is deployed (see reports/report4.md).
    An insurance helpdesk must answer what it can; the partial-refusal rule
    still covers the unsupported remainder. --full without flags = the
    shipping configuration."""
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []

    with Budget(limit_usd=1.00, label=f"lab4-full-{strictness}") as b:
        for q in questions:
            a = answer_question(q["question"], retriever,
                                k=RETRIEVE_K, final_k=FINAL_K,
                                strictness=strictness)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "n_repairs": a.n_repairs,
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nstrictness = {strictness}")
    print(f"n = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {statistics.fmean(r['citations_valid'] for r in rows):.3f}"
          "   (target 1.000)")
    print(f"faithfulness        {_mean([r['faithfulness'] for r in rows]):.3f}")
    print(f"correctness (0-2)   {_mean([r['correctness'] for r in ans]):.3f}"
          f"  normalised {_mean([r['correctness'] for r in ans]) / 2:.3f}")
    print(f"repair rate         {statistics.fmean(r['n_repairs'] > 0 for r in rows):.3f}"
          f"   ({sum(r['n_repairs'] for r in rows)} repairs / {len(rows)} questions)")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({len(refusals)} refusals total)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {_mean([r['correctness'] for r in sub])/2:.3f}"
              f"  n={len(sub)}")
    if refusals and ans:
        wrong = [r for r in refusals if not r["unanswerable"]]
        if wrong:
            print(f"\nwrongful refusals: {[r['id'] for r in wrong]}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_single(question_id: str, strictness: str = "default",
               k: int = RETRIEVE_K, final_k: int = FINAL_K) -> None:
    """Cheap spot-check of ONE question: prints the retrieved docs and answer,
    no judges. Used for the C2 (Q37) verification without spending a full run."""
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    if question_id not in questions:
        print(f"unknown id {question_id!r}; known ids: {sorted(questions)}")
        return
    q = questions[question_id]
    retriever = build_retriever()
    a = answer_question(q["question"], retriever, k=k, final_k=final_k,
                        strictness=strictness)
    print(f"\n[{question_id}] {q['question']}")
    print(f"kind={q['kind']} relevant={q['relevant_docs']}")
    print("retrieved:", ", ".join(h.doc_id for h in a.hits))
    print(f"refused={a.refused} citations_valid={a.citations_valid} "
          f"n_repairs={a.n_repairs}")
    print("---\n" + a.text + "\n---")


def run_gold_context(strictness: str = "lenient") -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    # Same answerable definition as run_full: exclude the FIVE 'unanswerable'
    # questions (Q36-Q40). The default load_questions() only drops the three
    # with no relevant docs, which would wrongly keep Q37/Q40 (both correct
    # refusals = full marks) and inflate B over 42 rows instead of 40.
    questions = [q for q in load_questions(include_unanswerable=True)
                 if q["relevant_docs"] and q["kind"] != "unanswerable"]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever,
                                k=RETRIEVE_K, final_k=FINAL_K,
                                strictness=strictness)
            retrieved_scores.append(
                judge_correctness(q["question"], a.text, q["gold_answer"]) / 2)
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus],
                strictness=strictness)
            gold_scores.append(
                judge_correctness(q["question"], g.text, q["gold_answer"]) / 2)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        pairs = [(h[f"human_{field}"], machine[h["id"]][field])
                 for h in human if h[f"human_{field}"] is not None
                 and machine[h["id"]][field] is not None]
        if not pairs:
            print(f"{field}: no (judge, human) label pairs yet")
            continue
        h = [p[0] for p in pairs]
        m = [p[1] for p in pairs]
        print(f"{field}: {judge_agreement(m, h)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--strictness", default="lenient",
                    choices=["default", "lenient", "strict"])
    ap.add_argument("--single", default="")
    ap.add_argument("--k", type=int, default=RETRIEVE_K)
    ap.add_argument("--final-k", type=int, default=FINAL_K)
    a = ap.parse_args()
    if a.full:
        run_full(a.save, strictness=a.strictness)
    if a.gold_context:
        run_gold_context()
    if a.single:
        run_single(a.single, strictness=a.strictness, k=a.k, final_k=a.final_k)
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if not any([a.full, a.gold_context, a.single, a.calibrate, a.kappa]):
        ap.print_help()