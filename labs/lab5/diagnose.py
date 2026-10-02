#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --precompute
    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.

`--precompute` runs FIRST and costs money: it measures the two signals
lab4.json does not carry (was the gold document in the top 30? does the
generator get it right on gold context?) and caches them to
reports/lab5_signals.json. Classification then reads that cache and is free.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cost import Budget  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402

# Imported from Lab 4 rather than re-declared, so the probe below cannot drift
# away from the configuration that actually produced reports/lab4.json.
from labs.lab4.evaluate import (  # noqa: E402
    FINAL_K,
    RETRIEVE_K,
    build_retriever,
    judge_correctness,
)
from labs.lab4.rag import answer_with_gold_context  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test. Crude keyword overlap, deliberately.

    TODO: this is a weak test -- it will pass on a paraphrase and fail on a
    numeric answer expressed differently. Improve it, and say in your report
    how you know your improvement is better.
    """
    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not text:
        return False
    tokens = [t for t in gold_answer.lower().split() if len(t) > 4]
    if not tokens:
        return True
    return sum(1 for t in tokens if t.strip(".,;()") in text) / len(tokens) > 0.4


def _gold_chunk(retriever, gold_answer: str, gold_docs: list[str]):
    """The chunk most likely to contain the gold answer.

    Picks by token overlap with the gold answer, so it works even when the
    answer is a number embedded in surrounding prose. Ties break towards the
    shortest chunk, on the theory that a chunk repeating the answer across many
    paragraphs is less likely to be the one that was cut badly.
    """
    toks = {t for t in gold_answer.lower().split() if len(t) > 3}
    if not toks:
        return None
    best, best_key = None, (0.0, 0)
    for c in retriever.chunks:
        if c.doc_id not in gold_docs:
            continue
        overlap = sum(1 for t in toks if t in c.text.lower()) / len(toks)
        key = (overlap, -len(c.text))
        if overlap > 0 and key > best_key:
            best, best_key = c, key
    return best


def _probe_own_text(retriever, gold_answer: str, gold_docs: list[str]) -> bool | None:
    """T4 §5 mode-3 test: can the gold chunk retrieve ITSELF by its own text?

    True  -> the chunk is intact and findable, so the original query was the
             problem (mode 3).
    False -> the chunk cannot be found even when you search for exactly what it
             says, which points at chunking rather than the query (mode 2 --
             still worth your eyes before you believe it).
    None  -> no gold chunk could be located at all.
    """
    chunk = _gold_chunk(retriever, gold_answer, gold_docs)
    if chunk is None:
        return None
    hits = retriever.search(chunk.text, k=30)
    return any(h.chunk.chunk_id == chunk.chunk_id for h in hits)


def load_signals(path: Path) -> dict[str, dict]:
    """Read the precompute cache. Empty dict when it does not exist yet."""
    if not path.exists():
        return {}
    out = {}
    for rec in json.loads(path.read_text(encoding="utf-8")):
        qid = rec.get("id")
        if qid is not None:
            out[qid] = rec
    return out


def precompute(failures: list[dict], questions: dict[str, dict],
               corpus: dict[str, str], path: Path, *, force: bool = False) -> None:
    """Measure the two signals the classification needs and lab4.json lacks.

    lab4.json records only the FINAL context (`retrieved`, i.e. the final_k
    chunks that survived), so it cannot answer "was the gold document in the
    top 30?" -- the question that separates modes 3/4 from mode 6. And it
    records nothing about what happens on gold context, which is the whole
    mode-6 test.

    Two measurements per failure:
      in_top_30 / gold_first_rank -- re-run the SAME retriever at k=30.
      gold_correctness           -- generate on gold context only, then judge.
      gold_chunk_in_final        -- did the chunk holding the answer survive
      all_relevant_in_final      -- did EVERY relevant document survive
      missing_relevant_docs      -- and if not, which ones

    On doc-level vs chunk-level: `in_final_k` asks whether any chunk from any
    gold document reached the final context. That is far too coarse. Q22 needs
    BOTH maternity-benefits AND dependents-and-family-floater; the final
    context held four maternity-benefits chunks and no dependents-and-family-
    floater at all, so doc-level reported "present" for a document that was
    entirely absent. Doc-level presence is also trivially satisfied whenever a
    gold doc happens to contribute two chunks. Hence `gold_chunk_in_final` and,
    more importantly, `missing_relevant_docs` -- which is the test that actually
    separates mode 4 from mode 6 on multi-hop and aggregation questions.

    On the ordering of the mode-6 branch: a refusal on an answerable question is
    NOT automatically a non-generation failure. Q20 refuses on gold context too
    (generation); Q41 and Q43 answer correctly on gold context but refuse on the
    real context (also generation, but a wrongful refusal rather than a wrong
    answer). An early `if row["refused"]` guard hid all three. The tree must
    reach mode 6 first.

    The final context is reconstructed exactly as rag.answer_question built it:
    search(k=RETRIEVE_K) then [:FINAL_K], with no reranker. Dense search over
    cached embeddings is deterministic, so this reproduces Lab 4's context
    without re-generating anything.

    Cached to reports/lab5_signals.json because both measurements cost money.
    Re-running this without --force reuses whatever is already on disk.
    """
    done = {} if force else load_signals(path)
    todo = [r for r in failures if r["id"] not in done]
    if not todo:
        print(f"all {len(failures)} signals already cached -> {path}")
        return

    retriever = build_retriever()
    n_cached = len(failures) - len(todo)
    print(f"precomputing {len(todo)} signals "
          f"({n_cached} cached, reranker=None so mode 5 is unreachable)")

    with Budget(limit_usd=1.00, label="lab5-precompute"):
        for r in todo:
            q = questions[r["id"]]
            gold_docs = [d for d in q["relevant_docs"] if d in corpus]

            # Which rank did a gold document first appear at, out of 30?
            hits = retriever.search(q["question"], k=30)
            rank = None
            for h in hits:
                if h.chunk.doc_id in gold_docs:
                    rank = h.rank
                    break

            # The gold chunk itself, and whether it survived into the context.
            gold_chunk = _gold_chunk(retriever, q["gold_answer"], gold_docs)
            final_hits = retriever.search(q["question"], k=RETRIEVE_K)[:FINAL_K]
            final_docs = {h.chunk.doc_id for h in final_hits}
            final_ids = {h.chunk.chunk_id for h in final_hits}
            # Which relevant documents never made it into the final context?
            # A multi-hop or aggregation question needs ALL of them, so this is
            # the test that separates mode 4 from mode 6 on those kinds.
            missing_docs = [d for d in q["relevant_docs"] if d not in final_docs]

            # The mode-6 test: same generator, no retrieval.
            rec: dict = {
                "id": r["id"],
                "in_top_30": rank is not None,
                "gold_first_rank": rank,
                "in_final_k": any(d in r.get("retrieved", []) for d in gold_docs),
                "gold_chunk_id": gold_chunk.chunk_id if gold_chunk else None,
                "gold_chunk_doc": gold_chunk.doc_id if gold_chunk else None,
                "gold_chunk_in_final": (gold_chunk.chunk_id in final_ids
                                        if gold_chunk else None),
                "all_relevant_in_final": not missing_docs,
                "missing_relevant_docs": missing_docs,
                "reranker": None,
                "dropped_by_reranker": False,  # Lab 4 ran reranker=None
                "k": RETRIEVE_K,
                "final_k": FINAL_K,
            }
            if gold_docs:
                g = answer_with_gold_context(
                    q["question"], [corpus[d] for d in gold_docs], strictness="lenient")
                score = judge_correctness(q["question"], g.text, q["gold_answer"])
                rec["gold_correctness"] = score
                rec["gold_refused"] = g.refused
            else:
                # No relevant document: mode 1 territory, and the gold-context
                # test is undefined rather than failed.
                rec["gold_correctness"] = None
                rec["gold_refused"] = None
            # The gold document never surfaced. That is mode 3 or mode 2, and
            # the discriminator is T4 §5's verbatim test: search using the gold
            # chunk's OWN text. If that finds it, the chunk is fine and the
            # original query was the problem (mode 3). If it does not, the chunk
            # itself is unfindable (mode 2 -- it may straddle a boundary).
            rec["gold_chunk_own_text_finds_it"] = None
            if rank is None and gold_docs:
                rec["gold_chunk_own_text_finds_it"] = _probe_own_text(
                    retriever, q["gold_answer"], gold_docs)
            done[r["id"]] = rec
            print(f"  {r['id']:<5} in_top_30={str(rec['in_top_30']):<5} "
                  f"rank={str(rec['gold_first_rank']):<5} "
                  f"gold_correctness={rec['gold_correctness']}"
                  + (f"  own_text={rec['gold_chunk_own_text_finds_it']}"
                     if rec["gold_chunk_own_text_finds_it"] is not None else ""))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"\nsaved {len(done)} signals -> {path}")


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None,
             signals: dict | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    TODO: complete the branches marked TODO. Follow the tree in the handout;
    do not invent your own ordering, because the ordering is what makes the
    modes mutually exclusive.

    `signals` carries the measurements from precompute(); see reports/
    lab5_signals.json. Any of them may be None when the cache was not built,
    in which case the branch is skipped rather than guessed.
    """
    sig = signals or {}
    gold_score = sig.get("gold_correctness")
    # An explicit keyword argument wins over the cache entry, so classify() can
    # still be called directly in a REPL with hand-supplied values.
    if in_top_30 is None:
        in_top_30 = sig.get("in_top_30")
    if dropped_by_reranker is None:
        dropped_by_reranker = sig.get("dropped_by_reranker")

    # Mode 7 first: right answer, wrong citation. Check this before anything
    # else, because a mode-7 failure is not a retrieval failure at all.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    # NB: there is deliberately NO `if row["refused"]` guard here, even though
    # a refusal looks like it should not be a generation failure. A refusal on
    # an ANSWERABLE question is exactly the case where the gold-context test is
    # most informative: Q20 refuses on gold context too (so mode 6), while Q41
    # and Q43 answer correctly on gold context but refuse on the real one (so
    # their refusal threshold, not retrieval, is at fault). Bailing out early
    # on `refused` hides all three behind needs_human_check.
    #
    # The genuinely un-discriminating cases -- no relevant document at all --
    # are already caught by mode 1 above, because answer_in_corpus returns
    # False when relevant_docs is empty.

    # Mode 6: does gold context fix it?
    # TODO: if gold_context_fixes_it is False, this is a generation failure.
    #       Note the direction -- gold context FIXING the answer means
    #       RETRIEVAL was at fault, not generation. People get this backwards.
    #
    # Read the test the right way round: if the generator is handed the gold
    # documents and STILL gets it wrong, no retriever could have saved it.
    if gold_score is not None and gold_score < 2:
        return 6, (f"wrong even on gold context (correctness {gold_score}/2), "
                   "so retrieval was never the problem")

    # Narrow guard, AFTER mode 6: the gold-context test is genuinely undefined
    # when there was no gold context to run (no relevant documents), so the
    # tree cannot continue on its own evidence.
    if gold_score is None:
        return 2, ("needs_human_check: gold-context test undefined (no relevant "
                   "document) -- inspect by hand")

    # Otherwise gold context DID fix it, which means retrieval was at fault.
    # Everything below localises *where* in retrieval.
    #
    # TODO Mode 4/5: gold doc in top 30 but not in the final k
    #       -> 5 if the reranker dropped it, else 4
    #
    # Retrieval lost something when EITHER the answer chunk was crowded out OR a
    # relevant document never arrived. Both are mode 4, and the two checks are
    # independent -- do not collapse them into one:
    #   Q22/Q35  every gold doc was in the top 30, but one was crowded out of
    #            k=8 while the answer chunk itself did arrive.
    #   Q23/Q29/Q32 every relevant DOCUMENT is present in the final context, but
    #            the specific chunk holding the answer was crowded out. Presence
    #            of the document is not presence of the passage.
    # Only when BOTH the answer chunk arrived and every relevant document
    # arrived did retrieval deliver everything, and a wrong answer is then
    # generation's fault rather than retrieval's.
    chunk_in_final = sig.get("gold_chunk_in_final")
    all_rel = sig.get("all_relevant_in_final")
    missing = sig.get("missing_relevant_docs")

    if in_top_30 is True and (missing or chunk_in_final is False):
        if dropped_by_reranker:
            return 5, ("gold doc was in the candidate pool and the reranker "
                       "dropped it")
        why = []
        if missing:
            why.append(f"{missing} never reached the final context")
        if chunk_in_final is False:
            why.append("the chunk holding the answer was crowded out")
        return 4, (f"all gold docs were in the top 30, but "
                   f"{'; '.join(why) or 'the needed passage was crowded out'} "
                   f"(final k={sig.get('final_k')})")

    # Everything retrieval had to deliver, it delivered: the answer chunk is in
    # the final context and every relevant document came with it. The generator
    # had what it needed and still got it wrong.
    if in_top_30 is True and chunk_in_final is True and all_rel:
        return 6, ("every relevant document AND the answer chunk were in the "
                   "final context, and gold context still fixed the answer -- "
                   "retrieval delivered everything, so this is generation")

    # The gold document never surfaced at all.
    # TODO Mode 3: gold doc not even in the top 30. Confirm by searching for
    #       the gold chunk's own text -- if THAT retrieves it, the query is the
    #       problem (mode 3). If it does not, the chunk itself is unfindable
    #       (mode 2, needs your eyes).
    if in_top_30 is False:
        own = sig.get("gold_chunk_own_text_finds_it")
        if own is True:
            return 3, ("gold doc absent from top 30, but its own text retrieves "
                       "it -- the query was the problem")
        if own is False:
            return 2, ("needs_human_check: gold doc absent from top 30 AND its "
                       "own text does not retrieve it -- print the chunks around "
                       "the gold answer and confirm a boundary split")
        return 2, ("needs_human_check: gold doc absent from top 30 and no gold "
                   "chunk located -- print the chunks around the gold answer")

    # No signals cached: fall back to the hand-check bucket rather than
    # guessing a mode.
    if in_top_30 is None or gold_score is None:
        return 2, ("needs_human_check: run --precompute first; without the "
                   "gold-context test this cannot be classified")

    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def main() -> None:
    # pareto() draws bars with U+2588, which a cp1252 Windows console cannot
    # encode. Reconfigure stdout rather than degrade the provided chart.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(AttributeError, OSError, ValueError):
                stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    ap.add_argument("--precompute", action="store_true",
                    help="measure the gold-context and top-30 signals, then exit")
    ap.add_argument("--signals", default="reports/lab5_signals.json")
    ap.add_argument("--force", action="store_true",
                    help="recompute signals already on disk")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    sig_path = ROOT / args.signals
    if args.precompute:
        precompute(failures, questions, corpus, sig_path, force=args.force)
        return

    signals = load_signals(sig_path)
    missing = [r["id"] for r in failures if r["id"] not in signals]
    if missing and not signals:
        print(f"no signals cache at {args.signals}.\n"
              f"run:  python labs/lab5/diagnose.py --precompute\n", file=sys.stderr)
    elif missing:
        print(f"warning: {len(missing)} failure(s) have no cached signal "
              f"({', '.join(missing)}); re-run --precompute\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        mode, evidence = classify(r, q, corpus, signals=signals.get(r["id"], {}))
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    n_human = tally.get(2, 0)
    print(f"\n{n_human} case(s) marked needs_human_check are Part A2. Open them.")
    if 5 not in tally:
        print("mode 5 (reranker) is empty because Lab 4 ran with reranker=None; "
              "no reranker existed to drop anything.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
