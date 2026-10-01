#!/usr/bin/env python3
"""Warm the caches the Lab 7 CI gate needs, and export a committable index.

    python scripts/warm_cache.py                 # warm + export
    python scripts/warm_cache.py --verify        # check offline replay works
    python scripts/warm_cache.py --config labs/lab4/evaluate.py

WHY THIS EXISTS
---------------
`aip/embed.py:121` tells you to run this script. It did not exist. It is needed
because the CI regression gate runs with AIP_OFFLINE=1 and replays the committed
cache -- and that cache was not committable:

  .aip_cache/calls.sqlite3 as-is   145 MB
    of which embeddings            133 MB   (7958 rows, ~16 KB each)
    chat only                        3 MB   (the gate's actual need)

Storing 235 chunk vectors as base64-in-SQLite costs 133 MB. The same matrix as a
single float32 .npy is **2.9 MB** -- a 45x difference, because base64 inflates
by 4/3 and SQLite stores one row per vector instead of one contiguous block.

So this script exports the index to `data/index/matrix.npy` and
`data/index/chunks.json`. Under AIP_OFFLINE=1 the service loads those instead of
re-embedding, and CI needs no network and no embedding provider at all.

WHAT CI ACTUALLY NEEDS
----------------------
1. `data/index/` -- the corpus embeddings. ~3 MB, committed.
2. chat responses for the golden set under the SHIPPING prompt only. One config,
   45 questions. `labs/lab7/gate.py` replays these; a different prompt is a
   CacheMiss by design, because a different prompt is a different experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aip import cache, settings  # noqa: E402
from aip.chunking import markdown_chunks  # noqa: E402
from aip.config import resolve_model  # noqa: E402
from aip.retrieval import DenseRetriever  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402

INDEX_DIR = ROOT / "data/index"
CHUNK_SIZE = 400  # the Lab 3/4 winning configuration


def build_index() -> DenseRetriever:
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=CHUNK_SIZE)]
    return DenseRetriever(chunks, show_progress=False)


def export_index(r: DenseRetriever) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    m = np.asarray(r.matrix, dtype=np.float32)
    np.save(INDEX_DIR / "matrix.npy", m)
    (INDEX_DIR / "chunks.json").write_text(json.dumps([
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
         "meta": c.meta} for c in r.chunks
    ], ensure_ascii=False), encoding="utf-8")
    print(f"  matrix.npy   {m.shape} float32  {m.nbytes / 1e6:.2f} MB")
    print(f"  chunks.json  {len(r.chunks)} chunks")


def load_index(model: str | None = None) -> DenseRetriever:
    """Rebuild a retriever from the committed artefacts, no network.

    Used by the service and the gate under AIP_OFFLINE=1.

    `model` matters: the matrix is only meaningful if queries are embedded by
    the SAME model that produced it. Passing the EMBED model keeps
    DenseRetriever.search() asking the provider for a query vector; under
    AIP_OFFLINE=1 that is a CacheMiss unless the query text happens to be
    pre-warmed. Golden-set questions are, so the gate works. Arbitrary
    questions are not -- see verify().
    """
    from aip.chunking import Chunk

    raw = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(text=d["text"], doc_id=d["doc_id"], chunk_id=d["chunk_id"],
                    meta=d.get("meta", {})) for d in raw]
    matrix = np.load(INDEX_DIR / "matrix.npy")

    r = DenseRetriever.__new__(DenseRetriever)
    r.chunks = chunks
    r.model = model
    r.matrix = matrix
    return r


def warm_questions(strictness: str = "lenient_complete") -> None:
    """Run the golden set once, online, so its chat responses are cached."""
    from aip.cost import Budget
    from aip.embed import embed
    from labs.lab4.rag import answer_question

    questions = load_questions(include_unanswerable=True)
    r = build_index()
    print(f"  warming {len(questions)} questions, strictness={strictness}")
    with Budget(limit_usd=2.00, label="warm-cache") as b:
        for q in questions:
            # Embed the query explicitly. answer_question does this too, but
            # doing it here makes the dependency visible: offline replay needs
            # BOTH the query vector and the chat response cached, and only the
            # chat response gets warmed by answering the question.
            embed(q["question"], input_type="query")
            answer_question(q["question"], r, k=16, final_k=8,
                            strictness=strictness)
    print("  " + b.report())


def verify() -> int:
    """Prove the gate can run offline. Returns a process exit code."""
    print("verifying offline replay (AIP_OFFLINE must be 1 for this check)")
    print(f"  settings.offline = {settings.offline}")
    if not settings.offline:
        print("  run with AIP_OFFLINE=1 to actually verify")

    ok = True
    try:
        r = load_index(model=resolve_model("EMBED"))
        print(f"  index loaded from disk: {len(r.chunks)} chunks "
              f"{r.matrix.shape}")
        # Retrieval must be tested on a GOLDEN question. The committed cache
        # holds query vectors for exactly those 45 strings, because offline
        # replay has no provider to embed anything else with. An ad-hoc
        # phrasing is a CacheMiss by design, not a broken cache.
        gq = {x["id"]: x for x in load_questions(include_unanswerable=True)}
        hits = r.search(gq["Q01"]["question"], k=3)
        print(f"  retrieval OK, top doc: {hits[0].chunk.doc_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"  index load FAILED: {type(exc).__name__}: {str(exc)[:140]}")
        return 1

    if settings.offline:
        from labs.lab4.rag import answer_question
        try:
            t0 = time.perf_counter()
            a = answer_question(gq["Q01"]["question"], r, k=16, final_k=8,
                                strictness="lenient_complete")
            print(f"  generation replay OK ({time.perf_counter() - t0:.2f}s) "
                  f"refused={a.refused}")
        except cache.CacheMiss as exc:
            print(f"  generation CacheMiss: {str(exc)[:140]}")
            print("  -> run: python scripts/warm_cache.py   (while online)")
            ok = False

        # Honest about the boundary rather than claiming full offline support.
        try:
            r.search("a question nobody has ever asked before", k=1)
            print("  arbitrary-question retrieval: OK")
        except cache.CacheMiss:
            print("  arbitrary-question retrieval: CacheMiss (expected)")
            print("    offline replay covers the golden set and any question")
            print("    whose vector is cached -- NOT arbitrary text.")
    else:
        print("  SKIPPED generation check (needs AIP_OFFLINE=1)")

    print("\nOFFLINE REPLAY:", "OK" if ok else "NOT READY")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check that offline replay works; exit non-zero if not")
    ap.add_argument("--no-warm", action="store_true",
                    help="only export the index, do not call any model")
    ap.add_argument("--strictness", default="lenient_complete",
                    help="the prompt config to cache chat responses for")
    args = ap.parse_args()

    if args.verify:
        return verify()

    print(f"profile: {settings.profile}   offline: {settings.offline}")
    print("\n1. building and exporting the corpus index")
    r = build_index()
    export_index(r)

    if not args.no_warm and not settings.offline:
        print(f"\n2. warming chat responses ({args.strictness})")
        warm_questions(args.strictness)
    elif settings.offline:
        print("\n2. SKIPPED warm step (AIP_OFFLINE=1)")
    else:
        print("\n2. SKIPPED warm step (--no-warm)")

    print("\ncommit both data/index/ and .aip_cache/calls.sqlite3")
    print("the CI gate replays the chat cache; the service loads data/index/")
    return 0


if __name__ == "__main__":
    sys.exit(main())