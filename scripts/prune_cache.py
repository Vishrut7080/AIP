#!/usr/bin/env python3
"""Cut the offline replay set out of the scratch cache, and write it where git can see it.

    python scripts/prune_cache.py --report    # what is in the scratch; write nothing
    python scripts/prune_cache.py --prune     # record what replay touches, then write the replay set

WHY THIS EXISTS
---------------
README.md promises "data/replay/ is committed, so the service starts with no network
at all", and labs/lab7/RUNSHEET.md Part D says "Commit the cache". Neither was
true. The CI gate and tests replay the golden set under AIP_OFFLINE=1, so on a
fresh clone every request was a CacheMiss: three tests failed with a 503 that
looked like a service bug, and a green build proved nothing.

The obvious fix -- commit .aip_cache/calls.sqlite3 as-is -- does not work. The
scratch cache is 144 MB: 8,617 embedding rows carrying 137 MB of base64 float32
payload, plus 5,217 chat rows. GitHub rejects any blob over 100 MB, which is
what broke the original push. And the attempt to fix that by un-tracking the
file took the offline replay with it.

So the two jobs are now on two paths, which is the whole point of this rewrite:

    .aip_cache/calls.sqlite3   scratch, 144 MB, gitignored, written by every call
    data/replay/calls.sqlite3  the replay set, ~1.3 MB, COMMITTED, written here

Same trade scripts/warm_cache.py already made for the index: 235 chunk vectors
as base64-in-SQLite cost 133 MB, and the identical matrix as one contiguous
float32 .npy costs 2.9 MB. So the answer is the same -- do not commit the
vectors, commit what is left. And because the committed path is not under
.aip_cache/, no gitignore rule can capture the 144 MB file and no ordering of
negations can quietly undo the commit.

WHY THIS RECORDS INSTEAD OF GUESSING
------------------------------------
The obvious filter is `kind='embed' AND input_type='query'`, and it does not
work: aip/embed.py sets `it = None` for any model where
_supports_input_type() is False, which includes gemini/gemini-embedding-001.
Every embedding row therefore stores `input_type: null` and query rows are
indistinguishable from passage rows. A filter written from the schema alone
silently deletes all 45 query vectors and offline retrieval breaks.

The alternative -- substring-match each chat request for a golden question --
is worse: it matches 3,176 of 5,217 rows on a heuristic, and misses the Lab 4
judge calls, whose prompts are built from the answer and the context rather than
from the question. Drop those and the gate cannot score correctness at all.

So this script derives the keep-set by MEASUREMENT: it instruments aip.cache.get,
replays the golden set through the real code paths -- gate.measure() and the
service's own POST /ask via TestClient -- and records every key that resolved.
That set is exactly what offline replay needs, by construction, and it re-derives
itself when the pipeline changes. Keeping it also drops the 5,000-odd chat rows
left behind by Labs 1-3, which nothing replays.

SAFETY
------
The source cache is never modified in place. Rows are deleted in a temporary
database, VACUUMed (SQLite does not shrink without it), and only then moved onto
data/replay/calls.sqlite3, so an interrupted run leaves the scratch cache intact
rather than a half-deleted one. The recorded keys must all still be present
after the prune, and --verify re-runs scripts/warm_cache.py's offline check,
because "the file got smaller" is not evidence that replay still works.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Must precede every aip import: aip.config reads the environment at import
# time, so this cannot be set later. Recording against a live provider would be
# meaningless anyway -- the point is to capture the OFFLINE replay set.
os.environ["AIP_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aip import cache  # noqa: E402
from aip.config import resolve_model, settings  # noqa: E402

DB_NAME = "calls.sqlite3"


def mb(n: int) -> str:
    return f"{n / 1024 / 1024:.2f} MB"


def db_path() -> Path:
    """The scratch cache: everything this machine has ever called."""
    return Path(settings.cache_dir) / DB_NAME


def replay_path() -> Path:
    """The committed replay set -- the only cache file that is in git."""
    return cache._REPLAY_DB_PATH


def golden_query_keys() -> set[str]:
    """Cache keys for the 45 golden questions' own query vectors.

    Computed with aip's own make_key rather than by inspecting stored rows,
    because the stored `text` is truncated to 200 characters and could not be
    hashed back into the key that produced it.
    """
    from labs.lab3.search import load_questions

    m = resolve_model("EMBED")
    return {
        cache.make_key("embed", {"model": m, "text": q["question"],
                                 "input_type": None})
        for q in load_questions(include_unanswerable=True)
    }


def record_replay_keys() -> set[str]:
    """Replay the golden set and return every cache key that resolved.

    Two entry points on purpose. gate.measure() is what CI runs, so its keys are
    mandatory. The TestClient pass covers the service's own POST /ask path --
    same answer_question call today, but if the service ever grows a stage the
    gate does not have, this is what keeps the demo working offline.
    """
    seen: set[str] = set()
    original_get = cache.get

    def recording_get(key: str):
        value = original_get(key)
        if value is not None:
            seen.add(key)
        return value

    cache.get = recording_get
    try:
        from labs.lab7 import gate

        metrics = gate.measure()
        print(f"  gate.measure()      -> {metrics['_n_questions']} questions, "
              f"{metrics['_n_cache_misses']} cache misses")

        from fastapi.testclient import TestClient

        from labs.lab3.search import load_questions
        from labs.lab7.service import app

        with TestClient(app) as client:
            for q in load_questions(include_unanswerable=True):
                r = client.post("/ask", json={"question": q["question"]})
                # 503 here would mean an uncached path, which is exactly what
                # this script exists to prevent -- so it is a hard failure.
                if r.status_code != 200:
                    raise SystemExit(
                        f"service returned {r.status_code} for {q['id']}: "
                        f"{r.text[:200]}\nThe replay set cannot be recorded until "
                        f"the golden set replays cleanly through POST /ask.")
        print("  POST /ask x45       -> all 200")
    finally:
        cache.get = original_get

    return seen | golden_query_keys()


def report() -> int:
    src = db_path()
    dst = replay_path()
    if not src.exists():
        print(f"no cache at {src}\n  run: python scripts/warm_cache.py  (online)")
        return 1
    conn = sqlite3.connect(src)
    print(f"{src}   {mb(src.stat().st_size)}   offline={settings.offline}\n")
    print("contents:")
    for kind, n, payload in conn.execute(
            "SELECT kind, COUNT(*), SUM(LENGTH(request)+LENGTH(response)) "
            "FROM calls GROUP BY kind ORDER BY 2 DESC"):
        print(f"  {kind:6} {n:6} rows   {mb(payload or 0):>10} of payload")
    total = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    print(f"  {'total':6} {total:6} rows")
    distinct = conn.execute(
        "SELECT DISTINCT json_extract(request,'$.input_type') FROM calls "
        "WHERE kind='embed'").fetchall()
    print(f"\n  embed input_type values stored: {distinct}")
    print("  (all null: _supports_input_type() is False for this model, so")
    print("   query rows cannot be told from passage rows in the row itself --")
    print("   which is why --prune records the replay set instead of filtering.)")
    conn.close()

    # The scratch is not the deliverable; the replay set is. Report both, so
    # "is my committed artefact still there" is answerable without a git command.
    if dst.exists():
        rconn = sqlite3.connect(f"{dst.as_uri()}?mode=ro", uri=True)
        try:
            n = rconn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        finally:
            rconn.close()
        print(f"\nreplay set: {dst}\n             {mb(dst.stat().st_size)}   "
              f"{n} rows   (COMMITTED)")
    else:
        print(f"\nreplay set: absent -- {dst}")
        print("  run: python scripts/prune_cache.py --prune")
    return 0


def prune(keep: set[str]) -> int:
    """Write the replay set: `keep` rows from the scratch, and nothing else."""
    src = db_path()
    dst = replay_path()
    before_bytes = src.stat().st_size
    conn = sqlite3.connect(src)
    total = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    conn.close()

    # The temp file lives beside the DESTINATION, not beside the source and not
    # in %TEMP%. os.replace is only reliably atomic within one directory, and
    # this repo sits inside OneDrive, which holds its own handles on files it is
    # syncing -- a cross-directory replace failed there with PermissionError
    # before this was tried. Beside the destination also means the source and
    # the committed file are never in the same directory operation.
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".sqlite3.pruning")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)

    tconn = sqlite3.connect(tmp)
    try:
        present = {r[0] for r in tconn.execute("SELECT key FROM calls")}
        missing = keep - present
        if missing:
            print(f"ERROR: {len(missing)} recorded key(s) are not in the cache; "
                  "aborting without touching anything.")
            for k in sorted(missing)[:5]:
                print(f"  {k}")
            return 1

        # Delete the COMPLEMENT of keep. Deleting `keep` itself -- the obvious
        # transcription of "keep these" -- deletes precisely the rows offline
        # replay needs and leaves every useless one behind, which is how this
        # first reported "kept 13659 of 13834 rows".
        drop = present - keep
        tconn.executemany("DELETE FROM calls WHERE key = ?",
                          [(k,) for k in sorted(drop)])
        tconn.commit()
        # VACUUM is what actually reclaims the bytes; without it the file keeps
        # its high-water mark and the prune saves nothing at all.
        tconn.execute("VACUUM")
        left = tconn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    finally:
        tconn.close()

    after_bytes = tmp.stat().st_size
    print(f"\nkept {left} of {total} rows")
    print(f"scratch  {mb(before_bytes)}   ({src})")
    # Phrased as a cut only when it is one. Re-running the prune on an already
    # pruned scratch keeps everything, and "a 100.0% cut" there is a lie.
    pct = 100 * after_bytes / before_bytes if before_bytes else 100.0
    print(f"replay   {mb(after_bytes)}   "
          + (f"({100 - pct:.1f}% smaller than scratch)"
             if pct < 99.5 else "(scratch was already pruned; nothing to cut)"))
    if left != len(keep):
        tmp.unlink(missing_ok=True)
        print(f"ERROR: expected {len(keep)} rows, wrote {left}. "
              "Nothing written.")
        return 1

    try:
        os.replace(tmp, dst)
    except PermissionError:
        # OneDrive or an antivirus scanner is holding the destination open.
        # Overwriting in place is not atomic, so keep the pruned file until the
        # copy has succeeded rather than unlinking it first.
        print("  os.replace was denied (OneDrive holding the file); "
              "overwriting in place instead")
        shutil.copyfile(tmp, dst)
        tmp.unlink(missing_ok=True)

    print(f"\nwrote {dst}   ({mb(dst.stat().st_size)})")
    print("git add it -- it is the deliverable. The scratch cache stays ignored:")
    print(f"  git add {dst.relative_to(ROOT).as_posix()}")
    print("\nNow prove it. A smaller file is not evidence that replay still works:")
    print("  AIP_OFFLINE=1 python scripts/warm_cache.py --verify")
    print("  AIP_OFFLINE=1 python labs/lab7/gate.py")
    print("and confirm _n_cache_misses is 0 in reports/gate_metrics.json.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--report", action="store_true",
                    help="print the scratch cache's composition; write nothing")
    ap.add_argument("--prune", action="store_true",
                    help="record the offline replay set, then write it to data/replay/")
    args = ap.parse_args()
    if not (args.report or args.prune):
        ap.error("choose --report or --prune")
    if args.report:
        return report()
    if not db_path().exists():
        return report()

    print("recording the offline replay set (golden set, via the real code paths)")
    keep = record_replay_keys()
    print(f"  {len(keep)} keys required by offline replay\n")
    return prune(keep)


if __name__ == "__main__":
    sys.exit(main())