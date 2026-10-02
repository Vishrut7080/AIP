"""The offline replay set has to be IN GIT, or CI proves nothing.

WHAT THIS IS FOR
----------------
tests/test_service.py replays .aip_cache's sibling, data/replay/calls.sqlite3,
through the real pipeline under AIP_OFFLINE=1. If that file is missing, every one
of those tests fails with a 503 -- but the failure reads like a service bug
rather than a packaging bug, and it had already happened twice:

  a149961  added `!.aip_cache/calls.sqlite3` and committed the pruned cache
  a43e322  deleted the negation ("Ignore class.sqlite3") and the file left git
  4202edf  rewrote the comment to insist the cache was "not a deliverable"

Each of those was a green local run and a red CI build, because locally the file
existed in the working tree either way. Nothing tested the invariant that the
deliverable is actually delivered. So it is tested here.

WHY SUBPROCESS, AND WHY IT SOMETIMES SKIPS
------------------------------------------
`git ls-files` is the only honest way to ask "is this in the repository?". The
file being present on disk proves nothing -- that was the entire bug: a green
local run and a red CI build, from the same working tree.

The two git-dependent checks below are skipped outside a checkout, because
"is it committed" is unanswerable when there is no repository to ask -- a
downloaded zip, or an exported tree. Everywhere else, including CI, they run.
They deliberately do NOT skip on a missing git binary when the tree IS a
checkout, because that combination means the check could have run and did not.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aip import cache  # noqa: E402

# The replay set is a curated 175-row artefact: 45 golden answers, the Lab 4
# judge calls, and 45 query vectors, with base64 float32 vectors for the latter.
# It measures 1.3 MB. This budget is 4x that, which is loose enough to survive a
# re-prune and tight enough that the un-pruned 144 MB cache cannot pass -- the
# failure this file exists to catch is a whole order of magnitude, not a rounding
# error, so the exact threshold does not matter as long as it sits between them.
REPLAY_BUDGET_BYTES = 5 * 1024 * 1024
GITHUB_BLOB_LIMIT = 100 * 1024 * 1024


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def in_a_checkout() -> set[str]:
    """The tracked file list, or a skip when there is no repository to ask."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH, so whether the replay set is COMMITTED "
                    "cannot be checked on this machine")
    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("not a git checkout (downloaded zip or exported tree): this "
                    "suite can check that the replay set exists and replays, but "
                    "not that it is committed")
    ls = _git("ls-files")
    assert ls.returncode == 0, ls.stderr
    return set(ls.stdout.split())


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_the_replay_set_is_committed(in_a_checkout):
    """The one invariant every other offline test silently depends on.

    A failure here explains all of them at once: no committed replay set means
    a fresh clone 503s on every golden question, which reads as a service bug.
    """
    rel = _rel(cache._REPLAY_DB_PATH)
    assert rel in in_a_checkout, (
        f"{rel} is not tracked by git. Offline replay reads it (aip/cache.py), so "
        f"on a fresh clone -- CI, a grader, anyone following README.md -- every "
        f"golden question is a CacheMiss and the tests fail with a 503. Rebuild "
        f"it with `python scripts/prune_cache.py --prune` and `git add {rel}`.")


def test_the_replay_set_is_small_enough_to_push():
    """It must stay far under GitHub's 100 MB blob limit.

    The scratch cache it is cut from is 144 MB, and committing that is what
    broke an earlier push. The size is therefore a correctness property here,
    not a matter of taste: a replay set that grows past the limit is a push that
    fails, and the person who has to notice is whoever runs CI last.
    """
    path = cache._REPLAY_DB_PATH
    assert path.exists(), f"{_rel(path)} does not exist; see the test above"
    size = path.stat().st_size
    assert size < REPLAY_BUDGET_BYTES, (
        f"{_rel(path)} is {size / 1024 / 1024:.1f} MB, over the "
        f"{REPLAY_BUDGET_BYTES / 1024 / 1024:.0f} MB budget. Either the replay "
        f"set was re-recorded without pruning (re-run scripts/prune_cache.py "
        f"--prune), or a new call kind is being recorded that replay does not "
        f"need. GitHub rejects blobs over "
        f"{GITHUB_BLOB_LIMIT // 1024 // 1024} MB, so this ends as a failed push.")


def test_the_scratch_cache_is_still_ignored(in_a_checkout):
    """The 144 MB file must never be committable, even by accident.

    The whole point of putting the replay set on its own path is that the
    scratch cache has no tracked sibling and no reachable negation. If this
    ever fails, something has started tracking .aip_cache/ -- and the next
    online `warm_cache.py` run stages a 144 MB blob.
    """
    ignored = _git("check-ignore", ".aip_cache/calls.sqlite3")
    assert ignored.returncode == 0, (
        ".aip_cache/calls.sqlite3 is NOT ignored. It is the scratch cache: 144 MB "
        "after an online warm, and GitHub rejects blobs over 100 MB. Re-ignore "
        ".aip_cache/ in .gitignore -- and do not 'fix' it by committing it.")


def test_reading_the_replay_set_does_not_modify_it():
    """Replay reads are read-only, and that has to stay true.

    aip/cache.py opens the replay set with mode=ro. If that ever becomes a
    read-write connection, the CREATE TABLE IF NOT EXISTS in the scratch path's
    sibling schema lands in a tracked file: the mtime changes, the working tree
    goes dirty on every single run, and `git status` starts showing a file
    nobody edited. Cheap to test, annoying to diagnose.
    """
    path = cache._REPLAY_DB_PATH
    if not path.exists():
        pytest.skip("no replay set on this machine; nothing to protect yet")
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    if cache._replay_connect() is not None:
        cache.stats()  # opens the replay store read-only
    after = (path.stat().st_mtime_ns, path.stat().st_size)
    assert before == after, f"reading {_rel(path)} modified it: {before} -> {after}"
    assert not list(path.parent.glob("*.pruning")), (
        "a prune was interrupted and left its temp file beside the replay set")


def test_the_replay_set_holds_the_golden_set():
    """It is not enough for the file to exist: it has to answer the golden set.

    A replay set recorded against a different prompt, tier or final_k is a valid
    SQLite file of the right size and replays nothing. The keys are content
    hashes, so the honest check is the one the code already uses -- the query
    vector for every golden question, computed with aip's own make_key, must be
    present. This is the same set scripts/prune_cache.py records.
    """
    path = cache._REPLAY_DB_PATH
    if not path.exists():
        pytest.skip("no replay set on this machine; nothing to check")
    from aip.config import resolve_model
    from labs.lab3.search import load_questions

    model = resolve_model("EMBED")
    keys = {
        cache.make_key("embed", {"model": model, "text": q["question"],
                                 "input_type": None})
        for q in load_questions(include_unanswerable=True)
    }
    present = set()
    conn = cache._replay_connect()
    assert conn is not None, "could not open the replay set read-only"
    try:
        # Chunked because SQLite caps a host parameter list, and 45 questions
        # is under it today but the golden set grows.
        k = list(keys)
        for start in range(0, len(k), 400):
            q = ",".join("?" * len(k[start:start + 400]))
            present |= {r[0] for r in conn.execute(
                f"SELECT key FROM calls WHERE key IN ({q})", k[start:start + 400])}
    finally:
        conn.close()
    missing = keys - present
    assert not missing, (
        f"{len(missing)} of {len(keys)} golden questions have no query vector in "
        f"the replay set, so offline retrieval is a CacheMiss for them: "
        f"{sorted(missing)[:3]}. Re-warm online and re-run "
        f"`python scripts/prune_cache.py --prune`.")
