"""Content-addressed disk cache for model calls.

Why this exists
---------------
An LLM call is an expensive, slow, non-deterministic function. Almost every
painful thing about developing against one goes away if you make repeated
identical calls free and instant:

* You can re-run a 200-item extraction loop while debugging your parser
  without paying 200 times.
* Your eval harness becomes reproducible: same inputs, same outputs, so a
  diff in your score is a diff in *your code*, not model sampling noise.
* Grading and demos work offline (AIP_OFFLINE=1).

The cache key is a hash of everything that could change the output: model,
messages, temperature, tools, response format, seed. Change any of them and
you get a miss, as you should.

TWO FILES, TWO JOBS
-------------------
There is exactly one schema and two databases that hold it:

    .aip_cache/calls.sqlite3     SCRATCH. Written by every call. 144 MB after an
                                 online warm, because it keeps every corpus
                                 embedding. Gitignored, deliberately and
                                 permanently.
    data/replay/calls.sqlite3    THE REPLAY SET. Committed. ~1.3 MB, holding only
                                 the keys offline replay reads (see
                                 scripts/prune_cache.py). Never written here.

`get()` reads the scratch first and, when AIP_OFFLINE=1, falls back to the
replay set -- so a fresh clone, a grader and CI all serve the golden set with no
network and no key. Online the fallback is deliberately NOT consulted: with a key
in hand a miss should be a real call, so that cost and latency stay measured.

They are separate paths on purpose. They used to be one file, and the single
path was both the gitignored scratch and the committed replay set -- so the
moment anyone ran scripts/warm_cache.py online, the tracked path became a
144 MB blob and the push died. Splitting them makes that impossible rather than
merely discouraged: nothing under .aip_cache/ is ever tracked, so there is no
gitignore rule whose ordering can break, and no way to stage the big file.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
from typing import Any

from aip.config import REPO_ROOT, settings

_LOCK = threading.Lock()
_DB_PATH = settings.cache_dir / "calls.sqlite3"
_REPLAY_DB_PATH = REPO_ROOT / "data" / "replay" / "calls.sqlite3"


class CacheMiss(RuntimeError):
    """Raised in offline mode when a request has no cached response."""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS calls (
               key TEXT PRIMARY KEY,
               kind TEXT NOT NULL,
               request TEXT NOT NULL,
               response TEXT NOT NULL,
               created_at REAL NOT NULL
           )"""
    )
    return conn


def _replay_connect() -> sqlite3.Connection | None:
    """Open the COMMITTED replay set, read-only. None when it is absent.

    Read-only is load-bearing, not tidiness. This runs on every cache miss, and
    a read-write connection would execute CREATE TABLE IF NOT EXISTS against a
    tracked file -- touching its mtime, leaving a permanently dirty working tree,
    and turning every replay run into an unexplained `git status` entry. The
    URI form also has to be percent-encoded: the repo path can contain spaces
    and a `!`, both of which SQLite would otherwise misparse.
    """
    if not _REPLAY_DB_PATH.exists():
        return None
    return sqlite3.connect(f"{_REPLAY_DB_PATH.as_uri()}?mode=ro", uri=True,
                           timeout=30)


def make_key(kind: str, payload: dict[str, Any]) -> str:
    """Stable hash of a request payload. Sorted keys so dict order is irrelevant."""
    blob = json.dumps({"kind": kind, **payload}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(key: str) -> dict[str, Any] | None:
    if not settings.cache_enabled:
        return None
    with _LOCK, _connect() as conn:
        row = conn.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
    if row is not None or not settings.offline:
        # Either the scratch has it, or we are online -- in which case the caller
        # makes a real call, exactly as before the replay set existed. Serving a
        # recorded answer to someone who has a key and did not ask for replay
        # would hide both the cost and the latency, and those are measurements
        # this module exists to take.
        return json.loads(row[0]) if row is not None else None
    # Offline, and the scratch does not have it: CI, a grader, a demo with no
    # key. The answer is in the committed replay set. One extra indexed SELECT
    # on a 175-row table, only on a miss -- which, offline, is the only kind
    # there is.
    with contextlib.closing(_replay_connect()) as rconn:
        if rconn is None:
            return None
        row = rconn.execute(
            "SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def put(key: str, kind: str, request: dict[str, Any], response: dict[str, Any]) -> None:
    if not settings.cache_enabled:
        return
    import time

    with _LOCK, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO calls (key, kind, request, response, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                kind,
                json.dumps(request, default=str)[:200_000],
                json.dumps(response, default=str),
                time.time(),
            ),
        )


def stats() -> dict[str, int]:
    with _LOCK, _connect() as conn:
        rows = conn.execute("SELECT kind, COUNT(*) FROM calls GROUP BY kind").fetchall()
    out = dict(rows)
    # Reported separately rather than summed in. Under AIP_OFFLINE=1 the scratch
    # counts are near zero and the replay set is what actually answered every
    # request -- merging them into one number would make /health read as though
    # the service had no cache at all.
    with contextlib.closing(_replay_connect()) as rconn:
        if rconn is not None:
            out["replay"] = rconn.execute(
                "SELECT COUNT(*) FROM calls").fetchone()[0]
    return out


def clear(kind: str | None = None) -> int:
    """Drop cached entries. Returns the number removed. Use with care.

    The scratch cache only. The committed replay set is a build artefact and a
    deliverable -- deleting rows from it would dirty the working tree and cost
    every fresh clone its offline replay -- so there is deliberately no way to
    clear it from here. Re-record it with scripts/prune_cache.py instead.
    """
    with _LOCK, _connect() as conn:
        if kind:
            cur = conn.execute("DELETE FROM calls WHERE kind = ?", (kind,))
        else:
            cur = conn.execute("DELETE FROM calls")
        return cur.rowcount
