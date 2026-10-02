"""Tests for the Lab 7 service contract.

WHAT THESE ARE FOR
------------------
Every graded item in Part A3 is a status code, and none of them had a test. The
suite covered `aip/` and the gate's refusal formula; nothing pinned the HTTP
contract, so a refactor could turn a provider outage into a 500 with a stack
trace -- or, far worse, turn a malformed request into a 500 -- and the suite
would stay green. The demo asks for a refusal and an outage on the projector,
and a 500 is what you do not want there.

The three error paths are the reason this file exists:

    malformed request  -> 422
    model outage       -> 503 + Retry-After   (NOT 500: the bug is not ours,
                                                and a 500 invites an immediate
                                                retry that makes it worse)
    budget exhausted   -> 429 + Retry-After

WHY THEY MONKEYPATCH INSTEAD OF SETTING AN ENV VAR
--------------------------------------------------
`aip.config.settings` is built once at import time, so AIP_OFFLINE=1 set inside
a test arrives too late to matter -- and pytest imports test_aip.py first, so
the module is already loaded. The fixture therefore flips `offline` on the
existing settings object. It matters that this is done honestly rather than by
faking the answers: with the real flag set, /ask replays the committed cache
through the real pipeline, so the happy-path and streaming tests exercise the
same code path CI does.

The error tests replace `_run_rag` so each failure is reachable without needing
a provider to actually be down, which is otherwise untestable.

NEEDS NO API KEY AND NO NETWORK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aip import cache, config  # noqa: E402
from aip.cost import BudgetExceeded  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402

# Loaded from the golden set rather than pasted in, because offline replay is
# keyed on a content hash of the exact question string. An earlier version of
# this file hardcoded a paraphrase lifted from README.md --
# "How long do I have to submit a reimbursement claim after discharge?" -- where
# Q01 actually reads "How many days do I have to submit...". The paraphrase is
# not in the cache, so every test failed with a 503 that was the service
# behaving perfectly correctly. Referencing the golden set makes that class of
# drift impossible, and it is the same reason prune_cache.py computes query keys
# through aip's own make_key.
GOLDEN = {q["id"]: q["question"] for q in
          load_questions(include_unanswerable=True)}["Q01"]

MALFORMED = [
    pytest.param({}, id="no-question"),
    pytest.param({"question": 123}, id="wrong-type"),
    pytest.param({"question": "ab"}, id="too-short"),          # min_length=3
    pytest.param({"question": "x" * 1001}, id="too-long"),     # max_length=1000
    pytest.param({"question": GOLDEN, "top_k": 0}, id="top-k-below-min"),
    pytest.param({"question": GOLDEN, "top_k": 99}, id="top-k-above-max"),
    pytest.param({"question": GOLDEN, "mode": "bogus"}, id="bad-mode"),
]


@pytest.fixture
def service():
    """The service module, with per-process state reset and offline replay on."""
    from labs.lab7 import service as svc

    # The exact/semantic caches and the counters are module globals that live
    # for the process. Without this reset, whichever test happens to run first
    # decides whether later ones see a cache hit -- and a cache hit skips the
    # error handling entirely, so the tests would pass or fail by ordering.
    svc._EXACT_CACHE.clear()
    svc._SEMANTIC_CACHE.clear()
    svc._ERRORS_BY_TYPE.clear()
    for k in svc.STATS:
        svc.STATS[k] = 0
    config.settings.offline = True
    return svc


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    with TestClient(service.app) as c:
        yield c


def _events(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    out = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            out.append((name, data))
    return out


# ---------------------------------------------------------------------------
# A3: the error contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("body", MALFORMED)
def test_malformed_request_is_422(client, body):
    """A3. A malformed request is the client's problem: 422, not 500.

    Every one of these is rejected by Pydantic before the handler runs, so the
    test also pins the schema constraints themselves -- loosening min_length or
    top_k's bounds would fail here rather than at 3am.
    """
    r = client.post("/ask", json=body)
    assert r.status_code == 422, r.text


def test_missing_field_is_422_not_500(client):
    """The specific shape a demo is likely to hit: no JSON body field at all."""
    r = client.post("/ask", json={"top_k": 4})
    assert r.status_code == 422
    assert "question" in r.text


def test_offline_cache_miss_is_503_with_retry_after(client, service, monkeypatch):
    """A3. Offline and uncached: 503, because the request is fine and the
    service simply cannot serve it. A 400 would blame the client for our
    configuration; a 500 would invite a retry that cannot possibly help."""
    def boom(question, top_k):
        raise cache.CacheMiss("not in the cache")

    monkeypatch.setattr(service, "_run_rag", boom)
    r = client.post("/ask", json={"question": GOLDEN})
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "60"


def test_upstream_outage_is_503_with_retry_after(client, service, monkeypatch):
    """A3. The graded one. A provider outage must not surface as a 500.

    A 500 tells the client the bug is ours and prompts an immediate retry,
    which is exactly wrong for a transient upstream failure and is how an
    outage turns into a self-inflicted denial of service.
    """
    def boom(question, top_k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(service, "_run_rag", boom)
    r = client.post("/ask", json={"question": GOLDEN})
    assert r.status_code == 503, r.text
    assert r.headers.get("Retry-After") == "5"
    # The stack trace must not leak to the client.
    assert "Traceback" not in r.text
    assert "connection reset" not in r.text


def test_budget_exhausted_is_429_with_retry_after(client, service, monkeypatch):
    """A3. Out of money is 429, and retrying immediately makes it worse."""
    def boom(question, top_k):
        raise BudgetExceeded("budget of $2.00 exhausted")

    monkeypatch.setattr(service, "_run_rag", boom)
    r = client.post("/ask", json={"question": GOLDEN})
    assert r.status_code == 429, r.text
    assert r.headers.get("Retry-After") == "30"


def test_errors_are_counted_by_type(client, service, monkeypatch):
    """C2. /metrics has to break errors down by type, not just total them.

    "3 errors" is not actionable. cache_miss and budget mean fix the config;
    internal means look at the provider.
    """
    def boom(question, top_k):
        raise cache.CacheMiss("nope")

    monkeypatch.setattr(service, "_run_rag", boom)
    client.post("/ask", json={"question": GOLDEN})
    m = client.get("/metrics").json()
    assert m["errors"] >= 1
    assert m["errors_by_type"].get("cache_miss") == 1


# ---------------------------------------------------------------------------
# A1: the response contract
# ---------------------------------------------------------------------------
def test_ask_returns_a_grounded_answer_with_citations(client):
    """A1. Replayed from the committed cache -- the same path CI exercises.

    cost_usd and trace_id are in the response on purpose: one is how whoever
    pays sees what a query costs, the other is how anyone debugging finds the
    trace. Both are trivial to drop and impossible to add later.
    """
    r = client.post("/ask", json={"question": GOLDEN})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].strip()
    assert not body["refused"]
    assert body["citations"], "an uncited answer is the thing we are paid to prevent"
    for key in ("latency_ms", "cost_usd", "cached", "trace_id", "cache_layer"):
        assert key in body, key
    assert body["trace_id"]
    for c in body["citations"]:
        assert c["doc_id"] and c["excerpt"]


def test_second_identical_question_hits_the_exact_cache(client):
    """B1. The exact layer is keyed on the normalised question."""
    first = client.post("/ask", json={"question": GOLDEN}).json()
    second = client.post("/ask", json={"question": GOLDEN}).json()
    assert first["cache_layer"] == "miss"
    assert second["cached"] is True
    assert second["cache_layer"] == "exact"


def test_cache_key_normalisation_collapses_trivial_variations(service):
    """B1. Case, punctuation and whitespace must not create a second entry.

    Pinned directly on the two pure functions rather than through HTTP, so a
    change to the normaliser cannot quietly cost 20% of the cache hit rate.
    """
    variants = [
        GOLDEN,
        GOLDEN.upper(),
        GOLDEN.lower(),
        GOLDEN + "   ",
        "  " + GOLDEN,
        GOLDEN.replace("claim after", "claim  after"),
    ]
    keys = {service._key(v) for v in variants}
    assert len(keys) == 1, f"normalisation is not collapsing variants: {keys}"
    # ...and a genuinely different question must not collide.
    assert service._key("What is the claims helpline number?") not in keys


# ---------------------------------------------------------------------------
# C: observability
# ---------------------------------------------------------------------------
def test_health_reports_index_model_and_cache(client):
    """C1. The first thing anyone checks."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["n_chunks"] > 0
    assert body["index"]["n_docs"] > 0
    assert body["model"]["generation_tier"]
    assert body["model"]["generate"]
    assert body["offline"] is True


def test_metrics_exposes_the_slo_numbers(client):
    """C2. Every field the rubric's SLOs are computed from.

    p95 is the number the demo quotes, so it has to be a number rather than
    absent -- and it is measured even under replay, which is the case that
    silently produced no latency at all before the gate started timing wall
    clock.
    """
    body = client.get("/metrics").json()
    for key in ("cost_usd", "cost_per_query", "requests", "errors",
                "error_rate", "refusals", "refusal_rate"):
        assert key in body, key
    for key in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms"):
        assert key in body, key
    assert "hit_rate" in body["cache"]
    assert "errors_by_type" in body


# ---------------------------------------------------------------------------
# B2/B3: streaming
# ---------------------------------------------------------------------------
def test_stream_emits_meta_deltas_and_a_final_done(client, service, monkeypatch):
    """B2. The stream is real, and B3's choice is visible in the event order.

    B3 is the problem that you cannot validate citations until the answer is
    complete but the stream has already sent it. The chosen resolution is to
    stream the prose and hold the citations to the end -- so `done` must be the
    LAST event and must be the one carrying citations. If a refactor moves
    citations into the first delta, the guarantee is gone and this fails.

    The generator is stubbed because the streaming path cannot be replayed from
    the committed cache -- see the next test for why. Everything downstream of
    it, which is where the B3 guarantee lives, is the real code.
    """
    def fake_tokens(question, top_k):
        # Real retrieval, canned prose: the citation check under test is
        # _citations() running over real hits, not the prose itself.
        hits = service.pipeline().search(question, k=service.RETRIEVE_K)
        yield from ("You have ", "30 days ", "from the date of discharge [1].")
        service._STREAM_STATE["result"] = (
            "You have 30 days from the date of discharge [1].", False, hits, 0.0)

    monkeypatch.setattr(service, "_run_rag_tokens", fake_tokens)
    r = client.post("/ask/stream", json={"question": GOLDEN})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _events(r.text)
    names = [n for n, _ in events]
    assert names[0] == "meta"
    assert names.count("delta") == 3
    assert names[-1] == "done"

    # The streamed prose must reassemble into exactly what `done` reports.
    streamed = "".join(d["text"] for n, d in events if n == "delta")
    done = events[-1][1]
    assert done["answer"] == streamed
    assert done["citations"], "citations belong in `done`, not in a delta"
    assert done["citations_valid"] is True
    assert done["ttft_ms"] > 0
    assert done["total_ms"] >= done["ttft_ms"]


def test_offline_stream_needs_its_own_warm_up(client, service):
    """Pins a real limitation rather than hiding it behind a stub.

    `_run_rag_tokens` builds its own prompt (service.py, delimit_untrusted
    context and STRICTNESS system message), which is NOT the prompt
    labs/lab4/rag.py sends -- and the cache key is a content hash of the whole
    request. So a warmed /ask is not a warmed /ask/stream, and
    scripts/warm_cache.py only ever warms the non-streaming path via
    answer_question(). Offline, streaming therefore ends in an `error` event.

    That is worth a test rather than a workaround: the alternative is a silent
    503 on the projector during the one demo item that is specifically about
    streaming. Note also that TTFT cannot be measured under replay anyway --
    a replayed stream arrives as a single delta, so TTFT equals total and the
    graded TTFT number has to come from an online run.
    """
    events = _events(client.post("/ask/stream",
                                 json={"question": GOLDEN}).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 503


def test_stream_reports_upstream_failure_as_an_error_event(client, service,
                                                           monkeypatch):
    """A stream cannot return a 503 -- the headers are already sent by the time
    generation fails. So the failure has to arrive as an `error` event carrying
    the status the non-streaming path would have used."""
    def boom(question, top_k):
        raise RuntimeError("upstream gone")
        yield from ()  # unreachable: makes this a generator, like the real one

    monkeypatch.setattr(service, "_run_rag_tokens", boom)
    events = _events(client.post("/ask/stream",
                                 json={"question": GOLDEN}).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 503