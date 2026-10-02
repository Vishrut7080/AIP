#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=[runs[0].stem])
rows = [json.loads(l) for p in runs if p.stem in chosen
        for l in p.open(encoding="utf-8") if l.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

c = st.columns(5)
c[0].metric("spans", len(df))
c[1].metric("total cost", f"${df.get('cost_usd', pd.Series([0])).fillna(0).sum():.4f}")
llm = df[df["name"] == "llm.call"]
c[2].metric("model calls", len(llm))
if len(llm):
    c[3].metric("cache hit rate", f"{llm.get('cached', pd.Series([False])).fillna(False).mean():.0%}")
c[4].metric("errors", int((df.get("status") == "error").sum()))

st.subheader("Latency by stage")
# C3: p50/p95 per span name. This is the table that answers "which stage should
# I optimise?" -- see Lab 7 Part B4.
#
# The stage that dominates is GENERATION, not retrieval: on the shipping config
# rag.generate is ~1500 ms of a ~2100 ms request while retrieve.dense is ~60 ms.
# Sum the column before optimising anything, because generate/solve/
# answer_question/http.ask all CONTAIN the generation call and will each look
# like the biggest row. The innermost span is the real cost.
stage = (df.groupby("name")["duration_ms"]
         .agg(n="count", p50="median",
              p95=lambda s: s.quantile(0.95), total="sum")
         .sort_values("total", ascending=False))
st.dataframe(stage, use_container_width=True)

inner = [n for n in ("rag.generate", "rag.generate.stream", "rag.retrieve",
                      "retrieve.dense", "embed.batch", "llm.call")
         if n in stage.index]
if inner:
    st.caption(
        "**Reading this table:** `rag.solve`, `rag.answer_question` and "
        "`http.ask` all *contain* `rag.generate`, so their totals double-count "
        "it. The innermost spans are the real cost: "
        + ", ".join(f"`{n}` {stage.loc[n, 'p50']:.0f}ms p50"
                    for n in inner) + ".")

st.subheader("Cost over time")
if "cost_usd" in df:
    cum = df.sort_values("ts").assign(cum=lambda d: d["cost_usd"].fillna(0).cumsum())
    st.line_chart(cum.set_index("ts")["cum"])

st.subheader("Errors")
errs = df[df.get("status") == "error"]
st.dataframe(errs[["ts", "name", "error"]] if len(errs) else pd.DataFrame(),
             use_container_width=True)

# ---------------------------------------------------------------------------
# C4: the alert.
# ---------------------------------------------------------------------------
# Refusal rate doubling is the alert, not p95.
#
# p95 breaches are usually a provider problem, which self-resolves or does not
# and either way is not actionable in under five minutes. A sudden JUMP in
# refusal rate means the SYSTEM changed underneath a working pipeline: the index
# silently failed to build, a retrieval regression emptied the context, or a
# prompt edit made the model cautious. The failure is invisible in correctness
# (it looks like the model being careful) and in latency (it gets FASTER). Only
# the refusal rate moves.
#
# So when it fires I would: freeze the deploy, and diff the index first --
# `GET /health` reports n_chunks and the matrix shape, so an index that built at
# 40 chunks instead of 235 is visible in seconds without touching the model. If
# the index is intact, the next suspects are, in order, a prompt/strictness
# change and a retrieval configuration change (final_k, retrieve_k, chunking).
# I would NOT reach for the model or the provider.
ALERT_REFUSAL_RATE = 0.35
ALERT_BASELINE_REFUSALS = 0.10

if "name" in df:
    asks = df[df["name"] == "http.ask"]
    refusals = df[df["name"] == "service.refused"]
    if len(asks):
        recent = asks.tail(max(10, len(asks) // 4))
        recent_r = refusals.tail(max(10, len(refusals) // 4)) if len(refusals) else refusals
        # Count refusals whose ts falls in the recent window, so the ratio is
        # like-for-like rather than a cumulative-total artefact.
        cutoff = recent["ts"].min().timestamp()
        recent_r = refusals[refusals["ts"] >= cutoff]
        observed = len(recent_r) / len(recent) if len(recent) else 0.0
        if observed > ALERT_REFUSAL_RATE:
            st.error(f"**ALERT: refusal rate {observed:.0%}** "
                     f"(baseline {ALERT_BASELINE_REFUSALS:.0%}, "
                     f"trips above {ALERT_REFUSAL_RATE:.0%})")
            st.markdown(
                "_Refusal rate spiked. **Freeze the deploy.** Check "
                "`GET /health` first: if `index.n_chunks` moved, the index broke "
                "and the model is innocent. If the index is intact, check for a "
                "prompt / strictness change and then a retrieval-config change "
                "(`final_k`, `retrieve_k`, chunking). Do **not** start with the "
                "model or the provider -- a refusal spike is a symptom of "
                "retrieval or configuration, and the model refusing more is "
                "what a working guard looks like._")
        else:
            st.success(f"refusal rate {observed:.0%} "
                       f"(trips above {ALERT_REFUSAL_RATE:.0%})")

    c2 = st.columns(3)
    cache_spans = df[df["name"] == "llm.call"]
    if len(cache_spans) and "cached" in cache_spans:
        c2[0].metric("cache hit rate",
                     f"{cache_spans['cached'].fillna(False).mean():.0%}")
    if "cost_usd" in df:
        c2[1].metric("cost/query",
                     f"${df['cost_usd'].fillna(0).sum() / max(1, len(asks)):.5f}")
    c2[2].metric("span total", f"{len(df)}")
