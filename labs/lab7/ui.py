#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json

import requests
import streamlit as st

API = st.sidebar.text_input("Service URL", "http://localhost:8000")

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")

# B2/B3: stream by default, so the demo shows TTFT rather than a spinner.
# Citations arrive in the final `done` event and are rendered below -- never
# inline -- which is exactly the B3 decision the service makes server-side.
stream = st.checkbox("Stream the answer", value=True,
                     help="Server-sent events. Citations are validated by the "
                          "server and sent in the final event, so they appear "
                          "after the prose rather than while it streams.")

if st.button("Ask", type="primary") and q:
    data = None
    if stream:
        with st.spinner("connecting"):
            try:
                with requests.post(f"{API}/ask/stream", json={"question": q},
                                   stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    placeholder = st.empty()
                    buf, done, ttft = "", None, None
                    event = None
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8", "replace")
                        if line.startswith("event:"):
                            event = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            payload = json.loads(line.split(":", 1)[1].strip())
                            if event == "delta":
                                buf += payload["text"]
                                placeholder.markdown(buf + "▌")
                            elif event == "done":
                                done = payload
                    data = done
            except requests.HTTPError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text[:300]}")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"service unreachable: {exc}")
                st.stop()
    else:
        with st.spinner("thinking"):
            try:
                r = requests.post(f"{API}/ask", json={"question": q}, timeout=60)
                r.raise_for_status()
                data = r.json()
            except requests.HTTPError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text[:300]}")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"service unreachable: {exc}")
                st.stop()

    if data is None:
        st.error("no answer received")
        st.stop()

    if data.get("refused"):
        st.warning(data["answer"])
    else:
        st.markdown(data["answer"])

    if not data.get("citations_valid", True):
        st.error("This answer was not fully grounded and was withheld.")

    # A4: citations as expandable source text. Grounding the user cannot check is
    # decoration, so each expander shows the actual retrieved chunk the citation
    # points at, not just its title.
    cits = data.get("citations", [])
    if cits:
        st.caption(f"{len(cits)} source(s) cited")
        for c in cits:
            with st.expander(f"[{c['index']}] {c['doc_id']}"):
                st.text(c.get("excerpt", ""))

    cols = st.columns(5)
    cols[0].metric("latency", f"{data.get('latency_ms') or data.get('total_ms', 0):.0f} ms")
    cols[1].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[2].metric("cached", "yes" if data.get("cached") else "no")
    cols[3].metric("sources", len(cits))
    if data.get("ttft_ms"):
        cols[4].metric("TTFT", f"{data['ttft_ms']:.0f} ms")
    st.caption(f"trace: `{data.get('trace_id', '')}`  ·  "
               f"layer: `{data.get('cache_layer', '')}`")

# TODO stretch: a thumbs-down button that appends the case to a review queue.
# That queue is how real golden sets get built.
