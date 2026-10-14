#!/usr/bin/env python3
"""Lab 7 — Streamlit front end (restyled).

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.

Design notes
- Palette: cool "polar night" ink on a pale mist background, aurora teal for
  actions, violet reserved for citations. Violet appears in two places only:
  the [n] markers in the answer and the left edge of each source card, so the
  eye learns that violet means "you can check this".
- Type: Bricolage Grotesque for interface and headings, Source Serif 4 for the
  answer and the quoted source text (documents read as documents).
"""
from __future__ import annotations

import html
import json
import re

import requests
import streamlit as st

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="Aurora Policy Assistant",
                   page_icon="🌌", layout="centered")

# Exact questions from data/eval/rag_golden.jsonl. Keeping the wording verbatim
# means these examples also work with the committed offline replay cache.
EXAMPLES = [
    ("Reimbursement deadline",
     "How many days do I have to submit a reimbursement claim after discharge?"),
    ("Cataract limit on Gold",
     "What is the cataract sub-limit on Gold?"),
    ("Pre-existing waiting period",
     "What is the waiting period for pre-existing diseases?"),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,500;12..96,600;12..96,700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

:root {
  --ink: #14262e;
  --muted: #5b6f77;
  --mist: #edf3f4;
  --mist-deep: #e1ebed;
  --paper: #ffffff;
  --line: #d3e0e3;
  --aurora: #0e8f7e;
  --aurora-deep: #0a6b5e;
  --violet: #5b4bc4;
  --violet-tint: #efedfb;
  --amber: #8a5300;
  --amber-tint: #fdf3df;
  --alert: #a8321f;
  --radius: 12px;
}

html, body, .stApp {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif;
  color: var(--ink);
  color-scheme: light;
}
.stApp { background: var(--mist); }
[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 3.2rem; padding-bottom: 4rem; }

/* ---- masthead ---- */
.masthead { margin-bottom: 1.6rem; }
.masthead .ribbon {
  width: 56px; height: 5px; border-radius: 3px; margin-bottom: 1rem;
  background: linear-gradient(90deg, var(--aurora), var(--violet));
}
.masthead h1 {
  font-size: 2.5rem; line-height: 1.05; font-weight: 700;
  letter-spacing: -0.02em; margin: 0 0 0.6rem 0; padding: 0; color: var(--ink);
}
.masthead p {
  font-size: 1.02rem; line-height: 1.5; color: var(--muted);
  max-width: 60ch; margin: 0;
}

/* ---- ask form ---- */
[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
.stTextInput input {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif;
  font-size: 1.05rem; padding: 0.8rem 1rem; color: var(--ink);
  background: var(--paper); border: 1px solid var(--line);
  border-radius: var(--radius);
}
.stTextInput input:focus {
  border-color: var(--aurora); box-shadow: 0 0 0 3px rgba(14, 143, 126, 0.18);
}
.stTextInput [data-baseweb="input"],
.stTextInput [data-baseweb="base-input"] {
  background: transparent; border: none;
}

button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--aurora); color: #fff; border: none; width: 100%;
  border-radius: var(--radius); font-weight: 600; font-size: 1rem;
  padding: 0.7rem 1rem; min-height: 3rem;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: var(--aurora-deep); color: #fff;
}

/* example questions */
.try-label { font-size: 0.9rem; color: var(--muted); margin: 0.9rem 0 0.4rem 0; }
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background: transparent; color: var(--ink); border: 1px solid var(--line);
  border-radius: 999px; font-size: 0.88rem; padding: 0.3rem 0.9rem;
  text-align: left; min-height: 0;
}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--aurora); color: var(--aurora-deep); background: var(--paper);
}

/* ---- answer card ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--paper); border: 1px solid var(--line);
  border-radius: 16px; padding: 0.6rem 0.9rem;
}
/* The answer is rendered as markdown inside the bordered container, not inside
   a wrapping <div> -- Streamlit treats a raw HTML block as opaque and would
   render **bold**, paragraphs and lists literally. So style the markdown
   Streamlit actually emits instead. */
[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown,
[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown p,
[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown li {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.14rem; line-height: 1.65; color: var(--ink);
}
.pending { color: var(--muted); font-style: italic; }
.badge {
  display: inline-block; font-family: 'Bricolage Grotesque', system-ui, sans-serif;
  font-size: 0.82rem; font-weight: 600;
  padding: 0.18rem 0.6rem; border-radius: 999px; margin-bottom: 0.6rem;
}
.badge-warn { background: var(--amber-tint); color: var(--amber); }
.badge-bad { background: #fbe9e6; color: var(--alert); }

/* inline [n] markers: same violet as the source cards they point to */
.cite {
  display: inline-block; vertical-align: 0.15em; margin: 0 0.12em;
  min-width: 1.35em; padding: 0 0.3em; text-align: center;
  font-family: 'Bricolage Grotesque', system-ui, sans-serif;
  font-size: 0.72rem; font-weight: 700; line-height: 1.45em;
  color: var(--violet); background: var(--violet-tint); border-radius: 6px;
}

/* ---- sources ---- */
.sources-title {
  font-size: 1.05rem; font-weight: 600; margin: 1.8rem 0 0.6rem 0;
}
.sources-title span { color: var(--muted); font-weight: 400; margin-left: 0.3rem; }

[data-testid="stExpander"] {
  background: var(--paper); border: 1px solid var(--line);
  border-left: 4px solid var(--violet); border-radius: var(--radius);
  margin-bottom: 0.5rem; overflow: hidden;
}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {
  background: transparent;
}
[data-testid="stExpander"] summary { font-weight: 600; padding: 0.7rem 1rem; }
[data-testid="stExpander"] summary:hover { color: var(--violet); }
.excerpt {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1rem; line-height: 1.6; color: var(--ink);
  white-space: pre-wrap; background: var(--mist);
  border-radius: 8px; padding: 0.9rem 1.1rem;
}

/* ---- run stats ---- */
.stats {
  display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1.6rem;
}
.stat {
  flex: 1 1 110px; background: var(--paper); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 0.6rem 0.85rem;
}
.stat span { display: block; font-size: 0.8rem; color: var(--muted); }
.stat b { font-size: 1.15rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.trace { font-size: 0.82rem; color: var(--muted); margin-top: 0.8rem; }
.trace code {
  background: var(--mist-deep); color: var(--ink);
  padding: 0.1rem 0.4rem; border-radius: 5px; font-size: 0.8rem;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: var(--mist-deep); }
[data-testid="stSidebar"] .stTextInput input { font-size: 0.92rem; padding: 0.5rem 0.7rem; }

[data-testid="stAlert"] { border-radius: var(--radius); }

@media (max-width: 640px) {
  .masthead h1 { font-size: 2rem; }
  .block-container { padding-top: 2rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("**Settings**")
    API = st.text_input("Service URL", "http://localhost:8000")
    # B2/B3: stream by default, so the demo shows TTFT rather than a spinner.
    # Citations arrive in the final `done` event and are rendered below -- never
    # inline -- which is exactly the B3 decision the service makes server-side.
    stream = st.checkbox("Stream the answer", value=True,
                         help="Server-sent events. Citations are validated by "
                              "the server and sent in the final event, so they "
                              "appear after the prose rather than while it "
                              "streams.")
    st.caption("start the dashboard with `--server.port 8502`")
    st.link_button("Operations dashboard →", "http://localhost:8502")

# ---------------------------------------------------------------- masthead
st.markdown(
    """
    <div class="masthead">
      <div class="ribbon"></div>
      <h1>Aurora Policy Assistant</h1>
      <p>Answers come only from Aurora's policy documents, and every claim is
      cited. When the documents don't cover a question, the assistant says so
      instead of guessing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _use_example(text: str) -> None:
    """Put an example in the query box without submitting it."""
    st.session_state["q"] = text


with st.form("ask", border=False):
    c_in, c_btn = st.columns([5, 1])
    q = c_in.text_input("Ask a question", key="q",
                        label_visibility="collapsed",
                        placeholder="How long do I have to file a reimbursement claim?")
    submitted = c_btn.form_submit_button("Ask", type="primary")

st.markdown('<div class="try-label">Try asking</div>', unsafe_allow_html=True)
_top, _bottom = st.columns(3), st.columns(2)
for col, (label, question) in zip(list(_top) + list(_bottom), EXAMPLES):
    col.button(label, key=f"ex-{label}", on_click=_use_example, args=(question,),
               width="stretch")

CITE_RE = re.compile(r"\[(\d+)\]")


def decorate(text: str) -> str:
    """Turn [n] markers into violet badges that match the source cards.

    Escape first, substitute second: the answer is model output and this goes
    through unsafe_allow_html, so anything left unescaped would render. The
    markers survive escaping because [ ] are not HTML metacharacters.
    """
    return CITE_RE.sub(r'<span class="cite">\1</span>', html.escape(text))


def stat(label: str, value: object) -> str:
    return (f'<div class="stat"><span>{html.escape(label)}</span>'
            f'<b>{html.escape(str(value))}</b></div>')


run = submitted and (q or "").strip()

if run:
    data = None
    with st.container(border=True):
        slot = st.empty()
        slot.markdown('<span class="pending">Searching the policy documents…</span>',
                      unsafe_allow_html=True)

        def fail(msg: str) -> None:
            slot.error(msg)
            st.stop()

        if stream:
            try:
                with requests.post(f"{API}/ask/stream", json={"question": q},
                                   stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    buf, done, event = "", None, None
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
                                slot.markdown(decorate(buf) + " ▌",
                                              unsafe_allow_html=True)
                            elif event == "done":
                                done = payload
                    data = done
            except requests.HTTPError as exc:
                fail(f"{exc.response.status_code}: {exc.response.text[:300]}")
            except requests.RequestException as exc:
                fail(f"Service unreachable: {exc}")
        else:
            try:
                r = requests.post(f"{API}/ask", json={"question": q}, timeout=60)
                r.raise_for_status()
                data = r.json()
            except requests.HTTPError as exc:
                fail(f"{exc.response.status_code}: {exc.response.text[:300]}")
            except requests.RequestException as exc:
                fail(f"Service unreachable: {exc}")

        if data is None:
            fail("No answer received.")

        # Final render replaces the streaming caret and adds the status badge.
        badge = ""
        if data.get("refused"):
            badge = '<div class="badge badge-warn">Not covered by the policy documents</div>\n\n'
        elif not data.get("citations_valid", True):
            badge = '<div class="badge badge-bad">Withheld: not fully grounded</div>\n\n'
        slot.markdown(badge + decorate(data["answer"]), unsafe_allow_html=True)

    if not data.get("citations_valid", True):
        st.error("This answer was not fully grounded and was withheld.")

    # A4: citations as expandable source text. Grounding the user cannot check is
    # decoration, so each expander shows the actual retrieved chunk the citation
    # points at, not just its title.
    cits = data.get("citations", [])
    if cits:
        st.markdown(f'<div class="sources-title">Sources<span>{len(cits)} cited</span></div>',
                    unsafe_allow_html=True)
        for c in cits:
            with st.expander(f"[{c['index']}]  {c['doc_id']}"):
                st.markdown(f'<div class="excerpt">{html.escape(c.get("excerpt", ""))}</div>',
                            unsafe_allow_html=True)

    latency = data.get("latency_ms") or data.get("total_ms") or 0
    cost = data.get("cost_usd") or 0
    stats = [
        stat("Latency", f"{latency:.0f} ms"),
        stat("Cost", f"${cost:.5f}"),
        stat("Cached", "yes" if data.get("cached") else "no"),
        stat("Sources", len(cits)),
    ]
    if data.get("ttft_ms"):
        stats.append(stat("Time to first token", f"{data['ttft_ms']:.0f} ms"))
    st.markdown(f'<div class="stats">{"".join(stats)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="trace">Trace <code>{html.escape(str(data.get("trace_id", "")))}</code>'
        f' &nbsp; Cache layer <code>{html.escape(str(data.get("cache_layer", "")))}</code></div>',
        unsafe_allow_html=True)

# TODO stretch: a thumbs-down button that appends the case to a review queue.
# That queue is how real golden sets get built.
