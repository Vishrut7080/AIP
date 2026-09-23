#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Framework: retrieve -> (rerank) -> generate -> validate -> maybe repair.

The answer prompt (Part A), the code-level citation check (Part B), the
refusal escape hatch (Part C) and the repair decision (B3) all live here.
`aip/rag.py` is the reference; this is my own stand alone implementation.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import tracing  # noqa: E402
from aip.chunking import markdown_chunks  # noqa: E402
from aip.guards import (  # noqa: E402
    UNTRUSTED_SYSTEM_CLAUSE,
    delimit_untrusted,
    enforce_citations,
)
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# The generation prompt (Part A). Six required elements per the runsheet:
#   1. answer ONLY from the numbered sources, never general knowledge
#   2. cite by index, [1] or [2][5]
#   3. never cite a number that was not supplied
#   4. the exact refusal string, verbatim
#   5. contradictions are surfaced, never silently resolved
#   6. length discipline -- output tokens dominate latency and cost
# Plus one rule the shipped corpus actually needs: a PARTIAL refusal. Q37 and
# Q40 are only part-way unanswerable -- the grounded part must be answered with
# a citation, and only the unsupported remainder refuses (T4 §6.3).
def _system_common(rules: list[str]) -> str:
    """Assemble ANSWER_SYSTEM from the shared non-refusal rules + untrusted clause."""
    body = "\n".join(rules)
    return f"""\
You answer questions using ONLY the numbered sources provided.

Rules, in priority order:
{body}
5. If sources disagree, say so and cite both. Do not pick one silently.
6. Be concise. Two or three sentences unless the question needs more.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


# The refusal dial (Part C4). All three prompts are identical except rule 1/2's
# refusal threshold, so the ONLY thing that changes between runs is how readily
# the system declines. `default` is the middle setting.
ANSWER_SYSTEM = _system_common([
    f'1. If the sources do not contain the answer, reply exactly:\n   "{REFUSAL}"\n'
    "   Do not guess, and do not fall back on general knowledge.",
    "2. If the sources contain part of the answer, answer that part with "
    "citations and refuse ONLY the unsupported remainder -- using the exact "
    "refusal sentence above for that remainder.",
    "3. Every factual sentence must end with a citation of the source(s) that "
    "support it, in the form [1] or [2][5].",
    "4. Never cite a number that was not given to you.",
])

# Lenient: answer from the sources whenever ANY part is usable; refuse only
# when NOTHING supports the question. Recovers the five wrongful refusals.
ANSWER_SYSTEM_LENIENT = _system_common([
    "1. Answer the question using everything the sources support, even if that "
    "is only part of what is asked. Reply with the exact refusal sentence "
    f'   "{REFUSAL}" ONLY if none of the numbered sources contains anything '
    "relevant. Do not add facts from general knowledge.",
    "2. If only part is answerable, answer that part with citations and say "
    "plainly that the rest is not in the provided sources (a partial answer is "
    "better than a refusal and better than inventing the missing part).",
    "3. Every factual sentence must end with a citation of the source(s) that "
    "support it, in the form [1] or [2][5].",
    "4. Never cite a number that was not given to you.",
])

# Strict: refuse on any doubt. Expected to push precision DOWN and recall UP
# (the trade-off C4 exists to measure), and it is NOT the deployment pick for
# an insurance helpdesk -- see the report.
ANSWER_SYSTEM_STRICT = _system_common([
    f"1. Reply with the exact refusal sentence \"{REFUSAL}\" if you are not "
    "certain the numbered sources fully contain the answer. Do not guess, and "
    "do not infer beyond what the text literally states.",
    "2. Only answer when the sources fully support the answer; otherwise the "
    "exact refusal sentence above is the entire reply.",
    "3. Every factual sentence must end with a citation of the source(s) that "
    "support it, in the form [1] or [2][5].",
    "4. Never cite a number that was not given to you.",
])

STRICTNESS = {
    "default": ANSWER_SYSTEM,
    "lenient": ANSWER_SYSTEM_LENIENT,
    "strict": ANSWER_SYSTEM_STRICT,
}


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False
    n_repairs: int = 0


_CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_answer(text: str, n_sources: int,
                    finish_reason: str | None = None) -> dict:
    """B2. Validate the output contract established in Part A.

    Returns at least:
        {"valid": bool, "refused": bool, "invalid_citations": [ints],
         "n_citations": int, "truncated": bool, "reason": str}

      - every [n] is between 1 and n_sources (aip.guards.enforce_citations)
      - not truncated (finish_reason == "length"): a cut-off prose answer
        LOOKS FINE -- T1 failure mode 4, the dangerous one
      - a non-refusal answer contains at least one citation
      - a refusal is allowed to carry no citations
    """
    text = (text or "").strip()
    truncated = finish_reason == "length"
    refused = text.startswith(REFUSAL)

    if not text:
        return {"valid": False, "refused": False, "invalid_citations": [],
                "n_citations": 0, "truncated": truncated, "reason": "empty answer"}
    if refused:
        return {"valid": not truncated, "refused": True, "invalid_citations": [],
                "n_citations": 0, "truncated": truncated,
                "reason": "refusal (no citations required)"}
    if truncated:
        return {"valid": False, "refused": False, "invalid_citations": [],
                "n_citations": len(_CITATION_RE.findall(text)),
                "truncated": True, "reason": "answer was truncated (finish_reason=length)"}

    ok, invalid = enforce_citations(text, n_sources)
    n_citations = len(_CITATION_RE.findall(text))
    if invalid:
        return {"valid": False, "refused": False, "invalid_citations": invalid,
                "n_citations": n_citations, "truncated": False,
                "reason": f"citations out of range: {invalid}"}
    if n_citations == 0:
        return {"valid": False, "refused": False, "invalid_citations": [],
                "n_citations": 0, "truncated": False,
                "reason": "non-refusal answer has no citation"}
    return {"valid": True, "refused": False, "invalid_citations": [],
            "n_citations": n_citations, "truncated": False, "reason": ""}


def _generate_text(question: str, hits: list[Hit], *, tier: str,
                   strictness: str = "default") -> tuple[str, str | None]:
    """Stage 6: one generation call, full response so we get finish_reason."""
    context = delimit_untrusted(format_context(hits))
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    with tracing.trace("rag.generate", n_sources=len(hits), tier=tier,
                       strictness=strictness):
        res = chat(prompt, system=STRICTNESS[strictness], tier=tier,
                   temperature=0.0, max_tokens=600, return_full=True)
    text = (res.get("text") or "").strip()
    return text, res.get("finish_reason")


_CORRECTIVE = (
    "Your answer above was rejected for the following reason: {reason}.\n"
    "Rewrite it to satisfy ALL of these: answer only from the numbered sources; "
    "never cite a number you were not given; a non-refusal needs at least one "
    "citation; if the sources lack the answer, reply with the exact refusal "
    "sentence; and do not truncate. Return only the corrected answer."
)


def _solve(question: str, hits: list[Hit], *, tier: str,
           max_repairs: int = 1, strictness: str = "default") -> Answer:
    """Generate -> validate -> maybe repair (B3). Shared by both paths so the
    E2 comparison never mixes generators.

    B3 decision: one corrective retry, then fall back to refusal. Rationale:
      - a single repair fixes the overwhelmingly common failures (forgot a
        citation, cited an out-of-range index) without burning budget;
      - if the retry still fails, the honest output is a refusal -- inventing
        content we could not ground is exactly what this lab exists to prevent;
      - the invariant is absolute: never return citations_valid=False with
        refused=False. That combination is the thing we are paid to stop.
    """
    with tracing.trace("rag.solve", n_hits=len(hits), tier=tier,
                       max_repairs=max_repairs, strictness=strictness) as span:
        text, finish = _generate_text(question, hits, tier=tier,
                                      strictness=strictness)
        check = validate_answer(text, len(hits), finish_reason=finish)

        repairs = 0
        while not check["valid"] and not check["refused"] and repairs < max_repairs:
            repairs += 1
            tracing.event("rag.repair", attempt=repairs, reason=check["reason"])
            with tracing.trace("rag.repair.generate", attempt=repairs, tier=tier):
                res = chat(
                    _CORRECTIVE.format(reason=check["reason"]),
                    system=STRICTNESS[strictness], tier=tier,
                    temperature=0.0, max_tokens=600, return_full=True,
                )
            text = (res.get("text") or "").strip()
            finish = res.get("finish_reason")
            check = validate_answer(text, len(hits), finish_reason=finish)

        span["repaired"] = repairs > 0
        span["n_repairs"] = repairs
        span["valid"] = check["valid"]
        span["refused"] = check["refused"]

    if check["valid"]:
        return Answer(
            question=question, text=text, hits=list(hits),
            refused=check["refused"], citations_valid=True,
            invalid_citations=check["invalid_citations"],
            n_citations=check["n_citations"], truncated=check["truncated"],
            n_repairs=repairs,
        )

    # Exhausted reparability: the only safe output left is a refusal.
    tracing.event("rag.fallback_refusal", reason=check["reason"])
    return Answer(
        question=question, text=REFUSAL, hits=list(hits), refused=True,
        citations_valid=True, invalid_citations=[], n_citations=0,
        truncated=False, n_repairs=repairs,
    )


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN",
                    strictness: str = "default") -> Answer:
    """Retrieve -> (rerank) -> generate -> validate -> maybe repair."""
    with tracing.trace("rag.answer_question", k=k, final_k=final_k,
                       strictness=strictness, question=question[:120]) as span:
        with tracing.trace("rag.retrieve", k=k):
            hits = retriever.search(question, k=k)
        span["n_retrieved"] = len(hits)
        if reranker is not None:
            with tracing.trace("rag.rerank", n_in=len(hits), n_out=final_k):
                final = reranker.rerank(question, hits, k=final_k)
        else:
            final = hits[:final_k]
        span["n_final"] = len(final)
        return _solve(question, final, tier=tier, strictness=strictness)


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN", strictness: str = "default") -> Answer:
    """E2: same generator as answer_question, but the context is the gold
    documents (chunked with the same Lab 3 winner config, markdown-400).
    No retrieval at all. The difference between the two calls is exactly the
    damage the retriever is doing.
    """
    hits: list[Hit] = []
    for i, doc_text in enumerate(gold_docs):
        chunks = markdown_chunks(doc_text, f"gold{i}", size=400)
        for c in chunks:
            hits.append(Hit(c, 1.0, "gold", len(hits)))
    with tracing.trace("rag.gold_context", n_docs=len(gold_docs),
                       n_chunks=len(hits), tier=tier, strictness=strictness):
        if not hits:
            return Answer(question=question, text=REFUSAL, hits=[],
                          refused=True, citations_valid=True, n_repairs=0)
        return _solve(question, hits, tier=tier, strictness=strictness)