#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# TODO A: write this before you read aip/rag.py::ANSWER_SYSTEM.
ANSWER_SYSTEM = f"""\
TODO. Six required elements, see the handout Part A. Include
{UNTRUSTED_SYSTEM_CLAUSE!r} or your own equivalent -- Lab 6 will attack this.
"""


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


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """TODO B2. Return a dict with at least:

        {"valid": bool, "refused": bool, "invalid_citations": [ints],
         "n_citations": int, "truncated": bool, "reason": str}

    Checks:
      - every [n] is between 1 and n_sources
      - not truncated (finish_reason == "length" means the answer was cut off,
        and a cut-off prose answer LOOKS FINE -- this is T1 failure mode 4 and
        it is the dangerous one)
      - a non-refusal answer contains at least one citation
    """
    raise NotImplementedError


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN") -> Answer:
    """TODO: retrieve -> (rerank) -> generate -> validate -> maybe repair.

    B3: decide what happens when validation fails. Whatever you decide, the
    function must never return an Answer with citations_valid=False and
    refused=False. That combination is the thing you are being paid to prevent.
    """
    raise NotImplementedError


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """TODO E2: same generator, but the context is the gold documents.

    No retrieval at all -- read data/corpus/<doc_id>.md for each gold doc,
    chunk it or pass it whole, and generate. The difference between this and
    answer_question() is the damage your retriever is doing.
    """
    raise NotImplementedError
