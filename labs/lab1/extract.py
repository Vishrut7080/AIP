#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # TODO B1a: Should `evidence` be declared here, BEFORE the fields it
    #           justifies, or after them? T2 §3.3. Decide, move it, and leave
    #           a one-line comment saying which effect you chose and why.

    category: CATEGORIES = Field(
        description="TODO B1b: define each of the six categories in one clause "
                    "each. Pay particular attention to the boundary between "
                    "'complaint' and the category the complaint is about."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="TODO B1c: define the 1-5 scale concretely. Anchor at least "
                    "points 1, 3 and 5 with a describable situation. If you do "
                    "not define the scale, the model invents one, and it will "
                    "not be the one the gold labels use."
    )

    # TODO B1d: sentiment  -> Literal["angry","frustrated","neutral","satisfied"]
    # TODO B1e: product    -> Literal["bronze","silver","gold","platinum","unknown"]
    #           Note "unknown" is a legal value. Say explicitly when to use it.
    # TODO B1f: language   -> Literal["en","hi-en"]
    # TODO B1g: evidence   -> str, max_length=200, "the span of the ticket that
    #           determined the category, quoted verbatim"

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.
    policy_number: str | None = Field(
        default=None,
        description="TODO B1h: state the exact format, and state explicitly "
                    "that null is required when no policy number appears. "
                    "Forbid inventing or reformatting one."
    )
    contains_pii: bool = Field(
        default=False,
        description="TODO B1i"
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        # TODO B1j: reject anything that is not exactly AUR-<7 digits>.
        #           Return None rather than raising if the model returned an
        #           empty string or the literal "null" -- decide which of those
        #           two behaviours you want and defend it in your report.
        return v


SYSTEM_PROMPT = """\
TODO B2: write this using the seven-component structure from T2 §2.

Order it for attention AND for prompt caching: stable instructions first,
volatile data last. The ticket text is injected by the caller, after this.

It should be shorter than your first instinct. Most of what you want to say
belongs in the field descriptions above.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    # TODO B3: call aip.llm.structured with TicketRecord.
    # TODO B4: catch StructuredOutputError and return a record with
    #          needs_human_review=True. This function must never raise.
    raise NotImplementedError


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    """TODO C1: return {'policy_number', 'contains_pii'} without a model call.

    policy_number:
        Find AUR-<7 digits>.

    TODO C3 -- the trap. Some tickets contain TWO policy-number-shaped strings:
        one in the live body, and one in a quoted reply below a '>' line from
        an earlier thread. They are not always the same number.

        Decide a rule. Write it down in a comment right here. Implement it.
        Then ask yourself whether it generalises or whether you have fitted it
        to this dataset -- the honest answer is worth marks.

    contains_pii:
        True if the ticket contains a phone number or an email address.
        aip.guards._PII_PATTERNS has the patterns. Note that a *name* alone
        does not count for this dataset's labels -- check the gold data and
        say in your report whether you think that definition is right.
    """
    raise NotImplementedError


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """TODO C1b: compute `escalate` in code.

        escalate = urgency >= 4 or 'ombudsman' appears in the ticket

    This is a business rule. It belongs in code where it can be read by a
    compliance officer, changed without touching a prompt, and unit-tested.
    Write the unit test in tests/ while you are here.
    """
    raise NotImplementedError


class TicketRecordC(BaseModel):
    """TODO C2: the reduced schema the model sees in Part C.

    Copy TicketRecord and delete the fields you now compute in code. Fewer
    fields means a shorter prompt, fewer output tokens, and three fields at
    100% accuracy. Measure all three effects.
    """


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))
