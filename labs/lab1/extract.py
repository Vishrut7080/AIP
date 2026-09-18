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

    # TODO B1g: evidence   -> str, max_length=200, "the span of the ticket that
        #           determined the category, quoted verbatim"

    evidence:str=Field(max_length=200, description="A direct quote or tight paraphrase (max 200 chars) from the ticket "
    "text that most directly supports the category and urgency you "
    "assigned. Must be grounded in the actual message — never invent or "
    "infer text that isn't there.",)

    # description="TODO B1b: define each of the six categories in one clause "
    #                     "each. Pay particular attention to the boundary between "
    #                     "'complaint' and the category the complaint is about."
    #     )
    category: CATEGORIES = Field(
        description="One of: billing, claims, policy_change, technical, complaint, "
        "information.\n"
        "billing = money in (premium, debits, refunds, invoices, 80D tax "
        "certificate, instalments).\n"
        "claims = an actual or intended claim (cashless, reimbursement, "
        "settlement amount, deduction, rejection).\n"
        "policy_change = altering the contract (add/remove a member, "
        "upgrade, port, change contact details).\n"
        "technical = app, portal, OTP, login, locator, or upload is broken.\n"
        "complaint = the subject is Aurora's own conduct — mis-selling, "
        "being kept on hold, an ignored grievance.\n"
        "information = a question with no pending transaction behind it.\n"
        "Key boundary: an angry message about a claim is still 'claims' if "
        "the customer wants the claim processed. It's only 'complaint' when "
        "Aurora's conduct itself is the subject, not the claim outcome.",
    )

    # "TODO B1c: define the 1-5 scale concretely. Anchor at least "
    #                 "points 1, 3 and 5 with a describable situation. If you do "
    #                 "not define the scale, the model invents one, and it will "
    #                 "not be the one the gold labels use."
    
    urgency: int = Field(
        ge=1, le=5,
        description="Urgency of this message, 1 to 5 (5 = most urgent). Judge by "
        "situation, not tone — shouting isn't urgency.\n"
        "1 = general knowledge/self-service, no account lookup needed "
        "(e.g. 'waiting period for cataract surgery?').\n"
        "2 = needs account lookup/action, or a transaction in flight "
        "(e.g. 'add my newborn', 'app crashes on upload').\n"
        "3 = something's already gone wrong and customer is waiting "
        "(e.g. 'debited twice').\n"
        "4 = repeated failure, money/access at risk now, or threatens "
        "escalation (e.g. 'THIS IS THE THIRD TIME').\n"
        "5 = active emergency, formal denial needing immediate reversal, or "
        "states (not threatens) they're escalating to Ombudsman "
        "(e.g. 'father in ICU, cashless DENIED').\n"
        "1v2: needing to touch the account = at least 2. "
        "4v5: 'will go to ombudsman' = 4; 'am filing' = 5. "
        "+1 (cap 5) if a same-day/next-morning deadline is stated.",
    )

    # TODO B1d: sentiment  -> Literal["angry","frustrated","neutral","satisfied"]
    sentiment:Literal["angry","frustrated","neutral","satisfied"]=Field(description="The customer's tone toward Aurora, one of:\n"
        "angry = hostile, shouting, threatening.\n"
        "frustrated = unhappy and tired of trying, but still civil.\n"
        "neutral = matter-of-fact. Is the default for a first request, "
        "however terse.\n"
        "satisfied = thanks or praise.\n"
        "Judge tone only — it is independent of urgency. A furious message "
        "about a tax certificate is neutral/frustrated, not angry.\n"
        "Key boundary: 'frustrated' requires a prior failure — a repeat "
        "attempt, an unanswered request, or a delay. A first-time complaint "
        "with no history is 'neutral', not 'frustrated'."
)
    # TODO B1e: product    -> Literal["bronze","silver","gold","platinum","unknown"]
    #           Note "unknown" is a legal value. Say explicitly when to use it.
    product:Literal["bronze","silver","gold","platinum","unknown"]=Field(description="The plan the customer names in the message, literally one of: "
        "bronze, silver, gold, platinum.\n"
        "Only set it if the plan is explicitly named in the message text. "
        "Use \"unknown\" in every other case.\n"
        "Never infer the plan from context — not from the sum insured, "
        "not from a premium amount, not from coverage details."
)
    # TODO B1f: language   -> Literal["en","hi-en"]
    language:Literal["en","hi-en"]=Field(description="The dominant language of the message, one of:\n"
        "en = English only.\n"
        "hi-en = a mix of English and Hindi, including Hindi transliterated "
        "into Latin script (e.g. kripya, jaldi, bahut, turant, paisa).\n"
        "A single Hindi word is enough to qualify as hi-en."
)

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
