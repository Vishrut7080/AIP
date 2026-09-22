#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, apply_business_rules, extract_deterministic,TicketRecordC, CATEGORIES
)

from pydantic import Field, BaseModel

from aip.llm import structured, StructuredOutputError
from aip.cost import BudgetExceeded

from typing import Literal

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# TODO A1: choose 6 dev-set tickets. For EACH, write one line saying what it
#          teaches that prose cannot. Pick edges, not averages (T2 §2.2):
#            - the billing/complaint boundary
#            - a ticket with no policy number (teaches null)
#            - a Hinglish ticket
#            - a satisfied-but-urgent ticket (the sentiment/urgency trap)
#            - a ticket whose policy number is only in a quoted reply
#            - one you got wrong in Lab 1
FEW_SHOT_IDS: list[str] = [
    "T0054",  # teaches: mis-sold policy -> complaint (not claims/billing); angry refund demand is urgency 4, NOT 5; no policy number -> null. You got urgency+escalate wrong on this in Lab 1.
    "T0097",  # teaches: the billing/complaint boundary -- ombudsman threat is anger, not a category change; double debit stays billing; urgency 4, escalate true.
    "T0200",  # teaches: Hinglish (language=hi-en); a question about a past claim stays claims; frustrated tone is NOT escalate; urgency 3. You got category+urgency wrong on this in Lab 1.
    "T0095",  # teaches: a factual "account question" (wellness points) is information, not a claim/query; urgency 2, not 1 or 3; PII true. You got urgency wrong on this in Lab 1.
    "T0048",  # teaches: an action request with an embedded question ("add my mother... what will the premium impact be?") stays policy_change; no policy number -> null; urgency 2.
    "T0222",  # teaches: the reverse sentiment trap -- a SATISFIED customer still needs routing (information, urgency 1); sentiment never sets urgency; extract product=platinum.
]

# Zero Shot - Direct Prompt
# Few Shot - Prompt + Examples (Edge Cases)

def load_examples(ids: list[str]) -> list[dict]:
    """Loads examples from the extraction_dev.jsonl file"""

    # [json.loads(line) for line in file]
    # path = (ROOT / "data/eval/extraction_dev.jsonl")
    # file = path.open(encoding="utf-8")
    # rows = [json.loads(line) for line in file]
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]

    # rows list[dict] -> [{id: "", input: ""}, {id: "", input: ""}, {}]
    by_id = {r["id"]: r for r in rows}
    # dict[dict] => { id: {id: "", input: ""}, id2: {} }
    # T0200: { id: "", input: "", expected: { category: "" } }

    # Error if invalid ID
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")

    # list[dict] -> [{ id: "", input: "" }, {}, {}]
    return [by_id[i] for i in ids]


# LLM Example -> Output {} -> Supporting Evidence.
# Input ~~> Output
_EVIDENCE_FOR = {
    "T0054": "Your agent mis-sold me this policy.",
    "T0097": "THIS IS THE THIRD TIME I am writing about teh double debit on AUR-7548999",
    "T0200": "My claim on AUR-9674338 was settled at Rs 41800 but the hospital bill was much higher",
    "T0095": "How many wellness points do I currently have on AUR-9746149",
    "T0048": "Please add my mother as a dependent on my policy.",
    "T0222": "Can you confirm the restoration benefit is still available this year on AUR-4940770?",
}


def few_shot_block(ids: list[str]) -> str:
    """Render the examples into the prompt.

    The example output format must be byte-identical to the format you are
    asking the model to produce. A mismatch here is a classic own goal.
    """
    prompt = "Examples:\n\n"
    for ex in load_examples(ids):
        out = {
            "evidence": _EVIDENCE_FOR.get(ex["id"], ex["input"][:200]),
            "category": ex["expected"]["category"],
            "urgency": ex["expected"]["urgency"],
            "sentiment": ex["expected"]["sentiment"],
            "product": ex["expected"]["product"],
            "language": ex["expected"]["language"],
        }
        prompt += f"INPUT:\n{ex['input']}\n\nOUTPUT:\n{json.dumps(out)}\n"
    return prompt

# IDS
# Example:
#
# INPUT:
# .........
# 
# OUTPUT:
# { category: "", urgency: "", ... }
# 
# INPUT:
# ...........
#
# OUTPUT:
# { category: "", urgency: "", ... }

# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------

def _fallback(e=""):
    """Baseline TicketRecord. """
    return {
        "category": "information",
        "urgency": 1,
        "sentiment": "neutral",
        "product": "unknown",
        "language": "en",
        "evidence": "",
        "policy_number": None,
        "contains_pii": False,
        "escalate": False,
        "needs_human_review": True,
        "review_reason": f"StructuredOutputError: {e}"
    }
    

# structured() -> calls llm with prompt + schema + system_prompt -- outputs -> structured JSON in the shape of (schema) TicketRecord

def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: Lab 1 Part C, no examples. This is your baseline."""
    # Zero Shot: Directly Provide the Ticket to the LLM without any alterations
    try:
        rec=structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier=tier).model_dump()
        # .model_dump converts into dict (json)
        # Record (JSON)
        # Add policy_number (obtained with regex) + contains_pii to the record
        rec.update(extract_deterministic(ticket))
        # Perform escalation logic -> updated record -> final structured output
        return apply_business_rules(rec,ticket)
    except StructuredOutputError as e:
        return _fallback(e) # Fallback TicketRecord with needs_human_review: true and review_reason: error
    except BudgetExceeded:
        raise


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: zero_shot + the few-shot block."""
    try:
        prompt = ticket + "\n\n" + few_shot_block(FEW_SHOT_IDS)
        # Hi, my name ...
        # ---------------
        # Examples
        # INPUT
        # ...
        # OUT
        # ...


        record = structured(prompt, schema=TicketRecordC, system=SYSTEM_PROMPT, tier=tier)
        res = record.model_dump()
        res.update(extract_deterministic(ticket))
        res = apply_business_rules(res, ticket)
        return res
    except StructuredOutputError as e:
        return _fallback(e)
    except BudgetExceeded:
        raise

# TicketRecordC

class TicketRecordReasoned(BaseModel):
    """TODO B: add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation. You want the
    first. Measure the difference in output tokens.
    """
    # description="A direct quote or tight paraphrase (max 200 chars) from the ticket "
    # "text that most directly supports the category and urgency you "
    # "assigned. Must be grounded in the actual message — never invent or "
    # "infer text that isn't there.",
    reasoning: str = Field(max_length=200, description="Explain the reasoning behind your logic. Must be grounded in the actual message and support the evidence.")    

    evidence:str=Field(max_length=200, description="A direct quote or tight paraphrase (max 200 chars) from the ticket "
    "text that most directly supports the category and urgency you "
        "assigned. Must be grounded in the actual message — never invent or "
        "infer text that isn't there.",)
    
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
    
    product:Literal["bronze","silver","gold","platinum","unknown"]=Field(description="The plan the customer names in the message, literally one of: "
            "bronze, silver, gold, platinum.\n"
            "Only set it if the plan is explicitly named in the message text. "
            "Use \"unknown\" in every other case.\n"
            "Never infer the plan from context — not from the sum insured, "
            "not from a premium amount, not from coverage details."
    )
    language:Literal["en","hi-en"]=Field(description="The dominant language of the message, one of:\n"
            "en = English only.\n"
            "hi-en = a mix of English and Hindi, including Hindi transliterated "
            "into Latin script (e.g. kripya, jaldi, bahut, turant, paisa).\n"
            "A single Hindi word is enough to qualify as hi-en."
    )
    
    needs_human_review: bool = False
    review_reason: str = ""



def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """TODO B: few_shot with TicketRecordReasoned."""
    try:
        prompt = ticket + "\n\n" + few_shot_block(FEW_SHOT_IDS)
        # TicketRecordReasoned includes reasoning field
        record = structured(prompt, schema=TicketRecordReasoned, system=SYSTEM_PROMPT, tier=tier)
        res = record.model_dump()
        res.update(extract_deterministic(ticket))
        res = apply_business_rules(res, ticket)
        return res
    except StructuredOutputError as e:
        return _fallback(e)
    except BudgetExceeded:
        raise


def cascade(ticket: str) -> dict:
    """TODO C: SMALL first; escalate to MAIN on a trigger you choose.

    Triggers, roughly in ascending order of how well they work:
      - validation failed                      (free, weak: misses confident errors)
      - evidence field empty or very short     (free, surprisingly decent)
      - urgency >= 4                           (free, but it is not a confidence signal)
      - two SMALL samples at T=0.7 disagree    (2x small cost, much the best)

    Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
    so grid.py can report the escalation rate.
    """
    trigger = False

    prompt = ticket + "\n\n" + few_shot_block(FEW_SHOT_IDS)

    # Ask two smaller models for the structred output.
    # s1: zero temperature. strict response
    # s2: 0.7 temperature. creative response
    s1 = s2 = _fallback("SMALL sample failed") # Initialize to something
    try:
        # Temperature -> Entropy / Randomness -> "Creative"
        s1 = structured(prompt, schema=TicketRecordC, system=SYSTEM_PROMPT,
                        tier="SMALL", temperature=0.0).model_dump()
        s2 = structured(prompt, schema=TicketRecordC, system=SYSTEM_PROMPT,
                        tier="SMALL", temperature=0.7).model_dump()
    except StructuredOutputError as e:
        trigger = True     # If validation fails, trigger MAIN model

    # or if the evidence is empty or too short (less than 40 character)
    if len(s1['evidence'].strip()) < 40:
         trigger = True
    # or If the two models disagree...
    elif (s1['category'] != s2['category']) or (s1['urgency'] != s2['urgency']) or (s1['sentiment'] != s2['sentiment']):
        trigger = True

    
    if trigger:
        # Use the larger model if required
        try:
            record = structured(prompt, schema=TicketRecordReasoned, system=SYSTEM_PROMPT, tier='MAIN')
            res = record.model_dump()
            res.update(extract_deterministic(ticket))
            res = apply_business_rules(res, ticket)
            res["_path"] = 'large'
            return res
        except StructuredOutputError as e:
            fb = _fallback(e)
            fb["_path"] = "large"
            return fb
    else:
        # Otherwise use the smaller model
        s1.update(extract_deterministic(ticket))
        s1 = apply_business_rules(s1, ticket)
        s1["_path"] = 'small'         
        return s1

           
VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
