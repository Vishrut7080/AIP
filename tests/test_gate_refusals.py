"""Regression tests for the Lab 7 gate's refusal metrics.

These exist because `refusal_precision` was a tautology: it averaged a list
that had already been filtered down to refusals, so every element was True and
the value was 1.0 for ANY behaviour. It passed a system that refuses every
answerable question. A gate metric that cannot fail is worse than no metric,
because it reads as coverage.

The tests below pin the formula itself, so the bug cannot come back silently.
They need no API key and no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labs.lab7.gate import refusal_metrics

UNA = {"Q36", "Q37", "Q38", "Q39", "Q40"}
ANSWERABLE = ("Q01", "Q02", "Q17", "Q23")


def pairs(refused: set[str]) -> list[tuple[str, bool]]:
    """The (id, refused) pairs the gate collects.

    Every question appears exactly once, so an unanswerable question the system
    answered instead of refusing is representable -- otherwise recall could
    never drop below 1.0 and the test for that would be vacuous.
    """
    rows: list[tuple[str, bool]] = [(i, i in refused) for i in sorted(UNA)]
    rows += [(q, q in refused) for q in ANSWERABLE]
    return sorted(rows)


def score(refused: set[str]) -> dict:
    return refusal_metrics(pairs(refused), UNA)


def test_perfect_system_scores_one():
    m = score(UNA)
    assert m["refusal_precision"] == 1.0
    assert m["refusal_recall"] == 1.0
    assert m["_n_refusals"] == 5
    assert m["_wrongful_refusals"] == []


def test_refuses_everything_is_caught():
    """The bug's worst case. The tautology reported 1.0; the truth is 0.5."""
    m = score(UNA | set(ANSWERABLE))
    assert m["refusal_recall"] == 1.0, "refusing everything maximises recall"
    assert m["refusal_precision"] == pytest.approx(5 / 9, abs=1e-3)
    assert m["_wrongful_refusals"] == list(ANSWERABLE)


def test_refuses_nothing_is_caught():
    """Recall must be 0.0, not the 0.0 default hiding behind an empty list."""
    m = score(set())
    assert m["refusal_recall"] == 0.0
    assert m["refusal_precision"] == 0.0
    assert m["_n_refusals"] == 0


def test_one_wrongful_refusal_lowers_precision():
    """Q17 is the single wrongful refusal the shipping config produces."""
    m = score(UNA | {"Q17"})
    assert m["refusal_precision"] == pytest.approx(5 / 6, abs=1e-3)
    assert m["_n_rightful_refusals"] == 5
    assert m["_wrongful_refusals"] == ["Q17"]


def test_missed_unanswerable_lowers_recall_only():
    m = score({"Q36", "Q37", "Q38", "Q39"})
    assert m["refusal_recall"] == pytest.approx(0.8, abs=1e-3)
    assert m["refusal_precision"] == 1.0, "a missed refusal is not a precision error"


def test_precision_is_never_vacuous():
    """Guards the exact shape of the original bug: a value that cannot move."""
    for extra in (set(), {"Q01"}, {"Q01", "Q17"}, set(ANSWERABLE)):
        m = score(UNA | extra)
        assert 0.0 <= m["refusal_precision"] <= 1.0
    # And it must actually RESPOND to a wrongful refusal, which the tautology
    # did not: adding one wrongful refusal has to move the number down.
    good = score(UNA)["refusal_precision"]
    worse = score(UNA | {"Q01"})["refusal_precision"]
    assert worse < good
