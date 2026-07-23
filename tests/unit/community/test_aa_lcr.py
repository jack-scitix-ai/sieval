"""Unit tests for the AA-LCR community prompt + scoring kernels.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import pytest

from sieval.community.aa_lcr import (
    aggregate_metrics,
    build_prompt,
    parse_grade,
)


def test_build_prompt_wraps_and_orders_documents():
    prompt = build_prompt(["first doc", "second doc"], "What is the trend?")
    # Per-document markers are 1-based and in the given order.
    assert "BEGIN DOCUMENT 1:\nfirst doc\nEND DOCUMENT 1" in prompt
    assert "BEGIN DOCUMENT 2:\nsecond doc\nEND DOCUMENT 2" in prompt
    assert prompt.index("first doc") < prompt.index("second doc")
    # Outer framing + question.
    assert prompt.startswith("BEGIN INPUT DOCUMENTS")
    assert "END INPUT DOCUMENTS" in prompt
    assert "START QUESTION\n\nWhat is the trend?\n\nEND QUESTION" in prompt


def test_build_prompt_single_document_no_join_separator():
    prompt = build_prompt(["only doc"], "q")
    assert "BEGIN DOCUMENT 1:\nonly doc\nEND DOCUMENT 1" in prompt
    assert "BEGIN DOCUMENT 2" not in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("CORRECT", "CORRECT"),
        ("correct", "CORRECT"),
        ("The answer is CORRECT.", "CORRECT"),
        ("INCORRECT", "INCORRECT"),
        # "INCORRECT" contains "correct" as a substring — must not misread.
        ("incorrect", "INCORRECT"),
        # Verbose replies mentioning both: INCORRECT verdict must win.
        ("This is not correct, so INCORRECT", "INCORRECT"),
        ("INCORRECT — the candidate contradicts the official answer", "INCORRECT"),
        ("", "INCORRECT"),
        ("gibberish", "INCORRECT"),
    ],
)
def test_parse_grade(reply: str, expected: str):
    assert parse_grade(reply) == expected


def test_aggregate_metrics_accuracy():
    m = aggregate_metrics(["CORRECT", "CORRECT", "INCORRECT", "INCORRECT"])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["is_correct"] == pytest.approx(0.5)
    assert m["is_incorrect"] == pytest.approx(0.5)


def test_aggregate_metrics_empty_is_zero():
    m = aggregate_metrics([])
    assert m["accuracy"] == 0.0
    assert m["is_correct"] == 0.0
    assert m["is_incorrect"] == 0.0
