"""Tests for the model plane's canonical fingerprint primitive.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import pytest

from sieval.core.models._fingerprint import fingerprint_mapping


def test_fingerprint_is_stable_across_mapping_order() -> None:
    assert fingerprint_mapping({"b": [2, 3], "a": 1}) == fingerprint_mapping(
        {"a": 1, "b": [2, 3]}
    )


def test_fingerprint_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        fingerprint_mapping({"value": float("nan")})
