"""Canonical, secret-free fingerprints shared by the model binding plane.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import hashlib
import json
from collections.abc import Mapping


def fingerprint_mapping(payload: Mapping[str, object]) -> str:
    """Hash a mapping through the model plane's single canonical JSON form."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
