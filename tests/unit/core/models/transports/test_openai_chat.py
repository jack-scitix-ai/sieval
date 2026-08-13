"""Compatibility checks for the deprecated OpenAI chat transport path.

Wire behavior is covered by ``tests/unit/core/models/dialects/test_openai_chat.py``;
this module only protects the one-cycle import alias.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from sieval.core.models.dialects.openai_chat import OpenAIChatDialect
from sieval.core.models.transports.openai_chat import OpenAIChatTransport


def test_openai_chat_transport_is_dialect_alias() -> None:
    assert OpenAIChatTransport is OpenAIChatDialect
