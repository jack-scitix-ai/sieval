"""Compatibility checks for the deprecated completions transport path.

Wire behavior is covered by
``tests/unit/core/models/dialects/test_openai_completions.py``; this module
protects the one-cycle import alias and its direct-call entry point.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sieval.core.models.dialects.openai_completions import OpenAICompletionsDialect
from sieval.core.models.ir import CompletionInput, Request
from sieval.core.models.transports.openai_completions import (
    OpenAICompletionsTransport,
)


def test_openai_completions_transport_is_dialect_alias() -> None:
    assert OpenAICompletionsTransport is OpenAICompletionsDialect


@pytest.mark.anyio
async def test_openai_completions_transport_retains_direct_arun() -> None:
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                text="answer",
                finish_reason="stop",
                logprobs=None,
            )
        ],
        usage=None,
        model="served-model",
        system_fingerprint=None,
    )
    create = AsyncMock(return_value=raw)
    client = SimpleNamespace(completions=SimpleNamespace(create=create))
    transport = OpenAICompletionsTransport(client, "requested-model")

    response = await transport.arun(Request(input=CompletionInput("prompt")))

    assert response.texts == ("answer",)
    create.assert_awaited_once_with(
        model="requested-model",
        prompt="prompt",
        stream=False,
    )
