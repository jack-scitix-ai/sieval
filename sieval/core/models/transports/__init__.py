"""Deprecated transport names and the temporary SGLang legacy executor.

The two OpenAI names re-export their canonical
:class:`~sieval.core.models.dialect.Dialect` implementations for one
compatibility cycle.  ``SglangTransport`` is different:
it remains the unchanged native ``/generate`` executor behind
``SglangGenModel``'s explicit ``sglang_legacy`` bypass.  It is not a canonical
Dialect and must not be registered as ``sglang_native`` before PR 5.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from .openai_chat import OpenAIChatTransport
from .openai_completions import OpenAICompletionsTransport
from .sglang import SglangTransport

__all__ = [
    "OpenAIChatTransport",
    "OpenAICompletionsTransport",
    "SglangTransport",
]
