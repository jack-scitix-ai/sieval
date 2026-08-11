"""Deprecated import path for :class:`OpenAICompletionsDialect`.

The alias retains the old direct ``arun(Request)`` entry point for one
compatibility cycle; canonical execution goes through ``Model`` and its pool.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from sieval.core.models.dialects.openai_completions import OpenAICompletionsDialect

OpenAICompletionsTransport = OpenAICompletionsDialect

__all__ = ["OpenAICompletionsTransport"]
