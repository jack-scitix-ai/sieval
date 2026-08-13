"""Deprecated import path for :class:`OpenAIChatDialect`.

The alias retains the old direct ``arun(Request)`` entry point for one
compatibility cycle; canonical execution goes through ``Model`` and its pool.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from sieval.core.models.dialects.openai_chat import OpenAIChatDialect

OpenAIChatTransport = OpenAIChatDialect

__all__ = ["OpenAIChatTransport"]
