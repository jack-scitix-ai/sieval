"""Deprecated compatibility name for the provider dialect protocol.

New production code imports :class:`sieval.core.models.dialect.Dialect`.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from .dialect import Dialect

Transport = Dialect

__all__ = ["Transport"]
