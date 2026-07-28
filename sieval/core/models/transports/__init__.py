"""Provider frontends (Transports) for the Model IR.

Each module here implements the :class:`~sieval.core.models.transport.Transport`
protocol for one provider wire protocol.

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
