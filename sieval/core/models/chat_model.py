"""ChatModel: chat completions API backend.

RFC #25 moved the wire logic (streaming accumulation, reasoning extraction,
logprob parsing) into
:class:`~sieval.core.models.transports.openai_chat.OpenAIChatTransport`; this
class is the backend selector that pairs the shared client/limiter pool with
that transport.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from collections.abc import Iterable

from openai.types.chat import ChatCompletionMessageParam

from .model import Model
from .transport import Transport


class ChatModel(Model[str | Iterable[ChatCompletionMessageParam]]):
    """Model backend for the chat completions API (streaming + non-streaming)."""

    def _build_default_transport(self) -> Transport:
        from .transports.openai_chat import OpenAIChatTransport

        return OpenAIChatTransport(client=self._client, model=self._model)
