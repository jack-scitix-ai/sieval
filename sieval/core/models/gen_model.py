"""GenModel: text completions API backend.

RFC #25 moved the wire logic (streaming accumulation, echo split, top-logprob
sanitizing) into
:class:`~sieval.core.models.transports.openai_completions.OpenAICompletionsTransport`;
this class is the backend selector that pairs the shared client/limiter pool
with that transport.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from .model import Model
from .transport import Transport


class GenModel(Model[str]):
    """Model backend for the completions API (streaming + non-streaming)."""

    def _build_default_transport(self) -> Transport:
        from .transports.openai_completions import OpenAICompletionsTransport

        return OpenAICompletionsTransport(client=self._client, model=self._model)
