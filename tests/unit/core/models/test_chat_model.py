"""Shell tests: backend selector wiring for ChatModel.

RFC #25 moved the chat-completions wire logic (streaming accumulation,
reasoning extraction, logprob parsing) into ``OpenAIChatTransport``; the
former ``_agenerate_impl`` / ``_alogprobs_impl`` coverage moved with it to
tests/unit/core/models/transports/test_openai_chat.py, and the request-builder
validation (n/stream types, alogprobs n=1) lives on ``Model`` in
tests/unit/core/models/test_model.py. What remains here is the selector
contract: ChatModel pairs the shared client with that transport, which
supplies the model's capabilities.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from sieval.core.models import Capability, ChatModel
from sieval.core.models.transports.openai_chat import OpenAIChatTransport


class TestDefaultTransport:
    def test_builds_openai_chat_transport(self):
        m = ChatModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAIChatTransport)
        assert Capability.Chat in m.capabilities

    def test_transport_bound_to_shared_client_and_model(self):
        m = ChatModel(model="m", api_key="k")
        assert m._transport._client is m._client
        assert m._transport._model == "m"
