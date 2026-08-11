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

import pytest

from sieval.core.models import (
    Capability,
    ChatInput,
    ChatMessage,
    ChatModel,
    GenModel,
    TextPart,
)
from sieval.core.models.transports.openai_chat import OpenAIChatTransport


class TestDefaultTransport:
    def test_builds_openai_chat_transport(self):
        m = ChatModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAIChatTransport)
        assert Capability.Chat in m.capabilities

    def test_transport_bound_to_shared_client_and_model(self):
        m = ChatModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAIChatTransport)
        assert m._transport._client is m._client
        assert m._transport._requested_model_id == "m"


class TestCompatibilityBranches:
    def test_coerce_input_preserves_chat_ir_and_rejects_mapping(self):
        model = ChatModel(model="m", api_key="k")
        prompt = ChatInput((ChatMessage("user", (TextPart("hi"),)),))

        assert model._coerce_input(prompt) is prompt
        with pytest.raises(TypeError, match="text, ChatInput, or messages"):
            model._coerce_input({"role": "user", "content": "hi"})

    def test_as_type_covers_both_targets_and_explicit_runtime_plan(self):
        model = ChatModel(model="m", api_key="k")
        plan = model.runtime_plan
        assert plan is not None

        same = model.as_type(ChatModel)
        completion = model.as_type(GenModel)
        explicit = model.as_type(ChatModel, plan)

        assert type(same) is ChatModel
        assert type(completion) is GenModel
        assert explicit.runtime_plan is plan

    def test_as_type_rejects_derived_wrapper_class(self):
        model = ChatModel(model="m", api_key="k")

        class DerivedChatModel(ChatModel):
            pass

        with pytest.raises(TypeError, match="exactly ChatModel or GenModel"):
            model.as_type(DerivedChatModel)

    def test_session_bound_chat_wrapper_requires_explicit_runtime_plan(self):
        model = ChatModel(model="m", api_key="k")
        session_wrapper = model.as_compat_type(ChatModel)
        assert isinstance(session_wrapper, ChatModel)

        with pytest.raises(ValueError, match="explicitly reconciled"):
            session_wrapper.as_type(GenModel)
