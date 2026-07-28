"""Tests for the IR primitive on Model: arun, capabilities, assert_capability,
and the alogprobs echo gate.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import pytest

from sieval.core.models import (
    Capability,
    CapabilityError,
    ChatModel,
    GenModel,
    Request,
    Response,
    UsageStats,
)


class _StubTransport:
    def __init__(self, caps: frozenset[Capability], response: Response):
        self._caps = caps
        self._response = response
        self.calls: list[Request] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._caps

    async def arun(self, req: Request) -> Response:
        self.calls.append(req)
        return self._response


class TestCapabilities:
    def test_chat_model_capabilities_from_transport(self):
        m = ChatModel(model="c", api_key="k")
        assert Capability.Chat in m.capabilities
        assert Capability.InputScoring not in m.capabilities

    def test_gen_model_has_input_scoring(self):
        m = GenModel(model="g", api_key="k")
        assert Capability.InputScoring in m.capabilities
        assert Capability.Completion in m.capabilities

    def test_assert_capability_passes_when_present(self):
        m = GenModel(model="g", api_key="k")
        m.assert_capability(Capability.Completion, Capability.InputScoring)

    def test_assert_capability_raises_when_missing(self):
        m = ChatModel(model="c", api_key="k")
        with pytest.raises(CapabilityError, match="InputScoring"):
            m.assert_capability(Capability.InputScoring)


class TestArun:
    @pytest.mark.anyio
    async def test_arun_delegates_to_transport(self):
        resp = Response(
            texts=("hi",), usage=UsageStats(output_tokens=1, total_tokens=1)
        )
        stub = _StubTransport(frozenset({Capability.Chat}), resp)
        m = ChatModel(model="c", api_key="k", transport=stub)
        out = await m.arun(Request(input="prompt"))
        assert out is resp
        assert len(stub.calls) == 1
        assert stub.calls[0].input == "prompt"

    @pytest.mark.anyio
    async def test_injected_transport_supplies_capabilities(self):
        stub = _StubTransport(frozenset({Capability.Chat}), Response(texts=()))
        m = ChatModel(model="c", api_key="k", transport=stub)
        assert m.capabilities == frozenset({Capability.Chat})


class TestAlogprobsEchoGate:
    @pytest.mark.anyio
    async def test_echo_true_on_chat_raises_capability_error(self):
        """The historical bug: echo=True on chat was silently ignored."""
        from tests.conftest import MockChatModel

        m = MockChatModel()
        with pytest.raises(CapabilityError, match="InputScoring"):
            await m.alogprobs("prompt", echo=True)

    @pytest.mark.anyio
    async def test_echo_true_on_gen_works(self):
        from tests.conftest import MockGenModel

        m = MockGenModel()
        out = await m.alogprobs("prompt", echo=True)
        assert out.logprobs is not None

    @pytest.mark.anyio
    async def test_echo_false_on_chat_skips_the_gate(self):
        from tests.conftest import MockChatModel

        m = MockChatModel()
        # echo=False must skip the InputScoring gate and reach the transport.
        # The chat stub serves no logprob channel, so the post-arun contract
        # check fires — proving the gate was bypassed and the request ran.
        with pytest.raises(RuntimeError, match="server returned none"):
            await m.alogprobs("prompt", echo=False)
        assert m._transport.requests[0].score_input is False
