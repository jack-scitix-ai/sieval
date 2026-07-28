"""
RFC #25 Phase-4 regression suite: legacy public Model surface on the new routing.

``Model._agenerate_impl`` / ``_alogprobs_impl`` are gone — ``agenerate`` and
``alogprobs`` are now sugar over ``arun(Request) -> Response`` through a
Transport, with the Response bridged back to the legacy ``ModelOutput`` shape.
These tests pin the observable legacy contract on transport-stubbed conftest
mocks (no real HTTP):

  - ``agenerate`` output shape (texts / finish_reasons / usage, n-sampling)
  - ``alogprobs`` output shape and the ``echo=True`` InputScoring gate
    (loud ``CapabilityError`` instead of the historical silent ignore)
  - ``with_args`` derivation: generation still works, resource pool is shared
  - ``meta()`` / ``get_quota_info()`` introspection dict shapes
  - ``Model.as_type`` no longer exists

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import pytest

from sieval.core.models import Capability, CapabilityError, Model, ModelOutput
from tests.conftest import MockChatModel, MockGenModel


class TestAgenerateLegacySurface:
    @pytest.mark.anyio
    async def test_agenerate_returns_legacy_model_output(self):
        out = await MockChatModel().agenerate("hello")

        assert isinstance(out, ModelOutput)
        assert out.texts == ["unknown"]
        assert out.finish_reasons == ["stop"]
        assert out.usage == {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        }

    @pytest.mark.anyio
    async def test_agenerate_n_sampling_returns_n_texts(self):
        out = await MockChatModel().agenerate("hello", n=3)

        assert isinstance(out, ModelOutput)
        assert len(out.texts) == 3
        assert out.texts == ["unknown"] * 3
        assert out.finish_reasons == ["stop"] * 3


class TestAlogprobsLegacySurface:
    @pytest.mark.anyio
    async def test_alogprobs_no_echo_returns_logprobs(self):
        out = await MockGenModel().alogprobs("prompt A", echo=False)

        assert isinstance(out, ModelOutput)
        assert out.logprobs_tokens == [" A"]
        assert out.logprobs == [-10.0]

    @pytest.mark.anyio
    async def test_alogprobs_echo_works_on_gen_model(self):
        model = MockGenModel()
        assert Capability.InputScoring in model.capabilities

        out = await model.alogprobs("prompt A", echo=True)

        assert isinstance(out, ModelOutput)
        assert out.logprobs_tokens == [" A"]
        assert out.logprobs == [-10.0]

    @pytest.mark.anyio
    async def test_alogprobs_echo_on_chat_model_raises_capability_error(self):
        # Historically echo=True was silently ignored on chat backends;
        # RFC #25 makes the missing InputScoring capability loud.
        model = MockChatModel()
        assert Capability.InputScoring not in model.capabilities

        with pytest.raises(CapabilityError, match="InputScoring"):
            await model.alogprobs("prompt A", echo=True)


class TestWithArgsDerivation:
    @pytest.mark.anyio
    async def test_derived_model_generates_and_shares_pool(self):
        base = MockChatModel(concurrency_limit=8)
        derived = base.with_args(concurrency_limit=4)

        out = await derived.agenerate("hello")
        assert isinstance(out, ModelOutput)
        assert out.texts == ["unknown"]

        # Two-level pool: the base's limiter becomes the derived model's parent.
        assert base._limiter is not None
        assert derived._parent_limiter is base._limiter


class TestIntrospectionSurface:
    def test_meta_returns_model_meta_dict_shape(self):
        meta = MockChatModel().meta()

        assert set(meta) >= {"model", "api_base", "default_params"}
        assert meta["model"] == "mock-chat"
        assert meta["api_base"] is None
        assert meta["default_params"] == {}

    def test_get_quota_info_returns_quota_dict_shape(self):
        base = MockChatModel(concurrency_limit=8)
        info = base.get_quota_info()

        assert set(info) == {"available", "total", "parent", "child"}
        assert info["available"] == 8
        assert info["total"] == 8
        assert info["parent"] is None
        assert info["child"] == {"available": 8, "total": 8}

    def test_get_quota_info_on_derived_model_reports_parent(self):
        base = MockChatModel(concurrency_limit=8)
        derived = base.with_args(concurrency_limit=4)
        info = derived.get_quota_info()

        assert info["parent"] == {"available": 8, "total": 8}
        assert info["child"] == {"available": 4, "total": 4}
        assert info["available"] == 4


class TestRemovedSurface:
    def test_as_type_no_longer_exists(self):
        base = MockChatModel()
        assert not hasattr(base, "as_type")
        assert not hasattr(Model, "as_type")
