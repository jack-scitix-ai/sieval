"""
Focused tests for non-overlapping Model behaviors.

with_args/meta branches are covered in test_model_derivation.py; the IR
primitive (arun/capabilities) in test_model_arun.py; wire lowering/lifting in
tests/unit/core/models/transports/. This file keeps unique checks: quota,
runtime concurrency paths, the legacy-kwargs request builders, and the
Response -> ModelOutput bridge.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import pytest

from sieval.core.models import (
    GenModel,
    InputScoringResult,
    ModelOutput,
    ReasoningOutput,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gen_model():
    return GenModel(model="test-gen", api_key="fake", concurrency_limit=16)


@pytest.fixture
def unlimited_model():
    return GenModel(model="test-unlimited", api_key="fake")


# ===================================================================
# Non-overlapping behaviors
# ===================================================================
class TestModelUnique:
    def test_derived_inherits_client(self, gen_model):
        child = gen_model.with_args(temperature=0.9)
        # Same client object
        assert child._client is gen_model._client

    def test_derived_overrides_kwargs(self, gen_model):
        child = gen_model.with_args(temperature=0.9, top_p=0.8)
        assert child._kwargs == {"temperature": 0.9, "top_p": 0.8}
        # Parent unchanged
        assert gen_model._kwargs == {}

    def test_as_type_is_gone(self, gen_model):
        """RFC #25 dropped cross-kind model conversion."""
        assert not hasattr(gen_model, "as_type")


# ===================================================================
# Response metadata
# ===================================================================
class TestModelOutputResponseMetadata:
    def test_model_output_response_metadata_defaults(self):
        """New response metadata fields default to None."""
        output = ModelOutput(
            model={"model": "test", "api_base": None, "default_params": {}},
            texts=["hello"],
        )
        assert output.response_model is None
        assert output.system_fingerprint is None

    def test_model_output_response_metadata_explicit(self):
        """New response metadata fields can be set explicitly."""
        output = ModelOutput(
            model={"model": "test", "api_base": None, "default_params": {}},
            texts=["hello"],
            response_model="gpt-4o-2024-08-06",
            system_fingerprint="fp_abc123",
        )
        assert output.response_model == "gpt-4o-2024-08-06"
        assert output.system_fingerprint == "fp_abc123"


# ===================================================================
# Quota info
# ===================================================================
class TestQuota:
    def test_total_quota_with_limit(self, gen_model):
        assert gen_model.get_total_quota() == 16

    def test_total_quota_unlimited(self, unlimited_model):
        assert unlimited_model.get_total_quota() == float("inf")

    def test_available_quota_equals_total_initially(self, gen_model):
        assert gen_model.get_available_quota() == 16

    def test_available_quota_unlimited(self, unlimited_model):
        assert unlimited_model.get_available_quota() == float("inf")

    def test_quota_info_structure(self, gen_model):
        info = gen_model.get_quota_info()
        assert info["total"] == 16
        assert info["available"] == 16
        assert info["parent"] is None
        assert info["child"]["total"] == 16

    def test_quota_info_derived(self, gen_model):
        child = gen_model.with_args(concurrency_limit=4)
        info = child.get_quota_info()
        assert info["total"] == 4
        assert info["parent"]["total"] == 16
        assert info["child"]["total"] == 4

    def test_quota_info_no_limiter(self, unlimited_model):
        info = unlimited_model.get_quota_info()
        assert info["total"] == float("inf")
        assert info["parent"] is None
        assert info["child"] is None


# ===================================================================
# agenerate / alogprobs concurrency paths
# (covers the parent_limiter branches around arun)
# ===================================================================
def _build_chat_model_for_path(path):
    from tests.conftest import MockChatModel

    if path == "parent_and_child":
        base = MockChatModel(concurrency_limit=8)
        return base.with_args(concurrency_limit=4)
    if path == "parent_only":
        base = MockChatModel(concurrency_limit=8)
        child = base.with_args()
        child._limiter = None
        child._parent_limiter = base._limiter
        return child
    if path == "own_only":
        return MockChatModel(concurrency_limit=4)
    if path == "no_limiter":
        return MockChatModel()
    raise ValueError(f"Unknown path: {path}")


def _build_gen_model_for_path(path):
    from tests.conftest import MockGenModel

    if path == "parent_and_child":
        base = MockGenModel(concurrency_limit=8)
        return base.with_args(concurrency_limit=4)
    if path == "parent_only":
        base = MockGenModel(concurrency_limit=8)
        child = base.with_args()
        child._limiter = None
        child._parent_limiter = base._limiter
        return child
    if path == "own_only":
        return MockGenModel(concurrency_limit=4)
    if path == "no_limiter":
        return MockGenModel()
    raise ValueError(f"Unknown path: {path}")


def _assert_path_shape(model, path):
    if path == "parent_and_child":
        assert model._parent_limiter is not None
        assert model._limiter is not None
        return
    if path == "parent_only":
        assert model._parent_limiter is not None
        assert model._limiter is None
        return
    if path == "own_only":
        assert model._parent_limiter is None
        assert model._limiter is not None
        return
    if path == "no_limiter":
        assert model._parent_limiter is None
        assert model._limiter is None
        return
    raise ValueError(f"Unknown path: {path}")


class TestConcurrencyPaths:
    """Exercise all four limiter combinations for agenerate and alogprobs."""

    @pytest.mark.anyio
    # (limiter_path, prompt)
    @pytest.mark.parametrize(
        "path,prompt",
        [
            ("parent_and_child", "hello"),
            ("parent_only", "hello"),
            ("own_only", "hello"),
            ("no_limiter", "hello"),
        ],
    )
    async def test_agenerate_paths(self, path, prompt):
        model = _build_chat_model_for_path(path)
        _assert_path_shape(model, path)
        result = await model.agenerate(prompt)
        assert result.texts == ["unknown"]

    @pytest.mark.anyio
    # (limiter_path, prompt)
    @pytest.mark.parametrize(
        "path,prompt",
        [
            ("parent_and_child", "A"),
            ("parent_only", "B"),
            ("own_only", "C"),
            ("no_limiter", "D"),
        ],
    )
    async def test_alogprobs_paths(self, path, prompt):
        model = _build_gen_model_for_path(path)
        _assert_path_shape(model, path)
        result = await model.alogprobs(prompt)
        assert result.texts == [""]
        assert result.logprobs is not None and len(result.logprobs) == 1
        assert result.logprobs_tokens is not None and len(result.logprobs_tokens) == 1

    @pytest.mark.anyio
    async def test_alogprobs_lowers_echo_to_score_input(self):
        """alogprobs args land on the Request the transport receives."""
        from tests.conftest import MockGenModel

        model = MockGenModel()
        await model.alogprobs("A", echo=False, max_tokens=2, logprobs=3)

        req = model._transport.requests[0]
        assert req.score_input is False
        assert req.return_logprobs is True
        assert req.top_k == 3
        assert req.sampling.max_tokens == 2
        assert req.sampling.temperature == 0.0

    @pytest.mark.anyio
    async def test_alogprobs_echo_true_sets_score_input(self):
        from tests.conftest import MockGenModel

        model = MockGenModel()
        await model.alogprobs("A", echo=True)

        req = model._transport.requests[0]
        assert req.score_input is True


# ===================================================================
# Request builders (legacy OpenAI-style kwargs -> IR)
# ===================================================================
class TestBuildGenerateRequest:
    def _model(self, **kwargs):
        return GenModel(model="m", api_key="k", **kwargs)

    def test_sampling_kwargs_map_to_sampling_params(self):
        req = self._model()._build_generate_request(
            "p",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            n=3,
        )
        sp = req.sampling
        assert sp.max_tokens == 64
        assert sp.temperature == 0.7
        assert sp.top_p == 0.9
        assert sp.seed == 42
        assert sp.frequency_penalty == 0.1
        assert sp.presence_penalty == 0.2
        assert sp.n == 3
        assert req.extra_wire_params is None

    def test_max_completion_tokens_aliases_max_tokens(self):
        req = self._model()._build_generate_request("p", max_completion_tokens=32)
        assert req.sampling.max_tokens == 32

    def test_max_tokens_wins_over_alias(self):
        req = self._model()._build_generate_request(
            "p", max_tokens=8, max_completion_tokens=32
        )
        assert req.sampling.max_tokens == 8

    def test_stop_string_becomes_tuple(self):
        req = self._model()._build_generate_request("p", stop="\n\n")
        assert req.sampling.stop == ("\n\n",)

    def test_stop_list_becomes_tuple(self):
        req = self._model()._build_generate_request("p", stop=["\n\n", "Q:"])
        assert req.sampling.stop == ("\n\n", "Q:")

    def test_top_k_kwarg_is_sampling_top_k(self):
        """`top_k` is the vLLM/sglang sampling knob, not the logprobs count."""
        req = self._model()._build_generate_request("p", top_k=40)
        assert req.sampling.top_k_sampling == 40
        assert req.top_k == 0

    def test_logprobs_bool_is_chat_switch(self):
        req = self._model()._build_generate_request("p", logprobs=True)
        assert req.return_logprobs is True
        assert req.top_k == 0

    def test_logprobs_int_is_completions_count(self):
        req = self._model()._build_generate_request("p", logprobs=5)
        assert req.return_logprobs is True
        assert req.top_k == 5

    def test_top_logprobs_sets_top_k(self):
        req = self._model()._build_generate_request("p", logprobs=True, top_logprobs=7)
        assert req.return_logprobs is True
        assert req.top_k == 7

    def test_model_kwargs_merge_with_call_kwargs(self):
        model = self._model(temperature=0.5, max_tokens=10)
        req = model._build_generate_request("p", max_tokens=99)
        assert req.sampling.temperature == 0.5
        assert req.sampling.max_tokens == 99  # call kwargs win

    def test_unknown_kwargs_ride_in_extra_wire_params(self):
        req = self._model()._build_generate_request("p", min_p=0.05, echo=True)
        assert req.extra_wire_params == {"min_p": 0.05, "echo": True}

    def test_stream_defaults_true_and_is_poppable(self):
        assert self._model()._build_generate_request("p").stream is True
        assert self._model()._build_generate_request("p", stream=False).stream is False

    def test_reasoning_effort_maps_to_reasoning_params(self):
        req = self._model()._build_generate_request("p", reasoning_effort="high")
        assert req.reasoning is not None
        assert req.reasoning.effort == "high"

    def test_response_format_and_tools(self):
        fmt = {"type": "json_object"}
        tools = [{"type": "function", "function": {"name": "f"}}]
        req = self._model()._build_generate_request(
            "p", response_format=fmt, tools=tools
        )
        assert req.response_format == fmt
        assert req.tools == tools

    def test_message_input_is_materialized(self):
        from sieval.core.models import ChatModel

        model = ChatModel(model="m", api_key="k")

        def gen():
            yield {"role": "user", "content": "hi"}

        req = model._build_generate_request(gen())
        assert req.input == [{"role": "user", "content": "hi"}]


class TestBuilderValidation:
    def _model(self):
        return GenModel(model="m", api_key="k")

    def test_n_must_be_int(self):
        with pytest.raises(TypeError, match="n must be an int"):
            self._model()._build_generate_request("p", n="3")

    def test_n_bool_rejected(self):
        with pytest.raises(TypeError, match="n must be an int"):
            self._model()._build_generate_request("p", n=True)

    def test_n_below_one_rejected(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            self._model()._build_generate_request("p", n=0)

    def test_stream_must_be_bool(self):
        with pytest.raises(TypeError, match="stream must be a bool"):
            self._model()._build_generate_request("p", stream="yes")

    def test_non_iterable_prompt_rejected(self):
        with pytest.raises(TypeError, match="prompt must be a string"):
            self._model()._build_generate_request(123)

    def test_alogprobs_rejects_n_gt_1(self):
        with pytest.raises(ValueError, match="alogprobs only supports n=1"):
            self._model()._build_logprobs_request(
                "p", max_tokens=1, logprobs=5, score_input=True, temperature=0.0, n=2
            )

    def test_logprobs_request_forces_logprob_fields(self):
        model = GenModel(model="m", api_key="k", logprobs=99)
        req = model._build_logprobs_request(
            "p", max_tokens=1, logprobs=5, score_input=True, temperature=0.0
        )
        # explicit alogprobs args override any _kwargs-borne logprobs config
        assert req.return_logprobs is True
        assert req.top_k == 5
        assert req.score_input is True
        assert req.sampling.max_tokens == 1


# ===================================================================
# Response -> ModelOutput bridge
# ===================================================================
class TestResponseBridge:
    def _bridge(self, resp: Response) -> ModelOutput:
        return GenModel(model="m", api_key="k")._response_to_model_output(resp)

    def test_basic_fields(self):
        out = self._bridge(
            Response(
                texts=("a", "b"),
                finish_reasons=("stop", "length"),
                usage=UsageStats(input_tokens=3, output_tokens=4, total_tokens=7),
                request_params={"max_tokens": 4},
                response_model="served-model",
                system_fingerprint="fp",
            )
        )
        assert isinstance(out, ModelOutput)
        assert out.texts == ["a", "b"]
        assert out.finish_reasons == ["stop", "length"]
        assert out.usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        assert out.request_params == {"max_tokens": 4}
        assert out.response_model == "served-model"
        assert out.system_fingerprint == "fp"
        assert out.model["model"] == "m"

    def test_usage_absent_stays_none(self):
        """Absence != zeros: a zero-filled usage dict would silently corrupt
        the ARC echoed-logprob slice."""
        out = self._bridge(Response(texts=("t",)))
        assert out.usage is None

    def test_input_scoring_is_flattened_ahead_of_sampled_logprobs(self):
        """The legacy echo layout: prompt tokens first, then the completion."""
        resp = Response(
            texts=("t",),
            input_scoring=InputScoringResult(
                token_logprobs=(
                    TokenLogprob(token="Hello", logprob=None),
                    TokenLogprob(token=" world", logprob=-0.5),
                )
            ),
            logprobs=(TokenLogprob(token=" !", logprob=-1.5),),
        )
        out = self._bridge(resp)
        assert out.logprobs_tokens == ["Hello", " world", " !"]
        assert out.logprobs == [None, -0.5, -1.5]

    def test_logprobs_absent_stays_none(self):
        out = self._bridge(Response(texts=("t",)))
        assert out.logprobs_tokens is None
        assert out.logprobs is None
        assert out.top_logprobs is None

    def test_logprobs_present_but_empty_is_empty_list(self):
        """Anomaly detection distinguishes present-but-empty from absent."""
        out = self._bridge(Response(texts=("t",), logprobs=()))
        assert out.logprobs_tokens == []
        assert out.logprobs == []

    def test_top_logprobs_coalesce_duplicates_by_max(self):
        """Distinct token ids can normalize to identical text (sglang Ġ);
        keep the highest logprob, matching legacy CMMLU semantics."""
        resp = Response(
            texts=("t",),
            top_logprobs=(
                (
                    TopKEntry(token=" A", logprob=-2.0, token_id=1),
                    TopKEntry(token=" A", logprob=-0.5, token_id=2),
                    TopKEntry(token=" B", logprob=-1.0, token_id=3),
                ),
            ),
        )
        out = self._bridge(resp)
        assert out.top_logprobs == [{" A": -0.5, " B": -1.0}]

    def test_top_logprobs_empty_collapses_to_none(self):
        out = self._bridge(Response(texts=("t",), top_logprobs=()))
        assert out.top_logprobs is None

    def test_reasoning_text_becomes_reasoning_texts(self):
        out = self._bridge(
            Response(texts=("t",), reasoning=ReasoningOutput(text="thinking..."))
        )
        assert out.reasoning_texts == ["thinking..."]

    def test_empty_reasoning_stays_none(self):
        out = self._bridge(Response(texts=("t",), reasoning=ReasoningOutput(text="")))
        assert out.reasoning_texts is None
