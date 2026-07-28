"""Unit tests for the provider-agnostic Model IR.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import dataclasses

import pytest

from sieval.core.models.ir import (
    Citation,
    GroundingChunk,
    GroundingMetadata,
    InputScoringResult,
    ReasoningOutput,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    ServerToolSpec,
    ServerToolUse,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.utils.serialization import (
    dict_to_obj,
    global_type_registry,
    obj_to_dict,
)


class TestRequestConstruction:
    def test_minimal_completion_request(self):
        req = Request(input="hello")
        assert req.input == "hello"
        assert req.sampling is None
        assert req.return_logprobs is False
        assert req.top_k == 0
        assert req.score_input is False
        assert req.session_id is None
        assert req.extra_wire_params is None

    def test_chat_request_input_is_message_list(self):
        req = Request(input=[{"role": "user", "content": "hi"}])
        assert isinstance(req.input, list)
        assert req.input[0]["role"] == "user"

    def test_extra_wire_params_passthrough_field(self):
        req = Request(input="x", extra_wire_params={"logit_bias": {"1": -100}})
        assert req.extra_wire_params == {"logit_bias": {"1": -100}}

    def test_stream_defaults_to_transport_choice(self):
        # None → the transport picks (single-shot); pure scheduling knob.
        assert Request(input="x").stream is None
        assert Request(input="x", stream=True).stream is True

    def test_sampling_params_defaults(self):
        sp = SamplingParams()
        assert sp.temperature is None
        assert sp.n == 1
        assert sp.stop is None

    def test_reasoning_params_opaque_roundtrip_is_str_or_none(self):
        rp = ReasoningParams(opaque_roundtrip="sig-abc")
        assert rp.opaque_roundtrip == "sig-abc"
        assert ReasoningParams().opaque_roundtrip is None

    def test_server_tool_spec(self):
        spec = ServerToolSpec(type="web_search", config={"max_uses": 3})
        req = Request(input="x", server_tools=(spec,))
        assert req.server_tools is not None
        assert req.server_tools[0].type == "web_search"


class TestResponseConstruction:
    def test_minimal_response(self):
        resp = Response(texts=("hi",))
        assert resp.texts == ("hi",)
        assert resp.reasoning is None
        assert resp.logprobs is None
        assert resp.session_id is None
        assert resp.finish_reasons is None

    def test_provenance_fields_default_none(self):
        resp = Response(texts=("hi",))
        assert resp.request_params is None
        assert resp.response_model is None
        assert resp.system_fingerprint is None

    def test_usage_defaults_to_none(self):
        # Absence != zeros: a server that reported no usage must not be
        # recorded as zero tokens (the bridge relies on this distinction).
        resp = Response(texts=("hi",))
        assert resp.usage is None

    def test_usage_stats_fields_default_zero(self):
        usage = UsageStats()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.reasoning_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    def test_grounding_metadata_rendered_content_optional(self):
        gm = GroundingMetadata(chunks=(GroundingChunk(uri="http://x"),))
        assert gm.rendered_content is None
        assert dataclasses.fields(GroundingMetadata)  # field exists
        gm2 = GroundingMetadata(chunks=(), rendered_content="<div/>")
        assert gm2.rendered_content == "<div/>"


class TestImmutability:
    @pytest.mark.parametrize(
        ("obj", "field_name", "value"),
        [
            (Request(input="x"), "input", "y"),
            (Response(texts=("a",)), "texts", ("b",)),
            (SamplingParams(), "temperature", 0.5),
            (ReasoningParams(), "effort", "high"),
            (UsageStats(), "input_tokens", 5),
            (TokenLogprob(token_id=1, token="a", logprob=-0.1), "logprob", 0.0),
        ],
    )
    def test_frozen_assignment_raises(self, obj, field_name, value):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field_name, value)


class TestSerializationRoundTrip:
    """Response is the persisted record schema; nested records must rehydrate
    back into typed objects, not plain dicts."""

    def test_response_round_trips_to_typed_object(self):
        data = obj_to_dict(Response(texts=("hi", "there")), add_type=True)
        back = dict_to_obj(data, global_type_registry)
        assert isinstance(back, Response)
        assert back.texts == ("hi", "there")

    def test_nested_reasoning_and_usage_rehydrate_as_records(self):
        resp = Response(
            texts=("answer",),
            reasoning=ReasoningOutput(text="think", thinking_tokens=12),
            usage=UsageStats(input_tokens=3, output_tokens=4, total_tokens=7),
        )
        back = dict_to_obj(obj_to_dict(resp, add_type=True), global_type_registry)
        assert isinstance(back.reasoning, ReasoningOutput)
        assert back.reasoning.text == "think"
        assert back.reasoning.thinking_tokens == 12
        assert isinstance(back.usage, UsageStats)
        assert back.usage.total_tokens == 7

    def test_tuple_of_token_logprobs_rehydrates_element_type(self):
        # The gap-4 case: a tuple of records only round-trips when the element
        # type is itself @sieval_record.
        resp = Response(
            texts=("x",),
            logprobs=(
                TokenLogprob(token_id=10, token=" A", logprob=-0.5),
                TokenLogprob(token_id=None, token="B", logprob=-1.5),
            ),
        )
        back = dict_to_obj(obj_to_dict(resp, add_type=True), global_type_registry)
        assert isinstance(back.logprobs, tuple)
        assert all(isinstance(t, TokenLogprob) for t in back.logprobs)
        assert back.logprobs[0].token_id == 10
        assert back.logprobs[0].token == " A"
        assert back.logprobs[1].token_id is None

    def test_nested_tuple_of_topk_entries_rehydrates(self):
        resp = Response(
            texts=("x",),
            top_logprobs=(
                (
                    TopKEntry(token_id=1, token="A", logprob=-0.1),
                    TopKEntry(token_id=2, token="B", logprob=-2.0),
                ),
            ),
        )
        back = dict_to_obj(obj_to_dict(resp, add_type=True), global_type_registry)
        assert isinstance(back.top_logprobs[0][0], TopKEntry)
        assert back.top_logprobs[0][1].token == "B"

    def test_input_scoring_and_grounding_and_tool_use_rehydrate(self):
        resp = Response(
            texts=("x",),
            input_scoring=InputScoringResult(
                token_logprobs=(TokenLogprob(token_id=5, token="q", logprob=-0.2),),
                byte_count=3,
                char_count=1,
            ),
            citations=(Citation(url="http://a", title="A"),),
            grounding=GroundingMetadata(
                chunks=(GroundingChunk(uri="http://c", title="C"),),
                rendered_content="<w/>",
            ),
            server_tool_uses=(
                ServerToolUse(
                    tool_type="web_search",
                    tool_use_id="t1",
                    input={"q": "x"},
                    result={"ok": True},
                ),
            ),
        )
        back = dict_to_obj(obj_to_dict(resp, add_type=True), global_type_registry)
        assert isinstance(back.input_scoring, InputScoringResult)
        assert back.input_scoring.byte_count == 3
        assert isinstance(back.input_scoring.token_logprobs[0], TokenLogprob)
        assert isinstance(back.citations[0], Citation)
        assert isinstance(back.grounding, GroundingMetadata)
        # Google ToS: rendered_content must survive the round-trip.
        assert back.grounding.rendered_content == "<w/>"
        assert isinstance(back.grounding.chunks[0], GroundingChunk)
        assert isinstance(back.server_tool_uses[0], ServerToolUse)
        assert back.server_tool_uses[0].result == {"ok": True}
