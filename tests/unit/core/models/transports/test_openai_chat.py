"""Unit tests for OpenAIChatTransport (lower/lift + InputScoring rejection).

Streaming accumulation coverage moved here from the legacy ChatModel tests
(tests/unit/core/models/test_chat_model.py) when RFC #25 relocated the wire
logic into the transport: everything is driven through
``Transport.arun(Request(...))`` with the OpenAI client mocked — no real
API traffic.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.exceptions import CapabilityError
from sieval.core.models.ir import (
    ReasoningParams,
    Request,
    SamplingParams,
    TokenLogprob,
    UsageStats,
)
from sieval.core.models.transports.openai_chat import OpenAIChatTransport


def _make_transport(response: object) -> tuple[OpenAIChatTransport, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=response)
    client.chat.completions.create = create
    return OpenAIChatTransport(client=client, model="m"), create


def _make_response(
    *,
    content="",
    finish_reason="stop",
    reasoning=None,
    reasoning_content=None,
    logprob_content=None,
    tool_calls=None,
    prompt_tokens=1,
    completion_tokens=1,
):
    resp = MagicMock()
    resp.model = None
    resp.system_fingerprint = None
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = finish_reason
    msg = MagicMock()
    msg.content = content
    msg.reasoning = reasoning
    msg.reasoning_content = reasoning_content
    msg.tool_calls = tool_calls
    choice.message = msg
    if logprob_content is None:
        choice.logprobs = None
    else:
        lp = MagicMock()
        lp.content = logprob_content
        choice.logprobs = lp
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp.usage = usage
    return resp


def _lp_item(token, logprob, top=None):
    item = MagicMock()
    item.token = token
    item.logprob = logprob
    item.top_logprobs = top or []
    return item


def _top(token, logprob):
    t = MagicMock()
    t.token = token
    t.logprob = logprob
    return t


# ---------------------------------------------------------------------------
# Async streaming helpers (ported from the legacy ChatModel tests)
# ---------------------------------------------------------------------------
class _AsyncIterator:
    """Wraps a list into an async iterator."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as e:
            raise StopAsyncIteration from e


def _make_chunk(
    index=0,
    content=None,
    finish_reason="",
    usage=None,
    reasoning=None,
    reasoning_content=None,
    logprob_content=None,
    model=None,
    system_fingerprint=None,
):
    """Build a minimal streaming chunk carrying one choice."""
    chunk = MagicMock()
    chunk.usage = usage
    chunk.model = model
    chunk.system_fingerprint = system_fingerprint
    choice = MagicMock()
    choice.index = index
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = content
    delta.reasoning = reasoning
    delta.reasoning_content = reasoning_content
    choice.delta = delta
    if logprob_content is None:
        choice.logprobs = None
    else:
        lp = MagicMock()
        lp.content = logprob_content
        choice.logprobs = lp
    chunk.choices = [choice]
    return chunk


def _make_multi_chunk(pairs, finish_reason=""):
    """One chunk carrying several choices: ``pairs`` of (index, content)."""
    chunk = MagicMock()
    chunk.usage = None
    chunk.model = None
    chunk.system_fingerprint = None
    chunk.choices = [
        _make_chunk(index=index, content=content, finish_reason=finish_reason).choices[
            0
        ]
        for index, content in pairs
    ]
    return chunk


def _make_usage_chunk(prompt_tokens=10, completion_tokens=5):
    """Final chunk carrying usage but no choices."""
    chunk = MagicMock()
    chunk.choices = []
    chunk.model = None
    chunk.system_fingerprint = None
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    chunk.usage = usage
    return chunk


class TestLowerRejections:
    def test_score_input_raises_capability_error(self):
        t, _ = _make_transport(_make_response())
        with pytest.raises(CapabilityError, match="InputScoring"):
            t._lower(Request(input="hi", score_input=True))

    def test_session_id_raises_capability_error(self):
        t, _ = _make_transport(_make_response())
        with pytest.raises(CapabilityError, match="session_id"):
            t._lower(Request(input="hi", session_id="resp_123"))


class TestLower:
    def test_str_input_wrapped_as_user_message(self):
        t, _ = _make_transport(_make_response())
        messages, _ = t._lower(Request(input="hi"))
        assert messages == [{"role": "user", "content": "hi"}]

    def test_message_list_passthrough(self):
        t, _ = _make_transport(_make_response())
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        messages, _ = t._lower(Request(input=msgs))
        assert messages == msgs

    def test_return_logprobs_and_top_k(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(Request(input="hi", return_logprobs=True, top_k=5))
        assert params["logprobs"] is True
        assert params["top_logprobs"] == 5

    def test_top_k_zero_omits_top_logprobs(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(Request(input="hi", return_logprobs=True, top_k=0))
        assert params["logprobs"] is True
        assert "top_logprobs" not in params

    def test_reasoning_effort_passthrough(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(
            Request(input="hi", reasoning=ReasoningParams(effort="high"))
        )
        assert params["reasoning_effort"] == "high"

    def test_response_format_and_tools(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(
            Request(
                input="hi",
                response_format={"type": "json_object"},
                tools=[{"type": "function"}],
            )
        )
        assert params["response_format"] == {"type": "json_object"}
        assert params["tools"] == [{"type": "function"}]

    def test_sampling_maps(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(
            Request(input="hi", sampling=SamplingParams(max_tokens=4, temperature=0.3))
        )
        assert params["max_tokens"] == 4
        assert params["temperature"] == 0.3


class TestLowerStream:
    """Wire scheduling: Request.stream lowering + stream_options injection."""

    def test_stream_true_sets_stream_and_injects_stream_options(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(Request(input="hi", stream=True))
        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}

    def test_explicit_stream_options_wins_over_injected_default(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(
            Request(
                input="hi",
                stream=True,
                extra_wire_params={"stream_options": {"include_usage": False}},
            )
        )
        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": False}

    def test_stream_none_defaults_to_single_shot(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(Request(input="hi", stream=None))
        assert params["stream"] is False
        assert "stream_options" not in params

    def test_stream_false_no_stream_options(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(Request(input="hi", stream=False))
        assert params["stream"] is False
        assert "stream_options" not in params


class TestLift:
    @pytest.mark.anyio
    async def test_text_and_usage(self):
        t, _ = _make_transport(
            _make_response(content="hello", prompt_tokens=3, completion_tokens=2)
        )
        out = await t.arun(Request(input="hi"))
        assert out.texts == ("hello",)
        assert out.usage == UsageStats(input_tokens=3, output_tokens=2, total_tokens=5)

    @pytest.mark.anyio
    async def test_reasoning_extracted(self):
        t, _ = _make_transport(_make_response(content="a", reasoning="because"))
        out = await t.arun(Request(input="hi"))
        assert out.reasoning is not None
        assert out.reasoning.text == "because"

    @pytest.mark.anyio
    async def test_reasoning_content_fallback(self):
        t, _ = _make_transport(
            _make_response(content="a", reasoning=None, reasoning_content="fallback")
        )
        out = await t.arun(Request(input="hi"))
        assert out.reasoning is not None
        assert out.reasoning.text == "fallback"

    @pytest.mark.anyio
    async def test_reasoning_takes_priority_over_reasoning_content(self):
        t, _ = _make_transport(
            _make_response(
                content="a", reasoning="primary", reasoning_content="secondary"
            )
        )
        out = await t.arun(Request(input="hi"))
        assert out.reasoning is not None
        assert out.reasoning.text == "primary"

    @pytest.mark.anyio
    async def test_no_reasoning_channel_is_none(self):
        t, _ = _make_transport(_make_response(content="a"))
        out = await t.arun(Request(input="hi"))
        assert out.reasoning is None

    @pytest.mark.anyio
    async def test_logprobs_and_top_logprobs_mapped(self):
        content = [
            _lp_item("A", -0.1, [_top("A", -0.1), _top("B", -2.0)]),
        ]
        t, _ = _make_transport(_make_response(content="A", logprob_content=content))
        out = await t.arun(Request(input="hi", return_logprobs=True, top_k=2))
        assert out.logprobs is not None
        assert out.logprobs[0].token == "A"
        assert out.logprobs[0].token_id is None
        assert out.top_logprobs is not None
        assert out.top_logprobs[0][1].token == "B"
        assert out.top_logprobs[0][1].logprob == -2.0

    @pytest.mark.anyio
    async def test_logprobs_object_with_empty_content_is_empty_tuple(self):
        t, _ = _make_transport(_make_response(content="x", logprob_content=[]))
        out = await t.arun(Request(input="hi", return_logprobs=True))
        assert out.logprobs == ()
        assert out.top_logprobs == ()

    @pytest.mark.anyio
    async def test_absent_logprobs_object_is_none(self):
        t, _ = _make_transport(_make_response(content="x"))
        out = await t.arun(Request(input="hi", return_logprobs=True))
        assert out.logprobs is None
        assert out.top_logprobs is None

    @pytest.mark.anyio
    async def test_tool_calls_captured(self):
        tc = MagicMock()
        tc.model_dump.return_value = {"id": "call_1", "type": "function"}
        t, _ = _make_transport(_make_response(content="", tool_calls=[tc]))
        out = await t.arun(Request(input="hi", tools=[{"type": "function"}]))
        assert out.tool_calls is not None
        assert out.tool_calls[0] == {"id": "call_1", "type": "function"}

    @pytest.mark.anyio
    async def test_usage_absent_is_none(self):
        resp = _make_response(content="hi")
        resp.usage = None
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="hi"))
        assert out.usage is None

    @pytest.mark.anyio
    async def test_response_model_and_fingerprint_captured(self):
        resp = _make_response(content="ok")
        resp.model = "served-model-v2"
        resp.system_fingerprint = "fp_xyz789"
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="hi"))
        assert out.response_model == "served-model-v2"
        assert out.system_fingerprint == "fp_xyz789"

    @pytest.mark.anyio
    async def test_request_params_is_lowered_params_dict(self):
        t, create = _make_transport(_make_response(content="ok"))
        out = await t.arun(
            Request(input="hi", sampling=SamplingParams(max_tokens=4, temperature=0.3))
        )
        assert out.request_params == {
            "max_tokens": 4,
            "temperature": 0.3,
            "stream": False,
        }
        # model/messages ride separately on the wire, never in request_params.
        call_kwargs = dict(create.call_args.kwargs)
        assert call_kwargs.pop("model") == "m"
        assert call_kwargs.pop("messages") == [{"role": "user", "content": "hi"}]
        assert out.request_params == call_kwargs

    @pytest.mark.anyio
    async def test_missing_finish_reason_defaults_to_empty(self):
        t, _ = _make_transport(_make_response(content="done", finish_reason=None))
        out = await t.arun(Request(input="hi"))
        assert out.texts == ("done",)
        assert out.finish_reasons == ("",)

    @pytest.mark.anyio
    async def test_content_none_stays_empty(self):
        t, _ = _make_transport(_make_response(content=None))
        out = await t.arun(Request(input="hi"))
        assert out.texts == ("",)

    @pytest.mark.anyio
    async def test_out_of_range_choice_index_ignored(self):
        resp = _make_response(content="ignored")
        resp.choices[0].index = 5
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="hi"))
        assert out.texts == ("",)
        assert out.finish_reasons == ("",)


class TestLiftStream:
    """Streaming accumulation, driven through arun(Request(stream=True))."""

    @pytest.mark.anyio
    async def test_delta_content_stitched(self):
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world", finish_reason="stop"),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.texts == ("Hello world",)
        assert resp.finish_reasons == ("stop",)

    @pytest.mark.anyio
    async def test_n_gt_1_routes_by_choice_index(self):
        chunks = [
            _make_multi_chunk([(0, "A1"), (1, "B1")]),
            _make_multi_chunk([(0, "A2"), (1, "B2")], finish_reason="stop"),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(
            Request(input="hi", sampling=SamplingParams(n=2), stream=True)
        )
        assert resp.texts == ("A1A2", "B1B2")
        assert resp.finish_reasons == ("stop", "stop")

    @pytest.mark.anyio
    async def test_out_of_range_choice_index_ignored(self):
        chunks = [
            _make_chunk(index=5, content="ignored", finish_reason="stop"),
            _make_chunk(index=0, content="kept", finish_reason="stop"),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.texts == ("kept",)
        assert resp.finish_reasons == ("stop",)

    @pytest.mark.anyio
    async def test_reasoning_accumulated_from_delta_reasoning(self):
        chunks = [
            _make_chunk(content="a", reasoning="think1"),
            _make_chunk(content="b", reasoning="think2", finish_reason="stop"),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.texts == ("ab",)
        assert resp.reasoning is not None
        assert resp.reasoning.text == "think1think2"

    @pytest.mark.anyio
    async def test_reasoning_content_fallback(self):
        chunks = [
            _make_chunk(content="a", reasoning_content="fallback", finish_reason="stop")
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.reasoning is not None
        assert resp.reasoning.text == "fallback"

    @pytest.mark.anyio
    async def test_reasoning_takes_priority_over_reasoning_content(self):
        chunks = [
            _make_chunk(
                content="a",
                reasoning="primary",
                reasoning_content="secondary",
                finish_reason="stop",
            )
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.reasoning is not None
        assert resp.reasoning.text == "primary"

    @pytest.mark.anyio
    async def test_no_reasoning_deltas_channel_is_none(self):
        chunks = [_make_chunk(content="a", finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.reasoning is None

    @pytest.mark.anyio
    async def test_usage_from_final_chunk(self):
        chunks = [
            _make_chunk(content="ok", finish_reason="stop"),
            _make_usage_chunk(8, 2),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.usage == UsageStats(
            input_tokens=8, output_tokens=2, total_tokens=10
        )

    @pytest.mark.anyio
    async def test_no_usage_chunk_usage_is_none(self):
        chunks = [_make_chunk(content="hi", finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.usage is None

    @pytest.mark.anyio
    async def test_response_metadata_captured_from_first_chunk(self):
        chunks = [
            _make_chunk(content="a", model="model-v1", system_fingerprint="fp_first"),
            _make_chunk(
                content="b",
                finish_reason="stop",
                model="model-v2",
                system_fingerprint="fp_second",
            ),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.response_model == "model-v1"
        assert resp.system_fingerprint == "fp_first"

    @pytest.mark.anyio
    async def test_missing_response_metadata_is_none(self):
        chunks = [_make_chunk(content="ok", finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.response_model is None
        assert resp.system_fingerprint is None

    @pytest.mark.anyio
    async def test_streaming_logprobs_collected_for_choice_zero(self):
        chunks = [
            _make_chunk(
                content="A",
                logprob_content=[
                    _lp_item("A", -0.1, [_top("A", -0.1), _top("B", -2.0)])
                ],
            ),
            _make_chunk(
                content="B",
                finish_reason="stop",
                logprob_content=[_lp_item("B", -0.5)],
            ),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(
            Request(input="hi", return_logprobs=True, top_k=2, stream=True)
        )
        assert resp.logprobs == (
            TokenLogprob(token="A", logprob=-0.1),
            TokenLogprob(token="B", logprob=-0.5),
        )
        assert resp.top_logprobs is not None
        assert resp.top_logprobs[0][1].token == "B"
        assert resp.top_logprobs[0][1].logprob == -2.0
        assert resp.top_logprobs[1] == ()

    @pytest.mark.anyio
    async def test_logprobs_only_collected_from_choice_zero(self):
        chunks = [
            _make_chunk(index=1, content="b", logprob_content=[_lp_item("X", -9.0)]),
            _make_chunk(
                index=0,
                content="a",
                finish_reason="stop",
                logprob_content=[_lp_item("A", -0.1)],
            ),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(
            Request(
                input="hi",
                sampling=SamplingParams(n=2),
                return_logprobs=True,
                stream=True,
            )
        )
        assert resp.logprobs == (TokenLogprob(token="A", logprob=-0.1),)

    @pytest.mark.anyio
    async def test_no_logprobs_object_on_any_chunk_is_none(self):
        chunks = [_make_chunk(content="x", finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", return_logprobs=True, stream=True))
        assert resp.logprobs is None
        assert resp.top_logprobs is None

    @pytest.mark.anyio
    async def test_logprobs_object_with_empty_content_is_empty_tuple(self):
        chunks = [_make_chunk(content="x", finish_reason="stop", logprob_content=[])]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", return_logprobs=True, stream=True))
        assert resp.logprobs == ()
        assert resp.top_logprobs == ()

    @pytest.mark.anyio
    async def test_delta_none_accumulates_nothing(self):
        chunk = _make_chunk(finish_reason="stop")
        chunk.choices[0].delta = None
        t, _ = _make_transport(_AsyncIterator([chunk]))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.texts == ("",)

    @pytest.mark.anyio
    async def test_delta_content_none_stays_empty(self):
        chunks = [_make_chunk(content=None, finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.texts == ("",)

    @pytest.mark.anyio
    async def test_request_params_include_stream_and_stream_options(self):
        chunks = [_make_chunk(content="x", finish_reason="stop")]
        t, create = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="hi", stream=True))
        assert resp.request_params == {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert create.call_args.kwargs["stream"] is True
        assert create.call_args.kwargs["stream_options"] == {"include_usage": True}


class TestLowerBranches:
    def test_all_sampling_params_map(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(
            Request(
                input="hi",
                sampling=SamplingParams(
                    top_p=0.8,
                    stop=("X",),
                    seed=7,
                    frequency_penalty=0.1,
                    presence_penalty=0.2,
                    n=2,
                ),
            )
        )
        assert params["top_p"] == 0.8
        assert params["stop"] == ["X"]
        assert params["seed"] == 7
        assert params["frequency_penalty"] == 0.1
        assert params["presence_penalty"] == 0.2
        assert params["n"] == 2

    def test_extra_wire_params_passthrough(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(Request(input="hi", extra_wire_params={"user": "u1"}))
        assert params["user"] == "u1"

    def test_explicit_ir_field_wins_over_extra_wire_params(self):
        t, _ = _make_transport(_make_response())
        _, params = t._lower(
            Request(
                input="hi",
                sampling=SamplingParams(max_tokens=4),
                extra_wire_params={"max_tokens": 99},
            )
        )
        assert params["max_tokens"] == 4


class TestToolCallConversion:
    def test_dict_tool_call_passthrough(self):
        from sieval.core.models.transports.openai_chat import _tool_call_to_dict

        assert _tool_call_to_dict({"id": "x"}) == {"id": "x"}

    def test_plain_object_tool_call_via_vars(self):
        from sieval.core.models.transports.openai_chat import _tool_call_to_dict

        class _Obj:
            def __init__(self):
                self.id = "y"

        assert _tool_call_to_dict(_Obj()) == {"id": "y"}
