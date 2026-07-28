"""Unit tests for OpenAICompletionsTransport (echo/InputScoring split).

Streaming accumulation and ``_completion_top_logprobs`` coverage moved here
from the legacy GenModel tests (tests/unit/core/models/test_gen_model.py) when
RFC #25 relocated the wire logic into the transport: everything is driven
through ``Transport.arun(Request(...))`` with the OpenAI client mocked — no
real API traffic.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.ir import Request, SamplingParams, TokenLogprob, UsageStats
from sieval.core.models.transports.openai_completions import (
    OpenAICompletionsTransport,
    _completion_top_logprobs,
)


def _make_transport(response: object) -> tuple[OpenAICompletionsTransport, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=response)
    client.completions.create = create
    return OpenAICompletionsTransport(client=client, model="m"), create


def _make_response(
    *,
    text="",
    tokens=None,
    token_logprobs=None,
    top_logprobs=None,
    prompt_tokens=0,
    completion_tokens=0,
    finish_reason="stop",
):
    resp = MagicMock()
    resp.model = None
    resp.system_fingerprint = None
    choice = MagicMock()
    choice.index = 0
    choice.text = text
    choice.finish_reason = finish_reason
    if tokens is None and token_logprobs is None:
        choice.logprobs = None
    else:
        lp = MagicMock()
        lp.tokens = tokens or []
        lp.token_logprobs = token_logprobs or []
        lp.top_logprobs = top_logprobs
        choice.logprobs = lp
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Async streaming helpers (ported from the legacy GenModel tests)
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
    text="",
    finish_reason="",
    tokens=None,
    token_logprobs=None,
    top_logprobs=None,
    model=None,
    system_fingerprint=None,
):
    """Build a minimal completions streaming chunk carrying one choice."""
    chunk = MagicMock()
    chunk.usage = None
    chunk.model = model
    chunk.system_fingerprint = system_fingerprint
    choice = MagicMock()
    choice.index = index
    choice.text = text
    choice.finish_reason = finish_reason
    if tokens is None and token_logprobs is None and top_logprobs is None:
        choice.logprobs = None
    else:
        lp = MagicMock()
        lp.tokens = tokens or []
        lp.token_logprobs = token_logprobs or []
        lp.top_logprobs = top_logprobs
        choice.logprobs = lp
    chunk.choices = [choice]
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


class TestLower:
    def test_score_input_true_sets_echo(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(Request(input="hi", score_input=True))
        assert params["echo"] is True

    def test_score_input_false_omits_echo(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(Request(input="hi", return_logprobs=True, top_k=3))
        assert "echo" not in params

    def test_logprobs_count_from_top_k(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(Request(input="hi", return_logprobs=True, top_k=5))
        assert params["logprobs"] == 5

    def test_sampling_maps_through(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(
            Request(input="hi", sampling=SamplingParams(max_tokens=8, temperature=0.7))
        )
        assert params["max_tokens"] == 8
        assert params["temperature"] == 0.7

    def test_all_sampling_params_map(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(
            Request(
                input="hi",
                sampling=SamplingParams(
                    top_p=0.8,
                    stop=("X",),
                    seed=42,
                    frequency_penalty=0.3,
                    presence_penalty=0.4,
                    n=2,
                ),
            )
        )
        assert params["top_p"] == 0.8
        assert params["stop"] == ["X"]
        assert params["seed"] == 42
        assert params["frequency_penalty"] == 0.3
        assert params["presence_penalty"] == 0.4
        assert params["n"] == 2

    def test_extra_wire_params_passthrough(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(Request(input="hi", extra_wire_params={"logit_bias": {}}))
        assert params["logit_bias"] == {}

    def test_explicit_ir_field_wins_over_extra_wire_params(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(
            Request(
                input="hi",
                sampling=SamplingParams(max_tokens=8),
                extra_wire_params={"max_tokens": 99},
            )
        )
        assert params["max_tokens"] == 8

    def test_non_str_input_raises(self):
        t, _ = _make_transport(_make_response())
        with pytest.raises(TypeError, match="str input"):
            t._lower(Request(input=[{"role": "user", "content": "x"}]))


class TestLowerStream:
    """Wire scheduling: Request.stream lowering + stream_options injection."""

    def test_stream_true_sets_stream_and_injects_stream_options(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(Request(input="hi", stream=True))
        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}

    def test_explicit_stream_options_wins_over_injected_default(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(
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
        params = t._lower(Request(input="hi", stream=None))
        assert params["stream"] is False
        assert "stream_options" not in params

    def test_stream_false_no_stream_options(self):
        t, _ = _make_transport(_make_response())
        params = t._lower(Request(input="hi", stream=False))
        assert params["stream"] is False
        assert "stream_options" not in params


class TestLiftEchoSplit:
    @pytest.mark.anyio
    async def test_prompt_completion_boundary_split(self):
        # 2 prompt tokens echoed + 1 completion token.
        resp = _make_response(
            text=" g",
            tokens=["a", "b", " g"],
            token_logprobs=[None, -1.0, -0.5],
            prompt_tokens=2,
            completion_tokens=1,
        )
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="ab", score_input=True, top_k=0))
        assert out.input_scoring is not None
        assert len(out.input_scoring.token_logprobs) == 2  # == prompt_tokens
        assert out.input_scoring.token_logprobs[0].logprob is None
        assert out.logprobs is not None
        assert len(out.logprobs) == 1
        assert out.logprobs[0].token == " g"
        assert out.logprobs[0].logprob == -0.5

    @pytest.mark.anyio
    async def test_missing_usage_boundary_zero(self):
        # No usage → the echo boundary defaults to 0: input_scoring stays
        # empty and every token lands in the sampled-completion channel.
        resp = _make_response(
            text=" g",
            tokens=["a", " g"],
            token_logprobs=[None, -0.5],
        )
        resp.usage = None
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="a", score_input=True, top_k=0))
        assert out.input_scoring is not None
        assert out.input_scoring.token_logprobs == ()
        assert out.logprobs is not None
        assert len(out.logprobs) == 2

    @pytest.mark.anyio
    async def test_no_score_input_all_tokens_in_logprobs(self):
        resp = _make_response(
            text="xy",
            tokens=["x", "y"],
            token_logprobs=[-0.1, -0.2],
            prompt_tokens=1,
            completion_tokens=2,
        )
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p", return_logprobs=True, top_k=0))
        assert out.input_scoring is None
        assert out.logprobs is not None
        assert len(out.logprobs) == 2

    @pytest.mark.anyio
    async def test_token_id_is_none_for_completions(self):
        resp = _make_response(
            text="x",
            tokens=["x"],
            token_logprobs=[-0.1],
            prompt_tokens=0,
            completion_tokens=1,
        )
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p", return_logprobs=True))
        assert out.logprobs is not None
        assert out.logprobs[0].token_id is None

    @pytest.mark.anyio
    async def test_no_logprobs_response(self):
        resp = _make_response(text="hi", prompt_tokens=1, completion_tokens=1)
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p"))
        assert out.logprobs is None
        assert out.top_logprobs is None
        assert out.texts == ("hi",)

    @pytest.mark.anyio
    async def test_usage_absent_is_none(self):
        resp = _make_response(text="hi")
        resp.usage = None
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p"))
        assert out.usage is None


class TestLift:
    """Non-streaming lift of the remaining Response fields."""

    @pytest.mark.anyio
    async def test_usage_present_maps_to_usage_stats(self):
        resp = _make_response(text="ok", prompt_tokens=6, completion_tokens=2)
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p"))
        assert out.usage == UsageStats(input_tokens=6, output_tokens=2, total_tokens=8)

    @pytest.mark.anyio
    async def test_response_model_and_fingerprint_captured(self):
        resp = _make_response(text="ok")
        resp.model = "served-model-v2"
        resp.system_fingerprint = "fp_xyz789"
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p"))
        assert out.response_model == "served-model-v2"
        assert out.system_fingerprint == "fp_xyz789"

    @pytest.mark.anyio
    async def test_request_params_is_lowered_params_dict(self):
        t, create = _make_transport(_make_response(text="ok"))
        out = await t.arun(
            Request(input="p", sampling=SamplingParams(max_tokens=8, temperature=0.7))
        )
        assert out.request_params == {
            "max_tokens": 8,
            "temperature": 0.7,
            "stream": False,
        }
        # model/prompt ride separately on the wire, never in request_params.
        call_kwargs = dict(create.call_args.kwargs)
        assert call_kwargs.pop("model") == "m"
        assert call_kwargs.pop("prompt") == "p"
        assert out.request_params == call_kwargs

    @pytest.mark.anyio
    async def test_logprobs_object_with_empty_channels_is_empty_tuple(self):
        resp = _make_response(text="x", tokens=[], token_logprobs=[])
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p", return_logprobs=True))
        assert out.logprobs == ()
        assert out.top_logprobs == ()

    @pytest.mark.anyio
    async def test_top_logprobs_sanitized_per_position(self):
        resp = _make_response(
            text="A",
            tokens=["A"],
            token_logprobs=[-0.1],
            top_logprobs=[{"A": -0.1, "B": -2.0}],
            prompt_tokens=0,
            completion_tokens=1,
        )
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p", return_logprobs=True, top_k=2))
        assert out.top_logprobs is not None
        assert len(out.top_logprobs) == 1
        assert {(e.token, e.logprob) for e in out.top_logprobs[0]} == {
            ("A", -0.1),
            ("B", -2.0),
        }

    @pytest.mark.anyio
    async def test_out_of_range_choice_index_ignored(self):
        resp = _make_response(text="ignored")
        resp.choices[0].index = 5
        t, _ = _make_transport(resp)
        out = await t.arun(Request(input="p"))
        assert out.texts == ("",)
        assert out.finish_reasons == ("",)


class TestLiftStream:
    """Streaming accumulation, driven through arun(Request(stream=True))."""

    @pytest.mark.anyio
    async def test_text_concatenated_across_chunks(self):
        chunks = [
            _make_chunk(text="Hello"),
            _make_chunk(text=" world", finish_reason="stop"),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", stream=True))
        assert resp.texts == ("Hello world",)
        assert resp.finish_reasons == ("stop",)

    @pytest.mark.anyio
    async def test_n_gt_1_routes_by_choice_index(self):
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.model = None
        chunk1.system_fingerprint = None
        chunk1.choices = [
            _make_chunk(index=0, text="A1").choices[0],
            _make_chunk(index=1, text="B1").choices[0],
        ]
        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.model = None
        chunk2.system_fingerprint = None
        chunk2.choices = [
            _make_chunk(index=0, text="A2", finish_reason="stop").choices[0],
            _make_chunk(index=1, text="B2", finish_reason="stop").choices[0],
        ]
        t, _ = _make_transport(_AsyncIterator([chunk1, chunk2]))
        resp = await t.arun(
            Request(input="p", sampling=SamplingParams(n=2), stream=True)
        )
        assert resp.texts == ("A1A2", "B1B2")
        assert resp.finish_reasons == ("stop", "stop")

    @pytest.mark.anyio
    async def test_out_of_range_choice_index_ignored(self):
        chunks = [
            _make_chunk(index=5, text="ignored", finish_reason="stop"),
            _make_chunk(index=0, text="kept", finish_reason="stop"),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", stream=True))
        assert resp.texts == ("kept",)

    @pytest.mark.anyio
    async def test_logprobs_accumulated_for_choice_zero(self):
        chunks = [
            _make_chunk(
                text="A",
                tokens=["A"],
                token_logprobs=[-0.1],
                top_logprobs=[{"A": -0.1, "B": -2.0}],
            ),
            _make_chunk(
                text="B",
                finish_reason="stop",
                tokens=["B"],
                token_logprobs=[-0.5],
            ),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(
            Request(input="p", return_logprobs=True, top_k=2, stream=True)
        )
        assert resp.logprobs == (
            TokenLogprob(token="A", logprob=-0.1),
            TokenLogprob(token="B", logprob=-0.5),
        )
        assert resp.top_logprobs is not None
        assert {(e.token, e.logprob) for e in resp.top_logprobs[0]} == {
            ("A", -0.1),
            ("B", -2.0),
        }

    @pytest.mark.anyio
    async def test_usage_from_final_chunk(self):
        chunks = [
            _make_chunk(text="ok", finish_reason="stop"),
            _make_usage_chunk(7, 3),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", stream=True))
        assert resp.usage == UsageStats(
            input_tokens=7, output_tokens=3, total_tokens=10
        )

    @pytest.mark.anyio
    async def test_no_usage_chunk_usage_is_none(self):
        chunks = [_make_chunk(text="hi", finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", stream=True))
        assert resp.usage is None

    @pytest.mark.anyio
    async def test_response_metadata_captured_from_first_chunk(self):
        chunks = [
            _make_chunk(text="a", model="model-v1", system_fingerprint="fp_first"),
            _make_chunk(
                text="b",
                finish_reason="stop",
                model="model-v2",
                system_fingerprint="fp_second",
            ),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", stream=True))
        assert resp.response_model == "model-v1"
        assert resp.system_fingerprint == "fp_first"

    @pytest.mark.anyio
    async def test_echo_split_at_usage_boundary(self):
        # Echoed prompt tokens arrive first, then the sampled completion,
        # then the usage chunk that defines the split boundary.
        chunks = [
            _make_chunk(text="", tokens=["a", "b"], token_logprobs=[None, -1.0]),
            _make_chunk(
                text=" g",
                finish_reason="stop",
                tokens=[" g"],
                token_logprobs=[-0.5],
            ),
            _make_usage_chunk(prompt_tokens=2, completion_tokens=1),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="ab", score_input=True, top_k=0, stream=True))
        assert resp.input_scoring is not None
        assert [tl.token for tl in resp.input_scoring.token_logprobs] == ["a", "b"]
        assert resp.input_scoring.token_logprobs[0].logprob is None
        assert resp.logprobs == (TokenLogprob(token=" g", logprob=-0.5),)

    @pytest.mark.anyio
    async def test_echo_split_missing_usage_boundary_zero(self):
        chunks = [
            _make_chunk(
                text=" g",
                finish_reason="stop",
                tokens=["a", " g"],
                token_logprobs=[None, -0.5],
            ),
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="a", score_input=True, top_k=0, stream=True))
        assert resp.input_scoring is not None
        assert resp.input_scoring.token_logprobs == ()
        assert resp.logprobs is not None
        assert len(resp.logprobs) == 2

    @pytest.mark.anyio
    async def test_no_logprobs_object_on_any_chunk_is_none(self):
        chunks = [_make_chunk(text="x", finish_reason="stop")]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", return_logprobs=True, stream=True))
        assert resp.logprobs is None
        assert resp.top_logprobs is None

    @pytest.mark.anyio
    async def test_logprobs_object_with_empty_channels_is_empty_tuple(self):
        chunks = [
            _make_chunk(text="x", finish_reason="stop", tokens=[], token_logprobs=[])
        ]
        t, _ = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", return_logprobs=True, stream=True))
        assert resp.logprobs == ()
        assert resp.top_logprobs == ()

    @pytest.mark.anyio
    async def test_request_params_include_stream_and_stream_options(self):
        chunks = [_make_chunk(text="x", finish_reason="stop")]
        t, create = _make_transport(_AsyncIterator(chunks))
        resp = await t.arun(Request(input="p", stream=True))
        assert resp.request_params == {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert create.call_args.kwargs["stream"] is True
        assert create.call_args.kwargs["stream_options"] == {"include_usage": True}


class TestCompletionTopLogprobs:
    """Cover _completion_top_logprobs parsing and its defensive branches.

    Moved from gen_model.py to transports/openai_completions.py by RFC #25.
    """

    def test_non_sequence_returns_empty(self):
        assert _completion_top_logprobs(None) == []
        assert _completion_top_logprobs("AB") == []

    def test_parses_and_skips_malformed_entries(self):
        # Mix of: a valid dict; a None entry (→ {}); a non-Mapping entry
        # (skipped); and a dict whose non-numeric value ("no") and non-str key
        # (2) are both filtered out, leaving only the valid pair.
        raw = [{"A": -0.1}, None, "x", {"y": -0.5, "1": "no", 2: -0.3}]
        assert _completion_top_logprobs(raw) == [{"A": -0.1}, {}, {"y": -0.5}]
