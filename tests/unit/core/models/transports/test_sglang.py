"""Unit tests for SglangTransport (lower/lift of the native /generate protocol).

URL derivation, token-text normalization, triple parsing, the radix-cache
guard, and the n>1 list-response handling moved here from the legacy
SglangGenModel tests (tests/unit/core/models/test_sglang_gen_model.py) when
RFC #25 relocated the wire logic into the transport: everything is driven
through ``Transport.arun(Request(...))`` with the OpenAI client's ``post``
mocked — no real traffic.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sieval.core.models.ir import (
    Request,
    SamplingParams,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from sieval.core.models.transports.sglang import SglangTransport, _normalize_token_text


def _make_transport(payload: dict | list) -> tuple[SglangTransport, AsyncMock]:
    client = MagicMock()
    post = AsyncMock(return_value=payload)
    client.post = post
    t = SglangTransport(client=client, model="m", api_base="http://host/v1")
    return t, post


def _meta(**overrides: Any) -> dict:
    meta = {
        "prompt_tokens": 0,
        "completion_tokens": 1,
        "cached_tokens": 0,
        "finish_reason": {"type": "stop"},
    }
    meta.update(overrides)
    return meta


# ===================================================================
# URL derivation (moved from the legacy SglangGenModel tests)
# ===================================================================
class TestGenerateUrl:
    def test_strips_v1_suffix(self):
        t = SglangTransport(
            client=MagicMock(), model="x", api_base="http://host:8000/v1"
        )
        assert t._generate_url() == "http://host:8000/generate"

    def test_trailing_slash_base(self):
        t = SglangTransport(
            client=MagicMock(), model="x", api_base="http://host:8000/v1/"
        )
        assert t._generate_url() == "http://host:8000/generate"

    def test_no_v1_suffix(self):
        t = SglangTransport(client=MagicMock(), model="x", api_base="http://host:8000")
        assert t._generate_url() == "http://host:8000/generate"

    def test_none_base(self):
        t = SglangTransport(client=MagicMock(), model="x", api_base=None)
        assert t._generate_url() == "/generate"


# ===================================================================
# Token text normalization (moved from the legacy SglangGenModel tests)
# ===================================================================
class TestNormalizeTokenText:
    def test_space_marker(self):
        assert _normalize_token_text("ĠA") == " A"

    def test_newline_marker(self):
        assert _normalize_token_text("Ċ") == "\n"

    def test_plain_unchanged(self):
        assert _normalize_token_text(" A") == " A"

    def test_none_text_raises(self):
        # Server without detokenization (--skip-tokenizer-init) returns no text;
        # fail loud rather than crash on None.replace or degrade to "".
        with pytest.raises(RuntimeError, match="no token text"):
            _normalize_token_text(None)


class TestLower:
    def test_max_tokens_maps_to_max_new_tokens(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", sampling=SamplingParams(max_tokens=7)))
        assert body["sampling_params"]["max_new_tokens"] == 7

    def test_score_input_sets_logprob_start_len_0(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", score_input=True, top_k=5))
        assert body["logprob_start_len"] == 0
        assert body["return_logprob"] is True
        assert body["top_logprobs_num"] == 5
        assert body["return_text_in_logprobs"] is True

    def test_no_score_input_sets_logprob_start_len_minus_1(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", return_logprobs=True, top_k=3))
        assert body["logprob_start_len"] == -1

    def test_top_k_maps_to_top_logprobs_num(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", return_logprobs=True, top_k=4))
        assert body["top_logprobs_num"] == 4

    def test_no_logprobs_request_omits_logprob_fields(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi"))
        assert "return_logprob" not in body
        assert "logprob_start_len" not in body

    def test_echo_never_appears_in_wire_body(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", score_input=True))
        assert "echo" not in body
        assert "echo" not in body["sampling_params"]

    def test_prefix_maps_to_prefill(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", prefix="PRE"))
        assert body["sampling_params"]["prefill"] == "PRE"

    def test_score_input_forces_min_one_new_token(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(Request(input="hi", score_input=True))
        assert body["sampling_params"]["max_new_tokens"] == 1

    def test_all_sampling_params_map(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            Request(
                input="hi",
                sampling=SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    top_k_sampling=40,
                    stop=("</s>",),
                    frequency_penalty=0.1,
                    presence_penalty=0.2,
                    n=3,
                ),
            )
        )
        sp = body["sampling_params"]
        assert sp["temperature"] == 0.7
        assert sp["top_p"] == 0.9
        assert sp["top_k"] == 40
        assert sp["stop"] == ["</s>"]
        assert sp["frequency_penalty"] == 0.1
        assert sp["presence_penalty"] == 0.2
        assert sp["n"] == 3

    def test_non_str_input_raises(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        with pytest.raises(TypeError, match="str input"):
            t._lower(Request(input=[{"role": "user", "content": "x"}]))


class TestExtraWireParams:
    """extra_wire_params go through the OpenAI→sglang table or are dropped."""

    def test_known_key_mapped_into_sampling_params(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            Request(input="hi", extra_wire_params={"repetition_penalty": 1.1})
        )
        assert body["sampling_params"]["repetition_penalty"] == 1.1
        assert "repetition_penalty" not in body  # nested, never top-level

    def test_unknown_keys_dropped(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            Request(input="hi", extra_wire_params={"seed": 1, "custom_flag": True})
        )
        assert "seed" not in body
        assert "seed" not in body["sampling_params"]
        assert "custom_flag" not in body
        assert "custom_flag" not in body["sampling_params"]

    def test_explicit_ir_field_wins_over_extra_wire_params(self):
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        body = t._lower(
            Request(
                input="hi",
                sampling=SamplingParams(max_tokens=7),
                extra_wire_params={"max_tokens": 99},
            )
        )
        assert body["sampling_params"]["max_new_tokens"] == 7


# ===================================================================
# Wire plumbing: URL, cast_to, body, provenance
# ===================================================================
class TestWirePlumbing:
    @pytest.mark.anyio
    async def test_posts_to_generate_url_with_body(self):
        t, post = _make_transport({"text": "out", "meta_info": _meta(prompt_tokens=1)})
        await t.arun(Request(input="hi"))
        assert post.call_args[0][0] == "http://host/generate"
        assert post.call_args[1]["cast_to"] is object
        body = post.call_args[1]["body"]
        assert body["text"] == "hi"
        assert "return_logprob" not in body

    @pytest.mark.anyio
    async def test_stream_flag_is_ignored_single_post(self):
        # Native /generate is always one POST; Request.stream never reaches
        # the wire body (pure scheduling, no content impact).
        t, post = _make_transport({"text": "x", "meta_info": _meta(prompt_tokens=1)})
        resp = await t.arun(Request(input="hi", stream=True))
        assert post.call_count == 1
        assert "stream" not in post.call_args[1]["body"]
        assert resp.texts == ("x",)

    @pytest.mark.anyio
    async def test_request_params_excludes_prompt_text(self):
        t, post = _make_transport(
            {"text": "x", "meta_info": _meta(prompt_tokens=1, completion_tokens=1)}
        )
        resp = await t.arun(
            Request(input="secret prompt", sampling=SamplingParams(temperature=0.0))
        )
        # the raw prompt is sent on the wire...
        assert post.call_args[1]["body"]["text"] == "secret prompt"
        # ...but is NOT persisted into per-call request_params.
        assert resp.request_params is not None
        assert "text" not in resp.request_params
        assert resp.request_params["sampling_params"] == {"temperature": 0.0}

    @pytest.mark.anyio
    async def test_response_model_is_configured_model(self):
        t, _ = _make_transport({"text": "x", "meta_info": _meta(prompt_tokens=1)})
        resp = await t.arun(Request(input="hi"))
        assert resp.response_model == "m"


# ===================================================================
# n>1: /generate returns a list of per-sample dicts
# ===================================================================
class TestNSamples:
    @pytest.mark.anyio
    async def test_list_response_yields_per_sample_texts(self):
        t, post = _make_transport(
            [
                {
                    "text": "a",
                    "meta_info": _meta(prompt_tokens=4, completion_tokens=1),
                },
                {
                    "text": "b",
                    "meta_info": _meta(
                        prompt_tokens=4,
                        completion_tokens=2,
                        finish_reason={"type": "length"},
                    ),
                },
                {
                    "text": "c",
                    "meta_info": _meta(prompt_tokens=4, completion_tokens=3),
                },
            ]
        )
        resp = await t.arun(Request(input="hi", sampling=SamplingParams(n=3)))
        assert resp.texts == ("a", "b", "c")
        assert resp.finish_reasons == ("stop", "length", "stop")
        # prompt tokens counted once, completions summed.
        assert resp.usage == UsageStats(
            input_tokens=4, output_tokens=6, total_tokens=10
        )
        assert post.call_args[1]["body"]["sampling_params"]["n"] == 3


class TestLift:
    @pytest.mark.anyio
    async def test_triple_populates_token_id(self):
        meta = _meta(
            prompt_tokens=1,
            output_token_logprobs=[[-0.5, 42, "hello"]],
        )
        t, _ = _make_transport({"text": "hello", "meta_info": meta})
        resp = await t.arun(Request(input="x", return_logprobs=True, top_k=0))
        assert resp.logprobs is not None
        assert resp.logprobs[0].token_id == 42
        assert resp.logprobs[0].token == "hello"
        assert resp.logprobs[0].logprob == -0.5

    @pytest.mark.anyio
    async def test_normalize_token_text_called(self):
        meta = _meta(output_token_logprobs=[[-0.1, 7, "ĠA"]])  # "ĠA"
        t, _ = _make_transport({"text": "", "meta_info": meta})
        resp = await t.arun(Request(input="x", return_logprobs=True))
        assert resp.logprobs is not None
        assert resp.logprobs[0].token == " A"  # Ġ normalized to space

    @pytest.mark.anyio
    async def test_newline_marker_normalized_in_tokens(self):
        meta = _meta(output_token_logprobs=[[-0.2, 9, "Ċ"]])
        t, _ = _make_transport({"text": "\n", "meta_info": meta})
        resp = await t.arun(Request(input="x", return_logprobs=True))
        assert resp.logprobs == (TokenLogprob(token="\n", logprob=-0.2, token_id=9),)

    @pytest.mark.anyio
    async def test_input_token_logprobs_go_to_input_scoring(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "a"], [-1.2, 2, "b"]],
            output_token_logprobs=[[-0.3, 3, "c"]],
            cached_tokens=0,
        )
        t, _ = _make_transport({"text": "c", "meta_info": meta})
        resp = await t.arun(Request(input="ab", score_input=True, top_k=0))
        assert resp.input_scoring is not None
        assert len(resp.input_scoring.token_logprobs) == 2
        assert resp.input_scoring.token_logprobs[0].logprob is None  # first token
        assert resp.input_scoring.token_logprobs[1].token_id == 2
        # output segment lands in logprobs, never merged with the input segment
        assert resp.logprobs is not None
        assert len(resp.logprobs) == 1
        assert resp.logprobs[0].token == "c"

    @pytest.mark.anyio
    async def test_usage_and_finish_reason(self):
        meta = _meta(
            prompt_tokens=4, completion_tokens=6, finish_reason={"type": "length"}
        )
        t, _ = _make_transport({"text": "out", "meta_info": meta})
        resp = await t.arun(Request(input="x"))
        assert resp.usage == UsageStats(
            input_tokens=4, output_tokens=6, total_tokens=10
        )
        assert resp.finish_reasons == ("length",)
        assert resp.texts == ("out",)

    @pytest.mark.anyio
    async def test_usage_none_when_prompt_tokens_absent(self):
        # echo=False path (no radix guard); _parse_usage returns None when the
        # server omitted the token counts — absence is not zeros.
        meta = {
            "completion_tokens": 1,
            "finish_reason": "stop",
            "output_token_logprobs": [[-0.1, 1, " A"]],
        }
        t, _ = _make_transport({"text": " A", "meta_info": meta})
        resp = await t.arun(Request(input="p", return_logprobs=True))
        assert resp.usage is None


# ===================================================================
# Top-k triple parsing
# ===================================================================
class TestTopKParsing:
    @pytest.mark.anyio
    async def test_top_k_preserves_token_id_and_normalizes_text(self):
        meta = _meta(
            output_token_logprobs=[[-0.7, 100, "ĠB"]],
            output_top_logprobs=[[[-0.7, 100, "ĠB"], [-1.2, 101, "Ċ"]]],
        )
        t, _ = _make_transport({"text": " B", "meta_info": meta})
        resp = await t.arun(Request(input="p", return_logprobs=True, top_k=2))
        assert resp.top_logprobs == (
            (
                TopKEntry(token=" B", logprob=-0.7, token_id=100),
                TopKEntry(token="\n", logprob=-1.2, token_id=101),
            ),
        )

    @pytest.mark.anyio
    async def test_empty_per_token_top_entry_becomes_empty_tuple(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "Q"], [-0.2, 2, " B"]],
            output_token_logprobs=[[-0.3, 3, " g"]],
            output_top_logprobs=[None, [[-0.3, 3, " g"]]],
        )
        t, _ = _make_transport({"text": " g", "meta_info": meta})
        resp = await t.arun(Request(input="Q B", score_input=True, top_k=1))
        assert resp.top_logprobs == (
            (),
            (TopKEntry(token=" g", logprob=-0.3, token_id=3),),
        )

    @pytest.mark.anyio
    async def test_top_logprobs_none_when_absent(self):
        meta = _meta(output_token_logprobs=[[-0.1, 1, " A"]])
        t, _ = _make_transport({"text": " A", "meta_info": meta})
        resp = await t.arun(Request(input="p", return_logprobs=True))
        assert resp.top_logprobs is None

    @pytest.mark.anyio
    async def test_none_token_text_in_top_raises(self):
        """A top-k entry with no token text (no detokenization) fails loud."""
        meta = _meta(
            output_token_logprobs=[[-0.1, 1, " A"]],
            output_top_logprobs=[[[-0.1, 1, None]]],
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="no token text"):
            await t.arun(Request(input="p", return_logprobs=True, top_k=1))


# ===================================================================
# Guards: response shape, radix cache, empty logprobs
# ===================================================================
class TestGuards:
    @pytest.mark.anyio
    async def test_radix_cache_hit_with_score_input_raises(self):
        meta = _meta(
            prompt_tokens=3,
            input_token_logprobs=[[None, 1, "a"]],  # truncated (cache hit)
            cached_tokens=2,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="radix prefix cache"):
            await t.arun(Request(input="abc", score_input=True))

    @pytest.mark.anyio
    async def test_truncated_input_logprobs_raises(self):
        # cached_tokens reads 0, but the returned count < prompt_tokens.
        meta = _meta(
            prompt_tokens=5,
            input_token_logprobs=[[-0.1, 1, " star"]],
            cached_tokens=0,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="partial echoed-input"):
            await t.arun(Request(input="abcde", score_input=True))

    @pytest.mark.anyio
    async def test_missing_prompt_tokens_with_score_input_raises(self):
        meta = {"completion_tokens": 1, "finish_reason": "stop"}
        meta["input_token_logprobs"] = [[None, 1, "a"]]
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="omitted prompt_tokens"):
            await t.arun(Request(input="a", score_input=True))

    @pytest.mark.anyio
    async def test_full_echoed_input_passes_guard(self):
        meta = _meta(
            prompt_tokens=2,
            input_token_logprobs=[[None, 1, "a"], [-0.1, 2, " b"]],
            cached_tokens=0,
        )
        t, _ = _make_transport({"text": "", "meta_info": meta})
        resp = await t.arun(Request(input="ab", score_input=True))
        assert resp.input_scoring is not None
        assert [tl.token for tl in resp.input_scoring.token_logprobs] == ["a", " b"]

    @pytest.mark.anyio
    async def test_no_score_input_skips_radix_guard(self):
        """Output-only logprobs (CMMLU shape) — cache truncation is irrelevant."""
        meta = _meta(
            prompt_tokens=5,
            cached_tokens=4,
            output_token_logprobs=[[-0.1, 1, " A"]],
        )
        t, _ = _make_transport({"text": " A", "meta_info": meta})
        resp = await t.arun(Request(input="p", return_logprobs=True))
        assert resp.logprobs is not None
        assert resp.logprobs[0].token == " A"

    @pytest.mark.anyio
    async def test_no_logprobs_channels_raises(self):
        """Logprobs requested but every channel came back empty → fail loud."""
        t, _ = _make_transport({"text": "", "meta_info": _meta()})
        with pytest.raises(RuntimeError, match="no logprobs"):
            await t.arun(Request(input="x", return_logprobs=True))

    @pytest.mark.anyio
    async def test_empty_echoed_input_raises(self):
        # score_input with an empty echoed input passes the radix guard
        # (0 == prompt_tokens) but still yields no logprobs at all.
        meta = _meta(prompt_tokens=0, input_token_logprobs=[])
        t, _ = _make_transport({"text": "", "meta_info": meta})
        with pytest.raises(RuntimeError, match="no logprobs"):
            await t.arun(Request(input="p", score_input=True))

    @pytest.mark.anyio
    async def test_missing_meta_info_raises(self):
        t, _ = _make_transport({"text": "hi"})
        with pytest.raises(RuntimeError, match="missing meta_info"):
            await t.arun(Request(input="x"))

    @pytest.mark.anyio
    async def test_list_with_non_dict_element_raises(self):
        t, _ = _make_transport(["nope"])
        with pytest.raises(RuntimeError, match="missing meta_info"):
            await t.arun(Request(input="x"))

    @pytest.mark.anyio
    async def test_empty_list_response_raises(self):
        t, _ = _make_transport([])
        with pytest.raises(RuntimeError, match="missing meta_info"):
            await t.arun(Request(input="x"))
