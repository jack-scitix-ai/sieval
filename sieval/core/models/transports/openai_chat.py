"""OpenAIChatTransport: OpenAI ``/v1/chat/completions`` frontend for the IR.

Capabilities: Chat, FunctionCalling, TopKLogprobs, SampledLogprobs,
StructuredOutput.

InputScoring is NOT supported — ``echo`` has no equivalent in chat completions.
A Request with ``score_input=True`` (or a stateful ``session_id``, which chat
completions cannot honour) is rejected here, never silently ignored. This is the
fix for the historical "echo silently dropped on chat" bug.

Streaming: ``Request.stream=True`` makes the transport consume the SSE stream
and accumulate deltas internally (ported from the legacy ``ChatModel`` impl);
the caller always receives one terminal :class:`Response`. ``stream=None``
defaults to a single-shot request.

``token_id`` is not available on this wire protocol, so it is left ``None``.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from typing import Any

from ..capabilities import Capability
from ..exceptions import CapabilityError
from ..ir import (
    ReasoningOutput,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK tool-call object to a plain dict."""
    if hasattr(tc, "model_dump"):
        return tc.model_dump()
    if isinstance(tc, dict):
        return dict(tc)
    return dict(vars(tc)) if hasattr(tc, "__dict__") else {}


def _usage_stats(raw: Any) -> UsageStats | None:
    """Map an OpenAI usage object to :class:`UsageStats` (None when absent)."""
    if raw is None:
        return None
    return UsageStats(
        input_tokens=raw.prompt_tokens,
        output_tokens=raw.completion_tokens,
        total_tokens=raw.total_tokens,
    )


def _delta_reasoning(part: Any) -> str:
    """Extract reasoning text from a message or delta (either field spelling)."""
    rc = getattr(part, "reasoning", None)
    if rc:
        return str(rc)
    rc = getattr(part, "reasoning_content", None)
    return str(rc) if rc else ""


def _content_to_ir(
    content: list,
) -> tuple[tuple[TokenLogprob, ...], tuple[tuple[TopKEntry, ...], ...]]:
    """Map chat logprobs content items to IR token / top-k tuples."""
    logprobs = tuple(
        TokenLogprob(token=item.token, logprob=item.logprob) for item in content
    )
    top_logprobs = tuple(
        tuple(
            TopKEntry(token=t.token, logprob=t.logprob)
            for t in (getattr(item, "top_logprobs", None) or [])
        )
        for item in content
    )
    return logprobs, top_logprobs


class OpenAIChatTransport:
    """Transport for OpenAI ``/v1/chat/completions``."""

    CAPABILITIES: frozenset[Capability] = frozenset(
        {
            Capability.Chat,
            Capability.FunctionCalling,
            Capability.TopKLogprobs,
            Capability.SampledLogprobs,
            Capability.StructuredOutput,
        }
    )

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    # ── lower ─────────────────────────────────────────────────────────────────

    def _lower(self, req: Request) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if req.score_input:
            raise CapabilityError(
                "OpenAIChatTransport does not support InputScoring "
                "(echo/score_input has no chat-completions equivalent)."
            )
        if req.session_id is not None:
            raise CapabilityError(
                "OpenAIChatTransport does not support stateful session_id "
                "(chat completions has no previous_response_id)."
            )

        if isinstance(req.input, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": req.input}]
        else:
            messages = list(req.input)

        params: dict[str, Any] = {}
        sp = req.sampling
        if sp is not None:
            if sp.max_tokens is not None:
                params["max_tokens"] = sp.max_tokens
            if sp.temperature is not None:
                params["temperature"] = sp.temperature
            if sp.top_p is not None:
                params["top_p"] = sp.top_p
            if sp.top_k_sampling is not None:
                # vLLM extension; upstream OpenAI rejects it, matching the
                # legacy behaviour of forwarding top_k verbatim.
                params["top_k"] = sp.top_k_sampling
            if sp.stop is not None:
                params["stop"] = list(sp.stop)
            if sp.stop_token_ids is not None:
                params["stop_token_ids"] = list(sp.stop_token_ids)
            if sp.seed is not None:
                params["seed"] = sp.seed
            if sp.frequency_penalty is not None:
                params["frequency_penalty"] = sp.frequency_penalty
            if sp.presence_penalty is not None:
                params["presence_penalty"] = sp.presence_penalty
            if sp.n != 1:
                params["n"] = sp.n

        if req.return_logprobs:
            params["logprobs"] = True
            if req.top_k > 0:
                params["top_logprobs"] = req.top_k

        if req.response_format is not None:
            params["response_format"] = req.response_format
        if req.tools is not None:
            params["tools"] = req.tools
        if req.reasoning is not None and req.reasoning.effort is not None:
            params["reasoning_effort"] = req.reasoning.effort

        params["stream"] = bool(req.stream)

        if req.extra_wire_params:
            for k, v in req.extra_wire_params.items():
                params.setdefault(k, v)

        # Injected default last so an explicit stream_options (via
        # extra_wire_params) wins, matching the legacy "if not present" rule.
        if params["stream"] and "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        return messages, params

    # ── lift (single-shot) ────────────────────────────────────────────────────

    def _lift(self, resp: Any, *, n: int, params: dict[str, Any]) -> Response:
        texts = [""] * n
        finish_reasons = [""] * n
        reasoning_text = ""
        tool_calls: tuple[dict[str, Any], ...] | None = None
        logprobs: tuple[TokenLogprob, ...] | None = None
        top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None

        for choice in resp.choices:
            idx = choice.index
            if not 0 <= idx < n:
                continue
            message = choice.message
            if message.content is not None:
                texts[idx] += message.content
            finish_reasons[idx] = choice.finish_reason or ""
            if idx == 0:
                reasoning_text = _delta_reasoning(message)
                raw_calls = getattr(message, "tool_calls", None)
                if raw_calls:
                    tool_calls = tuple(_tool_call_to_dict(tc) for tc in raw_calls)
                lp_obj = choice.logprobs
                if lp_obj is not None:
                    logprobs, top_logprobs = _content_to_ir(
                        getattr(lp_obj, "content", None) or []
                    )

        return Response(
            texts=tuple(texts),
            reasoning=ReasoningOutput(text=reasoning_text) if reasoning_text else None,
            tool_calls=tool_calls,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            usage=_usage_stats(getattr(resp, "usage", None)),
            finish_reasons=tuple(finish_reasons),
            request_params=dict(params),
            response_model=getattr(resp, "model", None),
            system_fingerprint=getattr(resp, "system_fingerprint", None),
        )

    # ── lift (streaming) ──────────────────────────────────────────────────────

    async def _lift_stream(
        self, stream: Any, *, n: int, params: dict[str, Any]
    ) -> Response:
        texts = [""] * n
        finish_reasons = [""] * n
        reasoning_text = ""
        usage: UsageStats | None = None
        saw_logprobs = False
        lp_tokens: list[TokenLogprob] = []
        lp_topk: list[tuple[TopKEntry, ...]] = []
        response_model: str | None = None
        system_fingerprint: str | None = None

        async for chunk in stream:
            if response_model is None:
                response_model = getattr(chunk, "model", None)
            if system_fingerprint is None:
                system_fingerprint = getattr(chunk, "system_fingerprint", None)
            if chunk.choices:
                for choice in chunk.choices:
                    idx = choice.index
                    if not 0 <= idx < n:
                        continue
                    finish_reasons[idx] = choice.finish_reason or ""
                    if choice.delta:
                        if choice.delta.content:
                            texts[idx] += choice.delta.content
                        if idx == 0:
                            reasoning_text += _delta_reasoning(choice.delta)
                    if idx == 0:
                        lp_obj = getattr(choice, "logprobs", None)
                        if lp_obj is not None:
                            saw_logprobs = True
                            content = getattr(lp_obj, "content", None) or []
                            if content:
                                chunk_lp, chunk_topk = _content_to_ir(content)
                                lp_tokens.extend(chunk_lp)
                                lp_topk.extend(chunk_topk)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage_stats(chunk_usage)

        return Response(
            texts=tuple(texts),
            reasoning=ReasoningOutput(text=reasoning_text) if reasoning_text else None,
            logprobs=tuple(lp_tokens) if saw_logprobs else None,
            top_logprobs=tuple(lp_topk) if saw_logprobs else None,
            usage=usage,
            finish_reasons=tuple(finish_reasons),
            request_params=dict(params),
            response_model=response_model,
            system_fingerprint=system_fingerprint,
        )

    # ── arun ──────────────────────────────────────────────────────────────────

    async def arun(self, req: Request) -> Response:
        messages, params = self._lower(req)
        n = req.sampling.n if req.sampling is not None else 1
        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, **params
        )
        if params["stream"]:
            return await self._lift_stream(resp, n=n, params=params)
        return self._lift(resp, n=n, params=params)
