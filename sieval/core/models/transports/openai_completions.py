"""OpenAICompletionsTransport: OpenAI ``/v1/completions`` frontend for the IR.

Capabilities: Completion, InputScoring (via ``echo=True``), TopKLogprobs,
SampledLogprobs.

InputScoring is implemented via ``echo=True`` — the only place in the entire
codebase where ``echo`` appears. It is an OpenAI-Completions-specific workaround
for the missing native scoring endpoint and must not leak into any other
transport or the IR layer. On lift, the echoed prompt tokens are split off into
``Response.input_scoring`` at the ``prompt_tokens`` boundary; the remaining
tokens are the sampled completion (``Response.logprobs``).

Streaming: ``Request.stream=True`` makes the transport consume the SSE stream
and accumulate chunks internally (ported from the legacy ``GenModel`` impl);
the caller always receives one terminal :class:`Response`. ``stream=None``
defaults to a single-shot request.

``token_id`` is not available on this wire protocol, so it is left ``None``.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from collections.abc import Mapping, Sequence
from typing import Any

from ..capabilities import Capability
from ..ir import (
    InputScoringResult,
    Request,
    Response,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)


def _completion_top_logprobs(raw: object) -> list[dict[str, float]]:
    """Sanitize the completions API's per-position top-logprob dicts."""
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return []

    top_logprobs = []
    for item in raw:
        if item is None:
            top_logprobs.append({})
            continue
        if not isinstance(item, Mapping):
            continue
        top_logprobs.append(
            {
                token: float(logprob)
                for token, logprob in item.items()
                if isinstance(token, str) and isinstance(logprob, int | float)
            }
        )
    return top_logprobs


def _usage_stats(raw: Any) -> UsageStats | None:
    """Map an OpenAI usage object to :class:`UsageStats` (None when absent)."""
    if raw is None:
        return None
    return UsageStats(
        input_tokens=raw.prompt_tokens,
        output_tokens=raw.completion_tokens,
        total_tokens=raw.total_tokens,
    )


class OpenAICompletionsTransport:
    """Transport for OpenAI ``/v1/completions``."""

    CAPABILITIES: frozenset[Capability] = frozenset(
        {
            Capability.Completion,
            Capability.InputScoring,
            Capability.TopKLogprobs,
            Capability.SampledLogprobs,
        }
    )

    def __init__(self, client: Any, model: str):
        self._client = client
        self._model = model

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    # ── lower ─────────────────────────────────────────────────────────────────

    def _lower(self, req: Request) -> dict[str, Any]:
        if not isinstance(req.input, str):
            raise TypeError(
                "OpenAICompletionsTransport requires str input (Completion modality)."
            )

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

        # echo appears ONLY here — the InputScoring workaround.
        if req.score_input:
            params["echo"] = True

        # completions `logprobs` is the count of top alternatives (0 → own token
        # logprob only); required to receive any logprobs at all.
        if req.return_logprobs or req.score_input:
            params["logprobs"] = req.top_k

        params["stream"] = bool(req.stream)

        if req.extra_wire_params:
            for k, v in req.extra_wire_params.items():
                params.setdefault(k, v)

        # Injected default last so an explicit stream_options (via
        # extra_wire_params) wins, matching the legacy "if not present" rule.
        if params["stream"] and "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}

        return params

    # ── lift ────────────────────────────────────────────────────────────────

    def _build_response(
        self,
        *,
        texts: list[str],
        finish_reasons: list[str],
        tokens: list[str],
        token_logprobs: list[float | None],
        top_raw: list[dict[str, float]],
        saw_logprobs: bool,
        score_input: bool,
        usage: UsageStats | None,
        params: dict[str, Any],
        response_model: str | None,
        system_fingerprint: str | None,
    ) -> Response:
        """Assemble the terminal Response shared by both wire paths.

        ``saw_logprobs`` is the presence signal (a logprobs object appeared on
        the wire): only then are ``logprobs``/``top_logprobs`` tuples — possibly
        empty — attached; otherwise they stay ``None`` (empty-vs-absent
        contract on :class:`Response`).
        """
        input_scoring: InputScoringResult | None = None
        logprobs: tuple[TokenLogprob, ...] | None = None
        top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None

        if saw_logprobs:
            all_tokens = tuple(
                TokenLogprob(token=tok, logprob=lp)
                for tok, lp in zip(tokens, token_logprobs, strict=False)
            )
            all_topk = tuple(
                tuple(TopKEntry(token=tok, logprob=lp) for tok, lp in per_token.items())
                for per_token in top_raw
            )
            if score_input:
                # Split at the prompt/completion boundary. The echoed prompt
                # occupies the first prompt_tokens positions.
                boundary = usage.input_tokens if usage is not None else 0
                input_scoring = InputScoringResult(token_logprobs=all_tokens[:boundary])
                logprobs = all_tokens[boundary:]
                top_logprobs = all_topk[boundary:]
            else:
                logprobs = all_tokens
                top_logprobs = all_topk

        return Response(
            texts=tuple(texts),
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            input_scoring=input_scoring,
            usage=usage,
            finish_reasons=tuple(finish_reasons),
            request_params=dict(params),
            response_model=response_model,
            system_fingerprint=system_fingerprint,
        )

    def _lift(
        self, resp: Any, *, n: int, req: Request, params: dict[str, Any]
    ) -> Response:
        texts = [""] * n
        finish_reasons = [""] * n
        tokens: list[str] = []
        token_logprobs: list[float | None] = []
        top_raw: list[dict[str, float]] = []
        saw_logprobs = False

        for choice in resp.choices:
            idx = choice.index
            if not 0 <= idx < n:
                continue
            texts[idx] += choice.text or ""
            finish_reasons[idx] = choice.finish_reason or ""
            if idx == 0:
                lp_obj = choice.logprobs
                if lp_obj is not None:
                    saw_logprobs = True
                    if getattr(lp_obj, "tokens", None):
                        tokens.extend(lp_obj.tokens)
                    if getattr(lp_obj, "token_logprobs", None):
                        token_logprobs.extend(lp_obj.token_logprobs)
                    top_raw.extend(
                        _completion_top_logprobs(getattr(lp_obj, "top_logprobs", None))
                    )

        return self._build_response(
            texts=texts,
            finish_reasons=finish_reasons,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_raw=top_raw,
            saw_logprobs=saw_logprobs,
            score_input=req.score_input,
            usage=_usage_stats(getattr(resp, "usage", None)),
            params=params,
            response_model=getattr(resp, "model", None),
            system_fingerprint=getattr(resp, "system_fingerprint", None),
        )

    async def _lift_stream(
        self, stream: Any, *, n: int, req: Request, params: dict[str, Any]
    ) -> Response:
        texts = [""] * n
        finish_reasons = [""] * n
        tokens: list[str] = []
        token_logprobs: list[float | None] = []
        top_raw: list[dict[str, float]] = []
        saw_logprobs = False
        usage: UsageStats | None = None
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
                    texts[idx] += choice.text or ""
                    finish_reasons[idx] = choice.finish_reason or ""
                    if idx == 0:
                        lp_obj = getattr(choice, "logprobs", None)
                        if lp_obj is not None:
                            saw_logprobs = True
                            if getattr(lp_obj, "tokens", None):
                                tokens.extend(lp_obj.tokens)
                            if getattr(lp_obj, "token_logprobs", None):
                                token_logprobs.extend(lp_obj.token_logprobs)
                            top_raw.extend(
                                _completion_top_logprobs(
                                    getattr(lp_obj, "top_logprobs", None)
                                )
                            )
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage_stats(chunk_usage)

        return self._build_response(
            texts=texts,
            finish_reasons=finish_reasons,
            tokens=tokens,
            token_logprobs=token_logprobs,
            top_raw=top_raw,
            saw_logprobs=saw_logprobs,
            score_input=req.score_input,
            usage=usage,
            params=params,
            response_model=response_model,
            system_fingerprint=system_fingerprint,
        )

    # ── arun ──────────────────────────────────────────────────────────────────

    async def arun(self, req: Request) -> Response:
        params = self._lower(req)
        n = req.sampling.n if req.sampling is not None else 1
        resp = await self._client.completions.create(
            model=self._model, prompt=req.input, **params
        )
        if params["stream"]:
            return await self._lift_stream(resp, n=n, req=req, params=params)
        return self._lift(resp, n=n, req=req, params=params)
