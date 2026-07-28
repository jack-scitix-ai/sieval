"""Abstract model base class and shared types for model backends.

RFC #25: ``arun(Request) -> Response`` is the one primitive — it acquires both
limiters and delegates to the composed :class:`Transport`. ``agenerate`` and
``alogprobs`` are capability-gated sugar: thin wrappers that build a
:class:`Request` from legacy OpenAI-style kwargs, run it through ``arun``, and
bridge the :class:`Response` back to the legacy :class:`ModelOutput` shape for
existing consumers. New code should call ``arun`` directly.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import copy
from dataclasses import dataclass, replace
from typing import Any, NotRequired, Self, TypedDict, cast

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue
from sieval.core.utils.concurrency import CompositeLimiter
from sieval.core.utils.serialization import sieval_record

from .capabilities import Capability
from .exceptions import CapabilityError
from .ir import ReasoningParams, Request, Response, SamplingParams
from .transport import Transport


class ModelUsage(TypedDict):
    """Token usage statistics from a single model API call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ModelMeta(TypedDict):
    """Model identification metadata: name, endpoint, and default parameters."""

    model: str
    api_base: str | None
    default_params: dict[str, JSONValue]
    extra: NotRequired[dict[str, JSONValue]]


class ModelCallMeta(TypedDict):
    """Per-API-call metadata: model info, usage, params, finish reasons."""

    model: ModelMeta
    usage: NotRequired[ModelUsage]
    request_params: NotRequired[dict[str, JSONValue]]
    finish_reasons: NotRequired[list[str]]
    response_model: NotRequired[str]
    system_fingerprint: NotRequired[str | None]


class ModelQuotaSnapshot(TypedDict):
    """Snapshot of a single limiter's available and total concurrency tokens."""

    available: int
    total: int


class ModelQuotaInfo(TypedDict):
    """Combined parent (shared) and child (local) limiter quota info."""

    available: int | float
    total: int | float
    parent: ModelQuotaSnapshot | None
    child: ModelQuotaSnapshot | None


@sieval_record
@dataclass
class ModelOutput:
    """Standard return type from model calls."""

    model: ModelMeta  # should be auto-attached by Model implementations
    texts: list[str]
    finish_reasons: list[str] | None = None
    reasoning_texts: list[str] | None = None
    # Token texts follow the OpenAI / literal-whitespace convention (leading
    # spaces preserved, e.g. " A"). Backends that emit other markers (e.g.
    # sglang byte-level "ĠA") normalize to this contract before populating it,
    # so consumers can match on literal whitespace rather than per-tokenizer
    # markers.
    logprobs_tokens: list[str] | None = None
    logprobs: list[float | None] | None = None
    top_logprobs: list[dict[str, float]] | None = None
    usage: ModelUsage | None = None
    request_params: dict[str, JSONValue] | None = None
    # From API response — what the server actually used
    response_model: str | None = None
    system_fingerprint: str | None = None


# Legacy OpenAI-style kwarg -> SamplingParams field handled by the request
# builders. Anything not listed here (and not a first-class Request field)
# falls through to Request.extra_wire_params.
_SAMPLING_KWARGS: dict[str, str] = {
    "max_tokens": "max_tokens",
    "max_completion_tokens": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k_sampling",
    "seed": "seed",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
}


class Model[TModelInput]:
    """Base for all model backends.

    Composes a resource pool (two-level ``anyio`` limiters) with a
    :class:`Transport` (the provider frontend). ``arun`` is the one primitive;
    ``agenerate`` / ``alogprobs`` are backward-compatible sugar over it.
    Concrete backends only select a default Transport via
    ``_build_default_transport``; a bare ``Model`` has no transport and no
    capabilities.
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        concurrency_limit: int | None = None,
        parent_limiter: anyio.CapacityLimiter | None = None,
        extra: dict[str, JSONValue] | None = None,
        transport: Transport | None = None,
        **kwargs,
    ):
        self._model = model
        self._api_base = api_base

        self._client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            max_retries=max_retries,
        )
        # Default kwargs for the model
        self._kwargs = kwargs
        # Catch-all for user-defined, schema-external config (e.g., sequence_wrappers).
        # Stored separately from _kwargs — not passed to the API backend.
        self._extra = extra

        # Two-level concurrency control:
        # - parent_limiter: shared API quota (from base model)
        # - _limiter: this model's reserved quota
        self._parent_limiter = parent_limiter
        self._limiter = (
            anyio.CapacityLimiter(concurrency_limit)
            if concurrency_limit is not None
            else None
        )

        # Provider frontend. When not injected, the subclass builds its default
        # transport over the shared client (backward-compatible path). The
        # transport is the single source of truth for `capabilities`; a model
        # without one simply declares no IR capabilities and cannot `arun`.
        self._transport: Transport | None = (
            transport if transport is not None else self._build_default_transport()
        )

    def _build_default_transport(self) -> Transport | None:
        """Build this model's default Transport over ``self._client``.

        Overridden by each concrete backend to return its transport. Returns
        ``None`` on the base so that a bare ``Model`` subclass remains
        constructible (it just exposes no capabilities). Kept as a hook (rather
        than an abstract method) so lazily-imported transport modules avoid an
        import cycle with the backend subclasses.
        """
        return None

    def with_args(
        self,
        concurrency_limit: int | None = None,
        extra: dict[str, JSONValue] | None = None,
        **kwargs,
    ) -> Self:
        """Create a derived model sharing the same resource pool.

        A non-``None`` *concurrency_limit* reserves a sub-quota from this
        model's limiter; ``None`` shares the existing limiter.
        Multi-level derivation is forbidden.

        RFC #25 eventually decomposes this: per-task ``infer_args`` become
        plain :class:`Request` fields and the sub-quota an explicit
        resource-pool collaborator. That lands together with the task layer
        migrating from ``agenerate(**kwargs)`` to ``arun(Request)``; until
        then this remains the supported derivation mechanism.

        Example::

            child = base.with_args(concurrency_limit=64)
        """
        # Prevent multi-level derivation
        if concurrency_limit is not None and self._parent_limiter is not None:
            raise ValueError(
                "Cannot create multi-level model derivation. "
                "Multi-level resource pools are not supported. "
                "Please fork from the base model instead."
            )

        new_model = copy.copy(self)
        new_model._kwargs = {**self._kwargs, **kwargs}

        if extra is not None:
            new_model._extra = extra

        # If specifying a new concurrency_limit, create a new limiter
        if concurrency_limit is not None:
            new_model._limiter = anyio.CapacityLimiter(concurrency_limit)
            # Parent's limiter becomes our parent
            new_model._parent_limiter = self._limiter
        # Otherwise, share the same limiter and parent_limiter

        return new_model

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The IR features this model's Transport honours (empty if none)."""
        if self._transport is None:
            return frozenset()
        return self._transport.capabilities

    def assert_capability(self, *caps: Capability) -> None:
        """Raise :class:`CapabilityError` if any requested capability is missing.

        This is the setup-time gate that replaces silently ignoring unsupported
        parameters (e.g. ``echo=True`` on a chat backend).
        """
        missing = frozenset(caps) - self.capabilities
        if missing:
            raise CapabilityError(
                f"{type(self._transport).__name__} does not support: "
                + ", ".join(sorted(c.name for c in missing if c.name is not None))
            )

    def get_available_quota(self) -> int | float:
        """Return the minimum available tokens across both limiters."""
        capacities = []

        if self._parent_limiter is not None:
            capacities.append(self._parent_limiter.available_tokens)

        if self._limiter is not None:
            capacities.append(self._limiter.available_tokens)

        if not capacities:
            return float("inf")

        return min(capacities)

    def get_total_quota(self) -> int | float:
        """Return the child limiter's total tokens (ignores parent)."""
        if self._limiter is None:
            return float("inf")
        return self._limiter.total_tokens

    def get_quota_info(self) -> ModelQuotaInfo:
        """Return a structured breakdown of both parent and child limiter quotas."""
        info: ModelQuotaInfo = {
            "available": self.get_available_quota(),
            "total": self.get_total_quota(),
            "parent": None,
            "child": None,
        }

        if self._parent_limiter is not None:
            info["parent"] = {
                "available": self._parent_limiter.available_tokens,
                "total": self._parent_limiter.total_tokens,
            }

        if self._limiter is not None:
            info["child"] = {
                "available": self._limiter.available_tokens,
                "total": self._limiter.total_tokens,
            }

        return info

    @property
    def extra(self) -> dict[str, JSONValue]:
        """User-defined, schema-external model config, not sent to the API backend.

        Returns a shallow copy; mutating the returned dict does not affect
        the model instance.
        """
        return dict(self._extra) if self._extra else {}

    def meta(self) -> ModelMeta:
        """Return a ``ModelMeta`` dict identifying this model."""
        result: ModelMeta = {
            "model": self._model,
            "api_base": self._api_base,
            "default_params": dict(self._kwargs),
        }
        if self._extra:
            result["extra"] = dict(self._extra)
        return result

    # ── the primitive ─────────────────────────────────────────────────────────

    async def arun(self, req: Request) -> Response:
        """Run one inference through the Transport; acquires both limiters first.

        The provider-agnostic primitive. Transport contract:
        1. Returns a terminal Response (all internal tool loops resolved).
        2. Stateful mode: pass ``Request.session_id``, get ``Response.session_id``
           back.
        3. Stateless mode: echo ``Response.reasoning.opaque_roundtrip`` back as
           ``Request.reasoning.opaque_roundtrip`` on the next turn.
        """
        if self._transport is None:
            raise CapabilityError(
                f"{type(self).__name__} has no Transport; arun() is unavailable."
            )
        async with CompositeLimiter(self._parent_limiter, self._limiter):
            return await self._transport.arun(req)

    # ── legacy sugar (thin wrappers over arun) ────────────────────────────────

    async def agenerate(self, prompt: TModelInput, **kwargs) -> ModelOutput:
        """Generate text. Sugar over :meth:`arun`, returning ``ModelOutput``."""
        req = self._build_generate_request(prompt, **kwargs)
        resp = await self.arun(req)
        return self._response_to_model_output(resp)

    async def alogprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 1,
        logprobs: int = 5,
        echo: bool = True,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        """Extract logprobs. Sugar over :meth:`arun`, returning ``ModelOutput``.

        ``echo=True`` requests prompt-side scoring, which requires the
        ``InputScoring`` capability. On a backend that lacks it (e.g. a chat
        completions transport) this raises :class:`CapabilityError` at the
        call boundary instead of being silently ignored (the historical bug).
        """
        if echo:
            self.assert_capability(Capability.InputScoring)
        req = self._build_logprobs_request(
            prompt,
            max_tokens=max_tokens,
            logprobs=logprobs,
            score_input=echo,
            temperature=temperature,
            **kwargs,
        )
        resp = await self.arun(req)
        if (
            resp.logprobs is None
            and resp.input_scoring is None
            and resp.top_logprobs is None
        ):
            raise RuntimeError("logprobs requested but the server returned none.")
        return self._response_to_model_output(resp)

    # ── request builders (legacy kwargs -> IR) ────────────────────────────────

    @staticmethod
    def _validate_n(final_kwargs: dict[str, Any]) -> int:
        """Validate and return ``n`` from merged kwargs."""
        n = final_kwargs.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}: {n!r}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return n

    @staticmethod
    def _coerce_input(prompt: Any) -> str | list[dict[str, Any]]:
        """Coerce a legacy prompt into ``Request.input``.

        Modality validation stays with the Transport (a completions transport
        rejects non-str input); this only normalizes the container shape.
        """
        if isinstance(prompt, str):
            return prompt
        try:
            return [dict(m) for m in prompt]
        except TypeError:
            raise TypeError(
                "prompt must be a string or an iterable of messages, "
                f"got {type(prompt).__name__}."
            ) from None

    def _kwargs_to_request(
        self, input_: str | list[dict[str, Any]], final_kwargs: dict[str, Any]
    ) -> Request:
        """Map merged OpenAI-style kwargs onto Request fields.

        Recognized keys become first-class IR fields; the remainder rides in
        ``extra_wire_params`` so ``with_args(**infer_args)`` keeps working for
        provider-specific params the IR does not model.
        """
        kw = dict(final_kwargs)

        n = self._validate_n(kw)
        kw.pop("n", None)

        stream = kw.pop("stream", True)
        if not isinstance(stream, bool):
            raise TypeError(
                f"stream must be a bool, got {type(stream).__name__}: {stream!r}"
            )

        sampling_fields: dict[str, Any] = {"n": n}
        for src, dst in _SAMPLING_KWARGS.items():
            if src in kw:
                value = kw.pop(src)
                if value is not None:
                    sampling_fields.setdefault(dst, value)
        if "stop" in kw:
            stop = kw.pop("stop")
            if stop is not None:
                sampling_fields["stop"] = (
                    (stop,) if isinstance(stop, str) else tuple(stop)
                )
        if "stop_token_ids" in kw:
            stop_ids = kw.pop("stop_token_ids")
            if stop_ids is not None:
                sampling_fields["stop_token_ids"] = tuple(stop_ids)

        # `logprobs` carries both dialects: chat's bool switch and the
        # completions-style int top-k count.
        return_logprobs = False
        top_k = 0
        lp = kw.pop("logprobs", None)
        if isinstance(lp, bool):
            return_logprobs = lp
        elif lp is not None:
            return_logprobs = True
            top_k = int(lp)
        tlp = kw.pop("top_logprobs", None)
        if tlp is not None:
            top_k = int(tlp)

        response_format = kw.pop("response_format", None)
        tools = kw.pop("tools", None)

        reasoning = None
        effort = kw.pop("reasoning_effort", None)
        if effort is not None:
            reasoning = ReasoningParams(effort=cast("Any", effort))

        return Request(
            input=input_,
            sampling=SamplingParams(**sampling_fields),
            return_logprobs=return_logprobs,
            top_k=top_k,
            response_format=response_format,
            tools=tools,
            reasoning=reasoning,
            stream=stream,
            extra_wire_params=kw or None,
        )

    def _build_generate_request(self, prompt: TModelInput, **kwargs) -> Request:
        """Build the Request for :meth:`agenerate` from merged kwargs."""
        final_kwargs = {**self._kwargs, **kwargs}
        return self._kwargs_to_request(self._coerce_input(prompt), final_kwargs)

    def _build_logprobs_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        logprobs: int,
        score_input: bool,
        temperature: float,
        **kwargs,
    ) -> Request:
        """Build the Request for :meth:`alogprobs` from merged kwargs."""
        final_kwargs = {
            **self._kwargs,
            **kwargs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        n = self._validate_n(final_kwargs)
        if n > 1:
            raise ValueError(f"alogprobs only supports n=1; received n={n}")
        req = self._kwargs_to_request(self._coerce_input(prompt), final_kwargs)
        return replace(
            req, return_logprobs=True, top_k=logprobs, score_input=score_input
        )

    # ── Response -> legacy ModelOutput bridge ─────────────────────────────────

    def _response_to_model_output(self, resp: Response) -> ModelOutput:
        """Bridge a Response back to the legacy ``ModelOutput`` shape.

        Deliberate, documented conversions:
        - Input scoring is re-flattened: ``input_scoring.token_logprobs`` are
          concatenated ahead of the sampled ``logprobs``, restoring the legacy
          echo layout that PPL consumers slice at ``usage.input_tokens``.
        - ``top_logprobs`` collapses ``TopKEntry`` tuples to ``{token: logprob}``
          dicts, coalescing duplicate normalized token texts by max (sglang
          byte-level collisions); ``token_id`` is dropped. On input scoring the
          prompt-side top-k is absent by IR design (no consumer reads it).
        - ``reasoning_texts`` carries the single IR reasoning channel (first
          choice) when present.
        """
        logprobs_present = resp.logprobs is not None or resp.input_scoring is not None
        logprobs_tokens: list[str] | None = None
        logprobs: list[float | None] | None = None
        if logprobs_present:
            segments = []
            if resp.input_scoring is not None:
                segments.extend(resp.input_scoring.token_logprobs)
            if resp.logprobs is not None:
                segments.extend(resp.logprobs)
            logprobs_tokens = [t.token for t in segments]
            logprobs = [t.logprob for t in segments]

        top_logprobs: list[dict[str, float]] | None = None
        if resp.top_logprobs is not None:
            top_logprobs = []
            for per_pos in resp.top_logprobs:
                merged: dict[str, float] = {}
                for entry in per_pos:
                    if entry.token not in merged or entry.logprob > merged[entry.token]:
                        merged[entry.token] = entry.logprob
                top_logprobs.append(merged)
            top_logprobs = top_logprobs or None

        usage: ModelUsage | None = None
        if resp.usage is not None:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        reasoning_texts: list[str] | None = None
        if resp.reasoning is not None and resp.reasoning.text:
            reasoning_texts = [resp.reasoning.text]

        return ModelOutput(
            model=self.meta(),
            texts=list(resp.texts),
            finish_reasons=(
                list(resp.finish_reasons) if resp.finish_reasons is not None else None
            ),
            reasoning_texts=reasoning_texts,
            logprobs_tokens=logprobs_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            usage=usage,
            request_params=(
                dict(resp.request_params) if resp.request_params is not None else None
            ),
            response_model=resp.response_model,
            system_fingerprint=resp.system_fingerprint,
        )
