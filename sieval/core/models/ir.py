"""Provider-agnostic Model IR: Request, Response, and all supporting types.

Design notes
------------
- All dataclasses are frozen (immutable) to allow safe sharing across async tasks.
- Collection fields use ``tuple``, not ``list``, for the same reason.
- ``session_id`` supports stateful multi-turn (the Transport passes
  ``previous_response_id`` / ``previous_interaction_id`` internally; the caller
  never touches opaque state).
- ``ReasoningParams.opaque_roundtrip`` supports stateless multi-turn: the caller
  echoes the value from ``Response.reasoning.opaque_roundtrip`` back into the
  next ``Request.reasoning.opaque_roundtrip``; the Transport re-embeds it into
  the correct wire item (OpenAI ``encrypted_content`` / Anthropic ``signature`` /
  Google ``signature``).

Persistence
-----------
``Response`` is the persisted record schema (coupled to RFC #24's resume
version gate). Every response-side nested type carries ``@sieval_record`` so it
round-trips back into a typed object rather than a plain dict: ``obj_to_dict``
only stamps ``__sieval_mod__``/``__sieval_cls__`` markers on records, and a
``tuple`` of records only rehydrates correctly when the element type is itself a
record. Request-side types (``SamplingParams``, ``ReasoningParams``,
``ServerToolSpec``) are not persisted and are intentionally undecorated.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from dataclasses import dataclass
from typing import Any, Literal

from sieval.core.utils.serialization import sieval_record

# ── Sampling params ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SamplingParams:
    """Provider-agnostic sampling controls."""

    temperature: float | None = None
    top_p: float | None = None
    top_k_sampling: int | None = None  # sampling top-k; distinct from logprobs top_k
    max_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    stop_token_ids: tuple[int, ...] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int = 1


# ── Reasoning controls (five axes) ─────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningParams:
    """Five-axis reasoning control.

    Axis 1 — intensity (transport picks whichever it supports):
      effort:        semantic level (OpenAI / Anthropic new / Google Gemini 3)
      budget_tokens: exact token cap (Anthropic older models / Google Gemini 2.5)

    Axis 2 — execution tier (OpenAI only):
      mode: "standard" | "pro"

    Axis 3 — cross-turn reasoning state:
      context: "current_turn" | "all_turns"  (OpenAI)

    Axis 4 — summary visibility:
      summary: "none" | "auto" | "concise" | "detailed"

    Axis 5 — cross-call total budget (Anthropic agentic loop):
      task_budget: int

    Stateless multi-turn:
      opaque_roundtrip: echo ``Response.reasoning.opaque_roundtrip`` back here on
      the next turn. The Transport re-embeds it into the correct wire format
      (OpenAI ``encrypted_content`` / Anthropic ``ThinkingBlock.signature`` /
      Google ``ThoughtStep.signature``). Do NOT inspect or modify this value.
    """

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    budget_tokens: int | None = None
    mode: Literal["standard", "pro"] | None = None
    context: Literal["current_turn", "all_turns"] | None = None
    summary: Literal["none", "auto", "concise", "detailed"] | None = None
    task_budget: int | None = None
    opaque_roundtrip: str | None = None  # stateless multi-turn round-trip


# ── Server tools ────────────────────────────────────────────────────────────────

ServerToolType = Literal[
    "web_search",
    "web_fetch",
    "code_execution",
    "file_search",
    "computer_use",
    "image_generation",
    "hosted_mcp",
]


@dataclass(frozen=True)
class ServerToolSpec:
    """Spec for a single provider-hosted tool."""

    type: ServerToolType
    config: dict[str, Any] | None = None


# ── Request IR ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Request:
    """Provider-agnostic inference request.

    input:
      str           → Completion modality (base model)
      list[dict]    → Chat modality (list of ``{"role": ..., "content": ...}``)

    session_id:
      Stateful multi-turn. When set, the Transport passes it as
      ``previous_response_id`` (OpenAI) or ``previous_interaction_id`` (Google)
      on each ``arun`` call. The caller never needs to manage opaque state.
      Obtain the value from ``Response.session_id`` of the previous turn.

    extra_wire_params:
      Unrecognized generation kwargs the request builders could not map onto a
      first-class IR field. Each Transport's ``lower()`` merges these into the
      native wire body last (so an explicit IR field always wins). This is the
      escape hatch that keeps ``with_args(**infer_args)`` working end-to-end.
    """

    input: str | list[dict[str, Any]]

    sampling: SamplingParams | None = None

    # logprobs / scoring
    return_logprobs: bool = False
    top_k: int = 0
    score_input: bool = False  # InputScoring: prompt-side logprobs

    # structured output
    response_format: dict[str, Any] | None = None

    # tools
    tools: list[dict[str, Any]] | None = None  # FunctionCalling
    server_tools: tuple[ServerToolSpec, ...] | None = None  # ServerTools

    # reasoning
    reasoning: ReasoningParams | None = None

    # prefill / FIM
    prefix: str | None = None
    suffix: str | None = None

    # stateful multi-turn
    session_id: str | None = None

    # wire scheduling: None → transport default (single-shot). True asks the
    # transport to stream internally and accumulate; the caller still receives
    # one terminal Response. Pure scheduling — never affects response content.
    stream: bool | None = None

    # passthrough for kwargs without a first-class IR field
    extra_wire_params: dict[str, Any] | None = None


# ── Response sub-types ────────────────────────────────────────────────────────


@sieval_record
@dataclass(frozen=True)
class TokenLogprob:
    """Single token with its log-probability.

    ``logprob`` is nullable: the first prompt token in an input-scoring result
    has no preceding context and therefore no log-probability (sglang emits
    ``None``; this matches the legacy ``ModelOutput.logprobs`` ``float | None``
    contract).

    ``token_id`` is always populated for sglang (native triple), and best-effort
    for other transports. Consumers that need ``token_id`` must declare the
    ``SampledLogprobsWithTokenIds`` capability.

    Both nullable fields are declared last with defaults so they survive
    persistence: ``obj_to_dict`` drops ``None`` fields, so a required nullable
    field could not be reconstructed on the path where it is ``None``.
    """

    token: str
    logprob: float | None = None
    token_id: int | None = None


@sieval_record
@dataclass(frozen=True)
class TopKEntry:
    """One candidate in a top-k logprob list.

    ``token_id`` is declared last with a default for the same persistence reason
    as :class:`TokenLogprob`.
    """

    token: str
    logprob: float
    token_id: int | None = None


@sieval_record
@dataclass(frozen=True)
class InputScoringResult:
    """Prompt-side logprobs for BPB / perplexity computation."""

    token_logprobs: tuple[TokenLogprob, ...]
    byte_count: int | None = None
    char_count: int | None = None


@sieval_record
@dataclass(frozen=True)
class ReasoningOutput:
    """Reasoning channel in the response.

    opaque_roundtrip:
      Provider-signed value that must be echoed back in
      ``Request.reasoning.opaque_roundtrip`` on the next turn (stateless mode).
      Source fields by provider:
        OpenAI     → ResponseReasoningItem.encrypted_content
        Anthropic  → ThinkingBlock.signature
        Google     → ThoughtStep.signature
      Do NOT inspect or modify this value.
    """

    text: str | None = None
    opaque_roundtrip: str | None = None
    thinking_tokens: int = 0
    effort_used: str | None = None


@sieval_record
@dataclass(frozen=True)
class ServerToolUse:
    """Record of one provider-hosted tool invocation and its result."""

    tool_type: str
    tool_use_id: str
    input: dict[str, Any]
    result: dict[str, Any] | None = None
    error_code: str | None = None


@sieval_record
@dataclass(frozen=True)
class Citation:
    """Web search source attribution."""

    url: str
    title: str | None = None
    page_age: str | None = None


@sieval_record
@dataclass(frozen=True)
class GroundingChunk:
    """A single source chunk backing a Google Search grounding result."""

    uri: str
    title: str | None = None


@sieval_record
@dataclass(frozen=True)
class GroundingMetadata:
    """Google Search grounding result.

    ``rendered_content`` MUST be preserved and rendered per Google ToS.
    A Transport's ``lift()`` must not drop this field.
    """

    chunks: tuple[GroundingChunk, ...]
    rendered_content: str | None = None  # Google ToS: must be rendered


@sieval_record
@dataclass(frozen=True)
class UsageStats:
    """Token usage. ``reasoning_tokens`` is billed separately on all three cloud
    providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


# ── Response IR ────────────────────────────────────────────────────────────────


@sieval_record
@dataclass(frozen=True)
class Response:
    """Provider-agnostic inference response.

    session_id:
      Populated by the Transport in stateful mode. Pass this value as
      ``Request.session_id`` on the next turn; the Transport will use it as
      ``previous_response_id`` / ``previous_interaction_id``.

    logprobs / top_logprobs empty-vs-absent contract:
      ``None`` means the server sent no logprob channel at all; an empty tuple
      means the channel was present but carried no entries. Anomaly detection
      distinguishes the two (present-but-empty is flagged), so Transports must
      not collapse ``()`` to ``None`` when logprobs were requested.

    Provenance fields (``request_params``, ``response_model``,
    ``system_fingerprint``) record what was actually sent on the wire and what
    the server reported back. They are persisted for reproducibility records
    (``build_model_call_meta``) and never branched on.
    """

    texts: tuple[str, ...]

    reasoning: ReasoningOutput | None = None
    tool_calls: tuple[dict[str, Any], ...] | None = None
    server_tool_uses: tuple[ServerToolUse, ...] | None = None

    logprobs: tuple[TokenLogprob, ...] | None = None
    top_logprobs: tuple[tuple[TopKEntry, ...], ...] | None = None
    input_scoring: InputScoringResult | None = None

    citations: tuple[Citation, ...] | None = None
    grounding: GroundingMetadata | None = None

    # stateful multi-turn: pass to next Request.session_id
    session_id: str | None = None

    # None when the server reported no usage (absence ≠ zeros).
    usage: UsageStats | None = None
    finish_reasons: tuple[str, ...] | None = None

    # provenance: the lowered wire params (prompt/messages excluded) and the
    # server-reported model / fingerprint.
    request_params: dict[str, Any] | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None
