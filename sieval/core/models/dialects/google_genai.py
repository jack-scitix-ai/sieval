"""Google Gemini native ``generateContent`` dialect for the model IR.

The adapter speaks the public REST protocol through the provider-neutral
``async_http_json`` connection.  Request leaves are translated only where the
shared IR has a lossless Google representation; provider-only controls remain
audited dialect options and ambiguous response shapes fail loudly.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast
from urllib.parse import quote

import httpx

from sieval.core.models._shared import copy_json_value
from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    Capability,
    CapabilityKey,
    DialectCapabilityBinding,
    DialectCapabilityDecision,
    HostedToolsOptions,
    MultimodalInputOptions,
    ReasoningOptions,
    Supported,
    TopLogprobsOptions,
    Unsupported,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.deployment import BINDING_RESOURCE_KEYS
from sieval.core.models.dialect import (
    DialectError,
    Guarantee,
    OutputContract,
    OutputContractError,
    OutputRule,
    PreparedRequest,
    RequestAudit,
    RuntimePlanView,
    WireEvidence,
    WireObservation,
    active_request_leaves,
    validate_reasoning,
    validate_request_invariants,
    validate_runtime_binding_plan,
    validate_structured_output,
    validate_tool_calls,
    validate_top_logprobs,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    Citation,
    FunctionToolCall,
    GroundingChunk,
    GroundingMetadata,
    ImagePart,
    ReasoningOutput,
    Request,
    Response,
    ServerToolUse,
    StructuredOutput,
    TextPart,
    TokenLogprob,
    ToolCallPart,
    ToolResultPart,
    TopKEntry,
    UsageStats,
)
from sieval.core.types import JSONValue

_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
_HOSTED_TOOL_KINDS = frozenset({"google_search", "url_context"})
_IMAGE_MEDIA_TYPES = frozenset(
    {
        "image/avif",
        "image/gif",
        "image/heic",
        "image/heif",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)
_OPAQUE_VERSION = 1
_IR_OWNED_KEYS = BINDING_RESOURCE_KEYS | frozenset(
    {
        "model",
        "contents",
        "systemInstruction",
        "generationConfig",
        "tools",
        "toolConfig",
        "cachedContent",
        "serviceTier",
        "store",
        "stream",
    }
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, JSONValue]],
) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, path: str) -> JSONValue:
    try:
        return cast(
            JSONValue,
            json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError(f"{path} is not strict JSON") from exc


def _tuple_validator(item_type: type, channel: str) -> Callable[[object], None]:
    def validate(value: object) -> None:
        if not isinstance(value, tuple) or not all(
            isinstance(item, item_type) for item in value
        ):
            raise OutputContractError(f"{channel} channel has invalid shape")

    return validate


def _validate_top_logprobs_config(options: object) -> None:
    assert isinstance(options, TopLogprobsOptions)
    if options.minimum > 20:
        raise ValueError("google_genai supports at most 20 top logprobs")


def _validate_reasoning_config(options: object) -> None:
    assert isinstance(options, ReasoningOptions)
    if options.summary not in {None, "none", "auto"}:
        raise ValueError("google_genai supports only reasoning summary 'auto'")
    if options.effort is not None and options.effort.lower() not in _THINKING_LEVELS:
        raise ValueError(f"unsupported Google thinking level {options.effort!r}")


def _validate_hosted_tools_config(options: object) -> None:
    assert isinstance(options, HostedToolsOptions)
    unsupported = set(options.kinds) - _HOSTED_TOOL_KINDS
    if unsupported:
        raise ValueError(
            "google_genai supports only google_search and url_context hosted tools; "
            f"unsupported kinds: {sorted(unsupported)}"
        )


def _validate_multimodal_config(options: object) -> None:
    assert isinstance(options, MultimodalInputOptions)
    unsupported = set(options.modalities) - {"image"}
    if unsupported:
        raise ValueError(f"google_genai does not support modalities {unsupported!r}")


def _decisions() -> Mapping[CapabilityKey, DialectCapabilityDecision]:
    decisions: dict[CapabilityKey, DialectCapabilityDecision] = {
        "input_scoring": Unsupported(
            "Gemini generateContent cannot score input tokens"
        ),
        "sampled_logprobs": Supported(
            DialectCapabilityBinding(
                "sampled_logprobs",
                request_leaves=("scoring.sampled_logprobs",),
                response_channels=("logprobs",),
            )
        ),
        "top_logprobs": Supported(
            DialectCapabilityBinding(
                "top_logprobs",
                request_leaves=("scoring.top_logprobs",),
                response_channels=("top_logprobs",),
                _config_validator=_validate_top_logprobs_config,
            )
        ),
        "reasoning": Supported(
            DialectCapabilityBinding(
                "reasoning",
                request_leaves=(
                    "reasoning.effort",
                    "reasoning.budget_tokens",
                    "reasoning.summary",
                ),
                response_channels=("reasoning",),
                _config_validator=_validate_reasoning_config,
            )
        ),
        "function_tools": Supported(
            DialectCapabilityBinding(
                "function_tools",
                request_leaves=("tools.functions", "tools.choice"),
                response_channels=("tool_calls",),
            )
        ),
        "hosted_tools": Supported(
            DialectCapabilityBinding(
                "hosted_tools",
                request_leaves=("tools.hosted",),
                response_channels=("server_tool_uses",),
                _config_validator=_validate_hosted_tools_config,
            )
        ),
        "structured_output": Supported(
            DialectCapabilityBinding(
                "structured_output",
                request_leaves=(
                    "structured_output.format",
                    "structured_output.schema",
                    "structured_output.name",
                    "structured_output.strict",
                ),
                response_channels=("structured_output",),
            )
        ),
        "stateful_session": Unsupported(
            "Gemini responseId is not a request continuation handle"
        ),
        "opaque_continuation": Supported(
            DialectCapabilityBinding(
                "opaque_continuation",
                request_leaves=("session.opaque_continuation",),
                response_channels=("reasoning",),
            )
        ),
        "multimodal_input": Supported(
            DialectCapabilityBinding(
                "multimodal_input",
                request_leaves=(
                    "input.modality.image",
                    "input.modality.image.media_type",
                ),
                _config_validator=_validate_multimodal_config,
            )
        ),
        "prefill": Unsupported("Gemini has no independent assistant-prefill field"),
        "fim": Unsupported("Gemini generateContent has no completion suffix field"),
    }
    if set(decisions) != set(CAPABILITY_KEYS):
        raise AssertionError("Google GenAI capability row is incomplete")
    return MappingProxyType(decisions)


CAPABILITY_DECISIONS = _decisions()

OUTPUT_CONTRACT = OutputContract(
    {
        "reasoning": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_reasoning),
        "tool_calls": OutputRule(Guarantee.BEST_EFFORT, validate_tool_calls),
        "server_tool_uses": OutputRule(
            Guarantee.BEST_EFFORT,
            _tuple_validator(ServerToolUse, "server_tool_uses"),
        ),
        "structured_output": OutputRule(
            Guarantee.PRESENT_OR_ERROR, validate_structured_output
        ),
        "logprobs": OutputRule(
            Guarantee.PRESENT_OR_ERROR,
            _tuple_validator(TokenLogprob, "logprobs"),
        ),
        "top_logprobs": OutputRule(Guarantee.PRESENT_OR_ERROR, validate_top_logprobs),
        "input_scoring": OutputRule(Guarantee.NEVER),
        "citations": OutputRule(
            Guarantee.BEST_EFFORT, _tuple_validator(Citation, "citations")
        ),
        "grounding": OutputRule(Guarantee.BEST_EFFORT),
        "session_id": OutputRule(Guarantee.NEVER),
        "usage": OutputRule(Guarantee.BEST_EFFORT),
    }
)


@dataclass(frozen=True)
class _GoogleContext:
    request: Request


@dataclass(frozen=True)
class _GoogleExecutionContext:
    request: Request
    request_params: Mapping[str, JSONValue]
    input_contents: tuple[Mapping[str, JSONValue], ...]


@dataclass(frozen=True)
class _LegacyPlan:
    dialect_id: str = "google_genai"
    available_capabilities: frozenset[str] = frozenset(
        key
        for key, decision in CAPABILITY_DECISIONS.items()
        if isinstance(decision, Supported)
    )
    capability_minimums: Mapping[str, Mapping[str, JSONValue]] = field(
        default_factory=dict
    )
    required_output_channels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _OperationVerifier:
    model_name: str
    stream: bool
    version_prefix: str

    def verify_prepared(self, prepared: PreparedRequest) -> str | None:
        method = "streamGenerateContent" if self.stream else "generateContent"
        expected = (
            f"{self.version_prefix}models/{quote(self.model_name, safe='')}:{method}"
        )
        if self.stream:
            expected += "?alt=sse"
        if prepared.operation != expected:
            return f"operation {prepared.operation!r} does not match {expected!r}"
        return None


@dataclass(frozen=True)
class _InputVerifier:
    expected_system: Mapping[str, JSONValue] | None
    expected_contents: tuple[JSONValue, ...]

    @classmethod
    def from_request(
        cls, request_input: ChatInput, opaque: str | None
    ) -> "_InputVerifier":
        system, contents = _verification_input_to_wire(request_input, opaque)
        return cls(system, tuple(contents))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if body.get("contents") != list(self.expected_contents):
            return "contents no longer match the provider-neutral input"
        if body.get("systemInstruction") != self.expected_system:
            return "systemInstruction no longer matches system/developer messages"
        return None


def _get(raw: object, name: str, default: object = None) -> object:
    if isinstance(raw, Mapping):
        return cast(Mapping[object, object], raw).get(name, default)
    return getattr(raw, name, default)


def _sequence(raw: object, path: str) -> Sequence[object]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise OutputContractError(f"{path} must be a sequence")
    return cast(Sequence[object], raw)


def _required_string(raw: object, path: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise OutputContractError(f"{path} must be a non-empty string")
    return raw


def _string(raw: object, path: str) -> str:
    if not isinstance(raw, str):
        raise OutputContractError(f"{path} must be a string")
    return raw


def _optional_string(raw: object, path: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise OutputContractError(f"{path} must be a string or None")
    return raw


def _finite_number(raw: object, path: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise OutputContractError(f"{path} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise OutputContractError(f"{path} must be finite")
    return value


def _nonnegative_int(raw: object, path: str, *, default: int = 0) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise OutputContractError(f"{path} must be a non-negative integer")
    return raw


def _optional_nonnegative_int(raw: object, path: str) -> int | None:
    if raw is None:
        return None
    return _nonnegative_int(raw, path)


def _function_declaration(raw: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    data = dict(raw)
    if data.get("type") == "function":
        nested = data.get("function")
        if nested is not None:
            if set(data) != {"type", "function"} or not isinstance(nested, Mapping):
                raise DialectError("invalid OpenAI-shaped function declaration")
            data = dict(nested)
        else:
            data.pop("type")
    allowed = {"name", "description", "parameters"}
    unknown = set(data) - allowed
    if unknown:
        raise DialectError(
            "unsupported Google function declaration field(s): "
            + ", ".join(sorted(unknown))
        )
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise DialectError("Google function declaration requires a name")
    result: dict[str, JSONValue] = {"name": name}
    if "description" in data:
        result["description"] = data["description"]
    if "parameters" in data:
        result["parametersJsonSchema"] = data["parameters"]
    return result


def _tool_choice(raw: JSONValue) -> dict[str, JSONValue]:
    if isinstance(raw, str):
        modes = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
        try:
            return {"functionCallingConfig": {"mode": modes[raw]}}
        except KeyError as exc:
            raise DialectError(f"unsupported Google tool choice {raw!r}") from exc
    if not isinstance(raw, Mapping):
        raise DialectError("Google tool choice must be a string or mapping")
    data = dict(raw)
    nested = data.get("function")
    if set(data) == {"type", "function"} and isinstance(nested, Mapping):
        name = nested.get("name")
    elif set(data) == {"type", "name"} and data.get("type") == "function":
        name = data.get("name")
    else:
        raise DialectError("invalid Google function-specific tool choice")
    if not isinstance(name, str) or not name:
        raise DialectError("Google function-specific tool choice requires a name")
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [name],
        }
    }


def _hosted_tool(spec: object) -> dict[str, JSONValue]:
    kind = getattr(spec, "kind", None)
    config = getattr(spec, "config", None)
    if kind not in _HOSTED_TOOL_KINDS or not isinstance(config, Mapping):
        raise DialectError(f"unsupported Google hosted tool {kind!r}")
    key = "googleSearch" if kind == "google_search" else "urlContext"
    return {key: dict(config)}


def _image_to_part(part: ImagePart) -> dict[str, JSONValue]:
    if part.detail is not None:
        raise DialectError("Google image parts have no per-image detail field")
    if part.url is not None:
        if not part.url:
            raise DialectError("Google image URLs must not be empty")
        data: dict[str, JSONValue] = {"fileUri": part.url}
        if part.media_type is not None:
            data["mimeType"] = part.media_type
        return {"fileData": data}
    if not part.data or not part.media_type:
        raise DialectError(
            "Google inline images require data and an explicit MIME type"
        )
    if part.media_type not in _IMAGE_MEDIA_TYPES:
        raise DialectError(f"unsupported Google image media type {part.media_type!r}")
    return {"inlineData": {"data": part.data, "mimeType": part.media_type}}


def _tool_result_payload(part: ToolResultPart) -> dict[str, JSONValue]:
    value = copy_json_value(part.result, "function response")
    if part.is_error:
        return {"error": value}
    if isinstance(value, dict):
        return dict(value)
    return {"result": value}


def _content_parts(
    message: ChatMessage,
    call_names: dict[str, str],
    answered_calls: set[str],
) -> list[dict[str, JSONValue]]:
    parts: list[dict[str, JSONValue]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            if message.role == "tool":
                raise DialectError("Google tool messages may contain only tool results")
            parts.append({"text": part.text})
        elif isinstance(part, ImagePart):
            if message.role == "tool":
                raise DialectError("Google tool messages may contain only tool results")
            parts.append(_image_to_part(part))
        elif isinstance(part, ToolCallPart):
            if message.role != "assistant":
                raise DialectError(
                    "Google historical function calls require assistant role"
                )
            if not part.call_id or not part.name:
                raise DialectError(
                    "Google historical function calls require id and name"
                )
            if part.call_id in call_names:
                raise DialectError(
                    f"duplicate Google function call id {part.call_id!r}"
                )
            call_names[part.call_id] = part.name
            parts.append(
                {
                    "functionCall": {
                        "id": part.call_id,
                        "name": part.name,
                        "args": copy_json_value(part.arguments, "function arguments"),
                    }
                }
            )
        elif isinstance(part, ToolResultPart):
            if message.role != "tool":
                raise DialectError("Google function results require tool role")
            try:
                name = call_names[part.call_id]
            except KeyError as exc:
                raise DialectError(
                    f"Google function result {part.call_id!r} has no prior call"
                ) from exc
            if part.call_id in answered_calls:
                raise DialectError(
                    f"duplicate Google function result for {part.call_id!r}"
                )
            answered_calls.add(part.call_id)
            parts.append(
                {
                    "functionResponse": {
                        "id": part.call_id,
                        "name": name,
                        "response": _tool_result_payload(part),
                    }
                }
            )
        else:
            raise DialectError(f"unsupported Google content part {type(part).__name__}")
    if not parts:
        raise DialectError("Google contents must not contain an empty message")
    return parts


_OPAQUE_PART_FIELDS = frozenset(
    {
        "text",
        "inlineData",
        "fileData",
        "functionCall",
        "functionResponse",
        "toolCall",
        "toolResponse",
    }
)


def _validate_opaque_contents(raw: object) -> list[dict[str, JSONValue]]:
    if not isinstance(raw, list) or not raw:
        raise DialectError("Google opaque continuation has no contents")
    copied = copy_json_value(raw, "Google opaque continuation contents")
    if not isinstance(copied, list):
        raise AssertionError("copied Google contents must remain a list")
    result: list[dict[str, JSONValue]] = []
    saw_signature = False
    for content_index, content in enumerate(copied):
        if not isinstance(content, dict) or set(content) != {"role", "parts"}:
            raise DialectError(
                f"Google opaque content {content_index} has invalid shape"
            )
        if content["role"] not in {"user", "model"}:
            raise DialectError(
                f"Google opaque content {content_index} has invalid role"
            )
        parts = content["parts"]
        if not isinstance(parts, list) or not parts:
            raise DialectError(f"Google opaque content {content_index} has no parts")
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise DialectError(
                    f"Google opaque part {content_index}.{part_index} is not an object"
                )
            unknown = set(part) - (
                _OPAQUE_PART_FIELDS | {"thought", "thoughtSignature"}
            )
            if unknown:
                raise DialectError(
                    "Google opaque part has unsupported field(s): "
                    + ", ".join(sorted(unknown))
                )
            data_fields = set(part) & _OPAQUE_PART_FIELDS
            if len(data_fields) != 1:
                raise DialectError("Google opaque part must contain one data field")
            if "text" in part and not isinstance(part["text"], str):
                raise DialectError("Google opaque text part must contain a string")
            for key in data_fields - {"text"}:
                if not isinstance(part[key], dict):
                    raise DialectError(f"Google opaque {key} must be an object")
            thought = part.get("thought")
            if thought is not None and not isinstance(thought, bool):
                raise DialectError("Google opaque thought marker must be boolean")
            signature = part.get("thoughtSignature")
            if signature is not None:
                if not isinstance(signature, str) or not signature:
                    raise DialectError(
                        "Google opaque thoughtSignature must be a non-empty string"
                    )
                saw_signature = True
        result.append(content)
    if not saw_signature:
        raise DialectError("Google opaque continuation has no thought signature")
    return result


def _decode_opaque(value: str) -> list[dict[str, JSONValue]]:
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DialectError("invalid Google opaque continuation JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {"version", "contents"}:
        raise DialectError("invalid Google opaque continuation envelope")
    if raw.get("version") != _OPAQUE_VERSION:
        raise DialectError("unsupported Google opaque continuation version")
    return _validate_opaque_contents(raw.get("contents"))


def _call_names_from_contents(
    contents: Sequence[Mapping[str, JSONValue]],
) -> dict[str, str]:
    names: dict[str, str] = {}
    for content in contents:
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            call = part.get("functionCall")
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            name = call.get("name")
            if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
                if call_id in names:
                    raise DialectError(f"duplicate Google function call id {call_id!r}")
                names[call_id] = name
    return names


def _input_to_wire(
    request_input: ChatInput, opaque: str | None
) -> tuple[dict[str, JSONValue] | None, list[JSONValue]]:
    system_parts: list[JSONValue] = []
    opaque_contents = _decode_opaque(opaque) if opaque is not None else []
    contents: list[JSONValue] = list(opaque_contents)
    call_names = _call_names_from_contents(opaque_contents)
    answered_calls: set[str] = set()
    saw_conversation = False
    for message in request_input.messages:
        if message.name is not None:
            raise DialectError("Google contents have no ChatMessage.name field")
        if message.role in {"system", "developer"}:
            if saw_conversation:
                raise DialectError(
                    "Google system/developer messages must precede conversation turns"
                )
            if any(not isinstance(part, TextPart) for part in message.content):
                raise DialectError("Google systemInstruction is text-only")
            parts = _content_parts(message, call_names, answered_calls)
            system_parts.extend(parts)
            continue
        saw_conversation = True
        parts = _content_parts(message, call_names, answered_calls)
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": parts})
    system = {"parts": system_parts} if system_parts else None
    return system, contents


def _verification_input_to_wire(
    request_input: ChatInput, opaque: str | None
) -> tuple[dict[str, JSONValue] | None, list[JSONValue]]:
    """Independently derive the expected input for request-audit verification."""

    opaque_contents = _decode_opaque(opaque) if opaque is not None else []
    contents: list[JSONValue] = [
        copy_json_value(item, "verified Google opaque content")
        for item in opaque_contents
    ]
    system_parts: list[JSONValue] = []
    call_names = _call_names_from_contents(opaque_contents)
    answered: set[str] = set()
    saw_conversation = False

    for message in request_input.messages:
        if message.name is not None:
            raise DialectError("Google contents have no ChatMessage.name field")
        is_instruction = message.role in {"system", "developer"}
        if is_instruction and saw_conversation:
            raise DialectError(
                "Google system/developer messages must precede conversation turns"
            )
        if not is_instruction:
            saw_conversation = True

        expected_parts: list[JSONValue] = []
        for part in message.content:
            if isinstance(part, TextPart):
                if message.role == "tool":
                    raise DialectError(
                        "Google tool messages may contain only tool results"
                    )
                expected_parts.append({"text": part.text})
                continue
            if isinstance(part, ImagePart):
                if message.role == "tool" or is_instruction:
                    raise DialectError("Google instruction/tool messages are text-only")
                if part.detail is not None:
                    raise DialectError(
                        "Google image parts have no per-image detail field"
                    )
                if part.url is not None:
                    file_data: dict[str, JSONValue] = {"fileUri": part.url}
                    if part.media_type is not None:
                        file_data["mimeType"] = part.media_type
                    expected_parts.append({"fileData": file_data})
                else:
                    if not part.data or not part.media_type:
                        raise DialectError(
                            "Google inline images require data and MIME type"
                        )
                    expected_parts.append(
                        {
                            "inlineData": {
                                "data": part.data,
                                "mimeType": part.media_type,
                            }
                        }
                    )
                continue
            if isinstance(part, ToolCallPart):
                if message.role != "assistant":
                    raise DialectError(
                        "Google historical function calls require assistant role"
                    )
                if part.call_id in call_names:
                    raise DialectError(
                        f"duplicate Google function call id {part.call_id!r}"
                    )
                call_names[part.call_id] = part.name
                expected_parts.append(
                    {
                        "functionCall": {
                            "id": part.call_id,
                            "name": part.name,
                            "args": copy_json_value(
                                part.arguments, "verified function arguments"
                            ),
                        }
                    }
                )
                continue
            if isinstance(part, ToolResultPart):
                if message.role != "tool":
                    raise DialectError("Google function results require tool role")
                if part.call_id not in call_names:
                    raise DialectError(
                        f"Google function result {part.call_id!r} has no prior call"
                    )
                if part.call_id in answered:
                    raise DialectError(
                        f"duplicate Google function result for {part.call_id!r}"
                    )
                answered.add(part.call_id)
                raw_result = copy_json_value(
                    part.result, "verified Google function response"
                )
                response: JSONValue
                if part.is_error:
                    response = {"error": raw_result}
                elif isinstance(raw_result, dict):
                    response = raw_result
                else:
                    response = {"result": raw_result}
                expected_parts.append(
                    {
                        "functionResponse": {
                            "id": part.call_id,
                            "name": call_names[part.call_id],
                            "response": response,
                        }
                    }
                )
                continue
            raise DialectError(f"unsupported Google content part {type(part).__name__}")

        if not expected_parts:
            raise DialectError("Google contents must not contain an empty message")
        if is_instruction:
            if any(not isinstance(part, TextPart) for part in message.content):
                raise DialectError("Google systemInstruction is text-only")
            system_parts.extend(expected_parts)
        else:
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": expected_parts})

    system = {"parts": system_parts} if system_parts else None
    return system, contents


def _structured_output(req: Request, text: str) -> StructuredOutput | None:
    if req.structured_output.format is None:
        return None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        return StructuredOutput(cast(JSONValue, value))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OutputContractError("Google structured output is not valid JSON") from exc


def _usage(raw: object) -> UsageStats | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OutputContractError("usageMetadata must be an object")
    prompt = _nonnegative_int(_get(raw, "promptTokenCount"), "usage.promptTokenCount")
    output = _nonnegative_int(
        _get(raw, "candidatesTokenCount"), "usage.candidatesTokenCount"
    )
    reported_total = _optional_nonnegative_int(
        _get(raw, "totalTokenCount"), "usage.totalTokenCount"
    )
    total = reported_total if reported_total is not None else prompt + output
    return UsageStats(
        input_tokens=prompt,
        output_tokens=output,
        total_tokens=total,
        reasoning_tokens=(
            _nonnegative_int(
                _get(raw, "thoughtsTokenCount"), "usage.thoughtsTokenCount"
            )
            if _get(raw, "thoughtsTokenCount") is not None
            else None
        ),
        cached_tokens=(
            _nonnegative_int(
                _get(raw, "cachedContentTokenCount"), "usage.cachedContentTokenCount"
            )
            if _get(raw, "cachedContentTokenCount") is not None
            else None
        ),
        reported_total_tokens=reported_total,
    )


def _logprobs(
    raw: object, path: str, requested_top_k: int
) -> tuple[tuple[TokenLogprob, ...], tuple[tuple[TopKEntry, ...], ...]]:
    chosen = _sequence(_get(raw, "chosenCandidates"), f"{path}.chosenCandidates")
    raw_tops = _get(raw, "topCandidates")
    tops = _sequence(raw_tops, f"{path}.topCandidates") if raw_tops is not None else ()
    if requested_top_k > 0 and len(chosen) != len(tops):
        raise OutputContractError("Google chosen/top logprob positions are misaligned")
    if requested_top_k == 0 and tops and len(chosen) != len(tops):
        raise OutputContractError("Google chosen/top logprob positions are misaligned")
    sampled: list[TokenLogprob] = []
    alternatives: list[tuple[TopKEntry, ...]] = []
    for index, item in enumerate(chosen):
        sampled.append(
            TokenLogprob(
                _required_string(_get(item, "token"), f"{path}.chosen[{index}].token"),
                _finite_number(
                    _get(item, "logProbability"),
                    f"{path}.chosen[{index}].logProbability",
                ),
                _optional_nonnegative_int(
                    _get(item, "tokenId"), f"{path}.chosen[{index}].tokenId"
                ),
            )
        )
        if not tops:
            continue
        entries = _sequence(_get(tops[index], "candidates"), f"{path}.top[{index}]")
        if len(entries) > requested_top_k and requested_top_k > 0:
            raise OutputContractError(
                "Google returned more top logprobs than requested"
            )
        alternatives.append(
            tuple(
                TopKEntry(
                    _required_string(_get(entry, "token"), f"{path}.top.token"),
                    _finite_number(
                        _get(entry, "logProbability"), f"{path}.top.logprob"
                    ),
                    _optional_nonnegative_int(
                        _get(entry, "tokenId"), f"{path}.top.tokenId"
                    ),
                )
                for entry in entries
            )
        )
    return tuple(sampled), tuple(alternatives)


def _citations(raw: object, path: str) -> list[Citation]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    result: list[Citation] = []
    for index, source in enumerate(
        _sequence(_get(raw, "citationSources", []), f"{path}.citationSources")
    ):
        uri = _get(source, "uri")
        if uri is None:
            raise OutputContractError(f"{path}[{index}].uri is absent")
        result.append(Citation(_required_string(uri, f"{path}[{index}].uri")))
    return result


def _url_context_citations(raw: object, path: str) -> list[Citation]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    result: list[Citation] = []
    for index, item in enumerate(
        _sequence(_get(raw, "urlMetadata", []), f"{path}.urlMetadata")
    ):
        url = _get(item, "retrievedUrl")
        if url is None:
            raise OutputContractError(f"{path}.urlMetadata[{index}] omitted its URL")
        result.append(Citation(_required_string(url, f"{path}.urlMetadata[{index}]")))
    return result


def _grounding(raw: object, path: str) -> GroundingMetadata | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise OutputContractError(f"{path} must be an object")
    chunks: list[GroundingChunk] = []
    for index, chunk in enumerate(
        _sequence(_get(raw, "groundingChunks", []), f"{path}.groundingChunks")
    ):
        sources = [
            _get(chunk, name)
            for name in ("web", "retrievedContext", "maps", "image")
            if _get(chunk, name) is not None
        ]
        if len(sources) != 1:
            raise OutputContractError(f"{path}.groundingChunks[{index}] is ambiguous")
        source = sources[0]
        uri = (
            _get(source, "sourceUri", _get(source, "imageUri"))
            if _get(chunk, "image") is not None
            else _get(source, "uri")
        )
        if uri is None:
            raise OutputContractError(
                f"{path}.groundingChunks[{index}] has no representable URI"
            )
        chunks.append(
            GroundingChunk(
                _required_string(uri, f"{path}.groundingChunks[{index}].uri"),
                _optional_string(
                    _get(source, "title"), f"{path}.groundingChunks[{index}].title"
                ),
            )
        )
    entry = _get(raw, "searchEntryPoint")
    rendered = _optional_string(
        _get(entry, "renderedContent") if entry is not None else None,
        f"{path}.searchEntryPoint.renderedContent",
    )
    return GroundingMetadata(tuple(chunks), rendered)


def _opaque_history(
    input_contents: Sequence[Mapping[str, JSONValue]],
    output_content: Mapping[str, JSONValue],
) -> str | None:
    history: list[JSONValue] = [
        copy_json_value(item, "Google input history") for item in input_contents
    ]
    history.append(copy_json_value(output_content, "Google output history"))
    saw_signature = False
    for content in history:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        if any(
            isinstance(part, dict)
            and isinstance(part.get("thoughtSignature"), str)
            and bool(part.get("thoughtSignature"))
            for part in parts
        ):
            saw_signature = True
            break
    if not saw_signature:
        return None
    return json.dumps(
        {"version": _OPAQUE_VERSION, "contents": history},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class GoogleGenAIDialect:
    """Executable adapter for Gemini native generateContent REST endpoints."""

    dialect_id = "google_genai"
    connection_family = "async_http_json"
    capability_decisions = CAPABILITY_DECISIONS
    output_contract = OUTPUT_CONTRACT
    CAPABILITIES = frozenset(
        {
            Capability.Chat,
            Capability.FunctionCalling,
            Capability.ServerTools,
            Capability.Reasoning,
            Capability.ReasoningEffort,
            Capability.SampledLogprobs,
            Capability.StructuredOutput,
            Capability.TopKLogprobs,
        }
    )

    def __init__(self, connection: object, requested_model_id: str):
        if not isinstance(connection, AsyncHTTPJSONConnection):
            raise TypeError("google_genai requires AsyncHTTPJSONConnection")
        if not requested_model_id:
            raise ValueError("requested_model_id must not be empty")
        model_name = requested_model_id.removeprefix("models/")
        if not model_name or "/" in model_name:
            raise ValueError(
                "requested_model_id must be a Gemini model id or models/<id>"
            )
        self._connection = connection
        self._requested_model_id = requested_model_id
        self._model_name = model_name
        base_path = connection.client.base_url.path.rstrip("/")
        self._version_prefix = "" if base_path.endswith("/v1beta") else "v1beta/"

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def validate_request(
        self, req: Request, audit: RequestAudit, plan: RuntimePlanView
    ) -> None:
        audit.require_matches_request(req)
        del plan
        has_named_message = isinstance(req.input, ChatInput) and any(
            message.name is not None for message in req.input.messages
        )
        for path in audit.active_paths:
            if path == "input.chat" and has_named_message:
                audit.rejected(path, "Google contents have no ChatMessage.name field")
            elif path.startswith("input.completion"):
                audit.rejected(path, "google_genai requires ChatInput")
            elif path == "input.modality.image.detail":
                audit.rejected(path, "Google images have no per-image detail field")
            elif path == "scoring.input_scoring":
                audit.rejected(path, "Gemini cannot score input tokens")
            elif path == "sampling.temperature" and not (
                0 <= cast(float, req.sampling.temperature) <= 2
            ):
                audit.rejected(path, "Google temperature must be between 0 and 2")
            elif path == "sampling.top_p" and not (
                0 <= cast(float, req.sampling.top_p) <= 1
            ):
                audit.rejected(path, "Google top_p must be between 0 and 1")
            elif path == "sampling.top_k" and cast(int, req.sampling.top_k) < 1:
                audit.rejected(path, "Google top_k must be positive")
            elif (
                path == "sampling.max_tokens" and cast(int, req.sampling.max_tokens) < 1
            ):
                audit.rejected(path, "Google max_tokens must be positive")
            elif (
                path == "sampling.stop"
                and len(cast(tuple[str, ...], req.sampling.stop)) > 5
            ):
                audit.rejected(path, "Google supports at most five stop sequences")
            elif path == "scoring.top_logprobs" and req.scoring.top_logprobs > 20:
                audit.rejected(path, "Gemini supports at most 20 top logprobs")
            elif path == "reasoning.effort" and (
                not isinstance(req.reasoning.effort, str)
                or req.reasoning.effort.lower() not in _THINKING_LEVELS
            ):
                audit.rejected(path, "unsupported Google thinking level")
            elif path == "reasoning.summary" and req.reasoning.summary not in {
                "none",
                "auto",
            }:
                audit.rejected(path, "Google supports only summary='auto'")
            elif path == "tools.parallel":
                audit.rejected(path, "Google has no per-request parallel tool switch")
            elif path == "tools.choice" and not req.tools.functions:
                audit.rejected(path, "Google tool choice requires function tools")
            elif path == "tools.hosted" and any(
                tool.kind not in _HOSTED_TOOL_KINDS for tool in req.tools.hosted
            ):
                audit.rejected(path, "unsupported Google hosted tool kind")
            elif path == "session.previous_response_id":
                audit.rejected(path, "Gemini responseId is not a continuation handle")
            elif (
                path == "structured_output.format"
                and req.structured_output.format
                not in {
                    "json_object",
                    "json_schema",
                }
            ):
                audit.rejected(path, "unsupported Google structured output format")
            elif path == "structured_output.schema" and (
                req.structured_output.format != "json_schema"
            ):
                audit.rejected(
                    path, "Google response schema requires format='json_schema'"
                )
            elif path == "structured_output.name":
                audit.rejected(path, "Google response JSON schemas have no name field")
            elif path == "structured_output.strict":
                audit.rejected(path, "Google exposes no JSON-schema strict switch")
            elif path.startswith("dialect_options."):
                key = path.removeprefix("dialect_options.")
                if not key:
                    audit.rejected(path, "dialect option keys must be non-empty")
                elif key in _IR_OWNED_KEYS:
                    audit.rejected(path, f"{key!r} must use its provider-neutral field")

    def prepare(self, req: Request, audit: RequestAudit) -> PreparedRequest:
        audit.require_matches_request(req)
        audit.raise_rejections()
        if not isinstance(req.input, ChatInput):
            raise DialectError("google_genai requires ChatInput")
        active = frozenset(audit.active_paths)

        def consume(path: str, *evidence: WireEvidence) -> None:
            if path in active and path not in audit.decisions:
                audit.consumed(path, *evidence)

        opaque = (
            req.session.opaque_continuation.value
            if req.session.opaque_continuation is not None
            else None
        )
        input_verifier = _InputVerifier.from_request(req.input, opaque)
        system, contents = _input_to_wire(req.input, opaque)
        body: dict[str, JSONValue] = {"contents": contents}
        if system is not None:
            body["systemInstruction"] = system
        for path in (
            "input.chat",
            "input.modality.text",
            "input.modality.image",
            "input.modality.image.media_type",
            "input.modality.tool_call",
            "input.modality.tool_result",
            "input.modality.tool_result.is_error",
        ):
            consume(path, input_verifier)
        if opaque is not None:
            consume("session.opaque_continuation", input_verifier)

        generation: dict[str, JSONValue] = {}
        sampling = req.sampling
        fields = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_tokens": "maxOutputTokens",
            "stop": "stopSequences",
            "seed": "seed",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
            "n": "candidateCount",
        }
        for source, destination in fields.items():
            value = getattr(sampling, source)
            default = getattr(type(sampling)(), source)
            if value != default:
                generation[destination] = (
                    list(value) if isinstance(value, tuple) else value
                )
                consume(
                    f"sampling.{source}",
                    WireObservation(("generationConfig", destination)),
                )

        if req.scoring.sampled_logprobs:
            generation["responseLogprobs"] = True
            consume(
                "scoring.sampled_logprobs",
                WireObservation(("generationConfig", "responseLogprobs")),
            )
        if req.scoring.top_logprobs > 0:
            generation["logprobs"] = req.scoring.top_logprobs
            consume(
                "scoring.top_logprobs",
                WireObservation(("generationConfig", "logprobs")),
            )

        thinking: dict[str, JSONValue] = {}
        if req.reasoning.effort is not None:
            thinking["thinkingLevel"] = req.reasoning.effort.upper()
            consume("reasoning.effort", _ThinkingEffortVerifier(req.reasoning.effort))
        if req.reasoning.budget_tokens is not None:
            thinking["thinkingBudget"] = req.reasoning.budget_tokens
            consume(
                "reasoning.budget_tokens",
                WireObservation(
                    ("generationConfig", "thinkingConfig", "thinkingBudget")
                ),
            )
        if req.reasoning.summary == "none":
            audit.noop(
                "reasoning.summary", "omitting includeThoughts requests no thoughts"
            )
        elif req.reasoning.summary == "auto":
            thinking["includeThoughts"] = True
            consume(
                "reasoning.summary",
                _ReasoningSummaryVerifier(),
            )
        if thinking:
            generation["thinkingConfig"] = thinking

        if req.structured_output.format is not None:
            generation["responseMimeType"] = "application/json"
            consume(
                "structured_output.format",
                _StructuredFormatVerifier(req.structured_output.format),
            )
        if req.structured_output.schema is not None:
            generation["responseJsonSchema"] = dict(req.structured_output.schema)
            consume(
                "structured_output.schema",
                WireObservation(("generationConfig", "responseJsonSchema")),
            )
        if generation:
            body["generationConfig"] = generation

        tools: list[JSONValue] = []
        if req.tools.functions:
            tools.append(
                {
                    "functionDeclarations": [
                        _function_declaration(item) for item in req.tools.functions
                    ]
                }
            )
            consume("tools.functions", _FunctionsVerifier(req.tools.functions))
        if req.tools.hosted:
            tools.extend(_hosted_tool(tool) for tool in req.tools.hosted)
            consume("tools.hosted", _HostedToolsVerifier(req.tools.hosted))
        if tools:
            body["tools"] = tools
        if req.tools.choice is not None:
            tool_config = _tool_choice(req.tools.choice)
            allowed = _get(
                _get(tool_config, "functionCallingConfig"),
                "allowedFunctionNames",
                [],
            )
            declared = {
                _get(_function_declaration(item), "name")
                for item in req.tools.functions
            }
            if any(name not in declared for name in _sequence(allowed, "tool choice")):
                raise DialectError("Google tool choice names an undeclared function")
            body["toolConfig"] = tool_config
            consume("tools.choice", _ToolChoiceVerifier(req.tools.choice))
        if req.tools.hosted:
            existing_tool_config = body.get("toolConfig")
            if existing_tool_config is None:
                tool_config: dict[str, JSONValue] = {}
            elif isinstance(existing_tool_config, Mapping):
                tool_config = dict(existing_tool_config)
            else:
                raise AssertionError("Google toolConfig must remain an object")
            tool_config["includeServerSideToolInvocations"] = True
            body["toolConfig"] = tool_config

        options = req.dialect_options
        if options is not None:
            if not options.values:
                audit.noop(
                    "dialect_options", "empty matching options have no wire semantics"
                )
            for key, value in options.values.items():
                path = f"dialect_options.{key}"
                if path in audit.decisions:
                    continue
                body[key] = value
                audit.consumed(path, _RootOptionVerifier(key, value))

        method = "streamGenerateContent" if req.scheduling.stream else "generateContent"
        operation = (
            f"{self._version_prefix}models/{quote(self._model_name, safe='')}:{method}"
        )
        if req.scheduling.stream:
            operation += "?alt=sse"
            consume(
                "scheduling.stream",
                _OperationVerifier(
                    self._model_name,
                    True,
                    self._version_prefix,
                ),
            )
        return PreparedRequest(operation, body, _GoogleContext(req))

    def _lift(self, raw: object, context: _GoogleExecutionContext) -> Response:
        req = context.request
        candidates = _sequence(_get(raw, "candidates"), "response.candidates")
        if len(candidates) != req.sampling.n:
            raise OutputContractError(
                f"expected {req.sampling.n} Google candidates, "
                f"received {len(candidates)}"
            )
        texts: list[str] = []
        finish_reasons: list[str] = []
        reasoning: list[ReasoningOutput | None] = []
        calls: list[FunctionToolCall] = []
        server_calls: dict[str, tuple[str, Mapping[str, JSONValue]]] = {}
        server_results: dict[str, tuple[str, JSONValue]] = {}
        citations: list[Citation] = []
        groundings: list[GroundingMetadata] = []
        sampled: list[TokenLogprob] = []
        top: list[tuple[TopKEntry, ...]] = []
        for index, candidate in enumerate(candidates):
            candidate_index = _get(candidate, "index", index)
            if candidate_index != index:
                raise OutputContractError("Google candidate indexes are not ordered")
            content = _get(candidate, "content")
            if content is None:
                content_mapping: Mapping[str, JSONValue] | None = None
                parts: Sequence[object] = ()
            elif isinstance(content, Mapping):
                content_mapping = cast(Mapping[str, JSONValue], content)
                unknown_content = set(content_mapping) - {"role", "parts"}
                if unknown_content:
                    raise OutputContractError(
                        "Google response content has unsupported field(s): "
                        + ", ".join(sorted(unknown_content))
                    )
                role = content_mapping.get("role")
                if role is not None and role != "model":
                    raise OutputContractError(
                        "Google response content role is not model"
                    )
                parts = _sequence(
                    content_mapping.get("parts"), f"candidates[{index}].parts"
                )
            else:
                raise OutputContractError("Google response content must be an object")
            text_parts: list[str] = []
            thought_text: list[str] = []
            for part_index, raw_part in enumerate(parts):
                if not isinstance(raw_part, Mapping):
                    raise OutputContractError("Google response part must be an object")
                part = cast(Mapping[str, JSONValue], raw_part)
                unknown_part = set(part) - (
                    _OPAQUE_PART_FIELDS | {"thought", "thoughtSignature"}
                )
                if unknown_part:
                    raise OutputContractError(
                        "Google response part has unsupported field(s): "
                        + ", ".join(sorted(unknown_part))
                    )
                thought_marker = part.get("thought")
                if thought_marker is not None and not isinstance(thought_marker, bool):
                    raise OutputContractError(
                        "Google response thought marker must be boolean"
                    )
                signature = part.get("thoughtSignature")
                if signature is not None and (
                    not isinstance(signature, str) or not signature
                ):
                    raise OutputContractError(
                        "Google response thoughtSignature must be a non-empty string"
                    )
                known = [
                    name
                    for name in ("text", "functionCall", "toolCall", "toolResponse")
                    if name in part
                ]
                if part.get("thought") is True:
                    if known != ["text"] or not isinstance(part.get("text"), str):
                        raise OutputContractError(
                            "Google thought part must contain exactly one text field"
                        )
                    thought_text.append(cast(str, part["text"]))
                    continue
                if len(known) != 1:
                    raise OutputContractError(
                        f"candidates[{index}].parts[{part_index}] has unsupported shape"
                    )
                if known[0] == "text":
                    text_parts.append(_string(part["text"], "response text"))
                elif known[0] == "functionCall":
                    if not req.tools.functions:
                        raise OutputContractError(
                            "Google returned an unrequested function call"
                        )
                    raw_call = part["functionCall"]
                    if not isinstance(raw_call, Mapping):
                        raise OutputContractError("functionCall must be an object")
                    call = cast(Mapping[str, object], raw_call)
                    arguments = copy_json_value(
                        _get(call, "args", {}), "functionCall.args"
                    )
                    if not isinstance(arguments, dict):
                        raise OutputContractError("functionCall.args must be an object")
                    calls.append(
                        FunctionToolCall(
                            _required_string(_get(call, "id"), "functionCall.id"),
                            _required_string(_get(call, "name"), "functionCall.name"),
                            arguments,
                        )
                    )
                elif known[0] == "toolCall":
                    if not req.tools.hosted:
                        raise OutputContractError(
                            "Google returned an unrequested server tool call"
                        )
                    raw_item = part["toolCall"]
                    if not isinstance(raw_item, Mapping):
                        raise OutputContractError("toolCall must be an object")
                    item = cast(Mapping[str, object], raw_item)
                    tool_id = _required_string(_get(item, "id"), "toolCall.id")
                    if tool_id in server_calls:
                        raise OutputContractError(
                            f"duplicate Google server tool call id {tool_id!r}"
                        )
                    raw_args = copy_json_value(_get(item, "args", {}), "toolCall.args")
                    if not isinstance(raw_args, dict):
                        raise OutputContractError("toolCall.args must be an object")
                    server_calls[tool_id] = (
                        _required_string(_get(item, "toolType"), "toolCall.toolType"),
                        raw_args,
                    )
                else:
                    if not req.tools.hosted:
                        raise OutputContractError(
                            "Google returned an unrequested server tool response"
                        )
                    raw_item = part["toolResponse"]
                    if not isinstance(raw_item, Mapping):
                        raise OutputContractError("toolResponse must be an object")
                    item = cast(Mapping[str, object], raw_item)
                    tool_id = _required_string(_get(item, "id"), "toolResponse.id")
                    if tool_id in server_results:
                        raise OutputContractError(
                            f"duplicate Google server tool response id {tool_id!r}"
                        )
                    tool_result = copy_json_value(
                        _get(item, "response"), "toolResponse.response"
                    )
                    if not isinstance(tool_result, dict):
                        raise OutputContractError(
                            "toolResponse.response must be an object"
                        )
                    server_results[tool_id] = (
                        _required_string(
                            _get(item, "toolType"), "toolResponse.toolType"
                        ),
                        tool_result,
                    )
            text = "".join(text_parts)
            texts.append(text)
            finish_reason = _required_string(
                _get(candidate, "finishReason"), "candidate.finishReason"
            )
            if finish_reason == "FINISH_REASON_UNSPECIFIED":
                raise OutputContractError("Google candidate is not terminal")
            finish_reasons.append(finish_reason)
            opaque = (
                _opaque_history(context.input_contents, content_mapping)
                if content_mapping is not None and parts
                else None
            )
            reasoning.append(
                ReasoningOutput(
                    text="\n".join(thought_text) or None, opaque_roundtrip=opaque
                )
                if thought_text or opaque is not None
                else None
            )
            citations.extend(
                _citations(_get(candidate, "citationMetadata"), "citationMetadata")
            )
            citations.extend(
                _url_context_citations(
                    _get(candidate, "urlContextMetadata"), "urlContextMetadata"
                )
            )
            grounding = _grounding(
                _get(candidate, "groundingMetadata"), "groundingMetadata"
            )
            if grounding is not None:
                groundings.append(grounding)
            raw_logprobs = _get(candidate, "logprobsResult")
            if req.scoring.sampled_logprobs:
                if raw_logprobs is None:
                    raise OutputContractError("Google reply omitted requested logprobs")
                chosen, alternatives = _logprobs(
                    raw_logprobs,
                    "candidate.logprobsResult",
                    req.scoring.top_logprobs,
                )
                sampled.extend(chosen)
                top.extend(alternatives)
        if req.reasoning.summary == "auto" and any(
            item is None or not item.text for item in reasoning
        ):
            raise OutputContractError(
                "Google reply omitted requested thoughts for one or more candidates"
            )
        if req.sampling.n > 1 and (citations or groundings):
            raise OutputContractError(
                "Google singular citation/grounding metadata cannot represent choices"
            )
        if len(groundings) > 1:
            raise OutputContractError(
                "multiple candidates returned singular grounding metadata"
            )
        text = texts[0] if len(texts) == 1 else ""
        server_uses: list[ServerToolUse] = []
        for tool_id, (tool_type, tool_input) in server_calls.items():
            result_entry = server_results.pop(tool_id, None)
            if result_entry is not None and result_entry[0] != tool_type:
                raise OutputContractError(
                    f"Google server tool type changed for {tool_id!r}"
                )
            server_uses.append(
                ServerToolUse(
                    tool_type=tool_type,
                    tool_use_id=tool_id,
                    input=tool_input,
                    result=None if result_entry is None else result_entry[1],
                )
            )
        if server_results:
            raise OutputContractError(
                "Google returned a server tool response without its call"
            )
        return Response(
            texts=tuple(texts),
            reasoning=tuple(reasoning) if any(reasoning) else None,
            finish_reasons=tuple(finish_reasons),
            tool_calls=tuple(calls) or None,
            server_tool_uses=tuple(server_uses) or None,
            structured_output=_structured_output(req, text),
            logprobs=tuple(sampled) if req.scoring.sampled_logprobs else None,
            top_logprobs=tuple(top) if req.scoring.top_logprobs > 0 else None,
            citations=tuple(citations) or None,
            grounding=groundings[0] if groundings else None,
            usage=_usage(_get(raw, "usageMetadata")),
            request_params=context.request_params,
            response_model=_optional_string(
                _get(raw, "modelVersion"), "response.modelVersion"
            ),
        )

    async def _lift_stream(
        self, response: httpx.Response, context: _GoogleExecutionContext
    ) -> Response:
        chunks: list[object] = []
        data_lines: list[str] = []

        def flush_event() -> None:
            if not data_lines:
                return
            payload = "\n".join(data_lines)
            data_lines.clear()
            chunks.append(_strict_json_loads(payload, "Google stream event"))

        async for line in response.aiter_lines():
            if not line:
                flush_event()
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                raise OutputContractError("Google stream emitted a malformed SSE line")
            if field == "data":
                data_lines.append(value.removeprefix(" "))
            elif field not in {"event", "id", "retry"}:
                raise OutputContractError(
                    f"Google stream emitted unsupported SSE field {field!r}"
                )
        flush_event()
        if not chunks:
            raise OutputContractError("Google stream emitted no response chunks")
        return self._merge_stream(chunks, context)

    def _merge_stream(
        self, chunks: list[object], context: _GoogleExecutionContext
    ) -> Response:
        # Each stream event carries candidate deltas plus cumulative/terminal
        # metadata. Merge only provider-defined additive content and require
        # finish/usage metadata to agree when repeated.
        by_index: dict[int, dict[str, object]] = {}
        usage: object = None
        model_version: object = None
        for chunk_index, chunk in enumerate(chunks):
            if _get(chunk, "usageMetadata") is not None:
                usage = _get(chunk, "usageMetadata")
            if _get(chunk, "modelVersion") is not None:
                current_model = _get(chunk, "modelVersion")
                if model_version is not None and current_model != model_version:
                    raise OutputContractError(
                        "Google stream changed modelVersion between events"
                    )
                model_version = current_model
            for raw_candidate in _sequence(
                _get(chunk, "candidates", []), f"stream[{chunk_index}].candidates"
            ):
                index = _get(
                    raw_candidate,
                    "index",
                    0 if context.request.sampling.n == 1 else None,
                )
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise OutputContractError(
                        "Google stream candidate index is invalid"
                    )
                state = by_index.setdefault(index, {"parts": []})
                content = _get(raw_candidate, "content")
                if content is not None:
                    role = _get(content, "role")
                    if role is not None and role != "model":
                        raise OutputContractError(
                            "Google stream candidate role is not model"
                        )
                    state["parts"] = cast(list[object], state["parts"]) + list(
                        _sequence(_get(content, "parts", []), "stream candidate parts")
                    )
                finish_reason = _get(raw_candidate, "finishReason")
                if finish_reason is not None:
                    previous = state.get("finishReason")
                    if previous is not None and previous != finish_reason:
                        raise OutputContractError(
                            "Google stream changed a candidate finishReason"
                        )
                    state["finishReason"] = finish_reason
                for name in ("citationMetadata", "groundingMetadata"):
                    value = _get(raw_candidate, name)
                    if value is not None:
                        state[name] = value
                raw_logprobs = _get(raw_candidate, "logprobsResult")
                if raw_logprobs is not None:
                    if not isinstance(raw_logprobs, Mapping):
                        raise OutputContractError(
                            "Google stream logprobsResult must be an object"
                        )
                    accumulated_raw = state.get("logprobsResult")
                    if accumulated_raw is None:
                        accumulated: dict[str, object] = {
                            "chosenCandidates": [],
                            "topCandidates": [],
                        }
                        state["logprobsResult"] = accumulated
                    elif isinstance(accumulated_raw, dict):
                        accumulated = cast(dict[str, object], accumulated_raw)
                    else:
                        raise AssertionError("Google logprobs accumulator changed type")
                    for field_name in ("chosenCandidates", "topCandidates"):
                        value = _get(raw_logprobs, field_name)
                        if value is not None:
                            target = accumulated.get(field_name)
                            if not isinstance(target, list):
                                raise AssertionError(
                                    "Google logprobs accumulator changed shape"
                                )
                            cast(list[object], target).extend(
                                _sequence(
                                    value,
                                    f"stream logprobsResult.{field_name}",
                                )
                            )
        candidates: list[JSONValue] = []
        for index in sorted(by_index):
            state = by_index[index]
            candidate: dict[str, JSONValue] = {
                "index": index,
                "content": {
                    "role": "model",
                    "parts": cast(list[JSONValue], state.pop("parts")),
                },
            }
            for key, value in state.items():
                candidate[key] = copy_json_value(value, f"stream.{key}")
            candidates.append(candidate)
        merged: dict[str, JSONValue] = {"candidates": candidates}
        if usage is not None:
            merged["usageMetadata"] = copy_json_value(usage, "stream usage")
        if model_version is not None:
            merged["modelVersion"] = copy_json_value(model_version, "stream model")
        return self._lift(merged, context)

    async def execute(self, prepared: PreparedRequest) -> Response:
        context = prepared.context
        if not isinstance(context, _GoogleContext):
            raise DialectError("google_genai received invalid prepared context")
        expected_suffix = (
            ":streamGenerateContent?alt=sse"
            if context.request.scheduling.stream
            else ":generateContent"
        )
        if not prepared.operation.endswith(expected_suffix):
            raise DialectError("google_genai received the wrong prepared operation")
        body = prepared.thaw_body()
        raw_contents = body.get("contents")
        if not isinstance(raw_contents, list):
            raise DialectError("prepared Google request has invalid contents")
        copied_contents = copy_json_value(raw_contents, "prepared Google contents")
        if not isinstance(copied_contents, list) or not all(
            isinstance(item, dict) for item in copied_contents
        ):
            raise DialectError("prepared Google request has invalid content items")
        headers = {"Content-Type": "application/json"}
        credential = self._connection.credential
        if credential is not None:
            headers["x-goog-api-key"] = credential
        request = self._connection.client.build_request(
            "POST",
            prepared.operation,
            headers=headers,
            json=body,
        )
        actual_body = _strict_json_loads(
            request.content.decode("utf-8"), "final Google request body"
        )
        if not isinstance(actual_body, dict):
            raise DialectError("final Google request body must be an object")
        actual_contents = actual_body.get("contents")
        if not isinstance(actual_contents, list) or not all(
            isinstance(item, dict) for item in actual_contents
        ):
            raise DialectError("final Google request body has invalid contents")
        request_params: dict[str, JSONValue] = {
            "method": request.method,
            "path": request.url.path,
            "body": {
                key: value
                for key, value in actual_body.items()
                if key not in {"contents", "systemInstruction"}
            },
        }
        actual_alt = request.url.params.get("alt")
        if actual_alt is not None:
            request_params["query"] = {"alt": actual_alt}
        execution_context = _GoogleExecutionContext(
            request=context.request,
            request_params=request_params,
            input_contents=tuple(cast(list[Mapping[str, JSONValue]], actual_contents)),
        )
        response = await self._connection.client.send(request)
        response.raise_for_status()
        if context.request.scheduling.stream:
            return await self._lift_stream(response, execution_context)
        payload = _strict_json_loads(response.text, "Google response body")
        return self._lift(payload, execution_context)

    async def arun(self, req: Request) -> Response:
        plan = _LegacyPlan()
        validate_request_invariants(req)
        validate_runtime_binding_plan(plan, req)
        audit = RequestAudit(active_request_leaves(req))
        self.validate_request(req, audit, plan)
        audit.raise_rejections()
        prepared = self.prepare(req, audit)
        audit.finish(prepared)
        response = await self.execute(prepared)
        self.output_contract.validate(plan, req, response)
        return response


@dataclass(frozen=True)
class _RootOptionVerifier:
    key: str
    value: JSONValue

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            copy_json_value(self.value, f"Google dialect option {self.key}"),
        )

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        return (
            None
            if self.key in body and body[self.key] == self.value
            else f"Google dialect option {self.key!r} changed"
        )


@dataclass(frozen=True)
class _ThinkingEffortVerifier:
    effort: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = _get(
            _get(_get(body, "generationConfig"), "thinkingConfig"), "thinkingLevel"
        )
        return None if actual == self.effort.upper() else "thinking level changed"


@dataclass(frozen=True)
class _ReasoningSummaryVerifier:
    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        actual = _get(
            _get(_get(body, "generationConfig"), "thinkingConfig"),
            "includeThoughts",
        )
        return None if actual is True else "visible-thought request changed"


@dataclass(frozen=True)
class _StructuredFormatVerifier:
    format: str

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        generation = _get(body, "generationConfig")
        if _get(generation, "responseMimeType") != "application/json":
            return "structured output MIME type changed"
        has_schema = _get(generation, "responseJsonSchema") is not None
        if has_schema != (self.format == "json_schema"):
            return "structured output schema presence changed"
        return None


@dataclass(frozen=True)
class _FunctionsVerifier:
    functions: tuple[Mapping[str, JSONValue], ...]

    def __post_init__(self) -> None:
        copied: list[Mapping[str, JSONValue]] = []
        for index, function in enumerate(self.functions):
            item = copy_json_value(function, f"Google function {index}")
            if not isinstance(item, dict):
                raise TypeError("Google function declarations must be objects")
            copied.append(item)
        object.__setattr__(self, "functions", tuple(copied))

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        declarations: list[JSONValue] = []
        for function in self.functions:
            source = dict(function)
            nested = source.get("function")
            if source.get("type") == "function" and isinstance(nested, dict):
                source = dict(nested)
            else:
                source.pop("type", None)
            declaration: dict[str, JSONValue] = {"name": source.get("name")}
            if "description" in source:
                declaration["description"] = source["description"]
            if "parameters" in source:
                declaration["parametersJsonSchema"] = source["parameters"]
            declarations.append(declaration)
        expected = {"functionDeclarations": declarations}
        tools = _get(body, "tools")
        return (
            None
            if isinstance(tools, Sequence) and expected in tools
            else "function tools changed"
        )


@dataclass(frozen=True)
class _HostedToolsVerifier:
    hosted: tuple[object, ...]

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        tools = _get(body, "tools")
        expected: list[dict[str, JSONValue]] = []
        for item in self.hosted:
            kind = getattr(item, "kind", None)
            config = getattr(item, "config", None)
            if not isinstance(config, Mapping):
                return "hosted tool config is invalid"
            key = "googleSearch" if kind == "google_search" else "urlContext"
            expected.append({key: dict(config)})
        if not isinstance(tools, Sequence) or not all(
            item in tools for item in expected
        ):
            return "hosted tools changed"
        if (
            _get(_get(body, "toolConfig"), "includeServerSideToolInvocations")
            is not True
        ):
            return "hosted-tool invocation evidence was disabled"
        return None


@dataclass(frozen=True)
class _ToolChoiceVerifier:
    choice: JSONValue

    def verify(self, body: Mapping[str, JSONValue]) -> str | None:
        if isinstance(self.choice, str):
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}.get(self.choice)
            expected: JSONValue = {"functionCallingConfig": {"mode": mode}}
        elif isinstance(self.choice, dict):
            nested = self.choice.get("function")
            name = (
                nested.get("name")
                if isinstance(nested, dict)
                else self.choice.get("name")
            )
            expected = {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [name],
                }
            }
        else:
            return "source tool choice is invalid"
        actual = _get(body, "toolConfig")
        if not isinstance(actual, Mapping):
            return "tool choice changed"
        actual_choice = {
            key: value
            for key, value in actual.items()
            if key != "includeServerSideToolInvocations"
        }
        return None if actual_choice == expected else "tool choice changed"
