"""Versioned GenerateContent routes and function-schema representations.

Contracts follow Google's v1/v1beta Discovery revision 20260928:
https://generativelanguage.googleapis.com/$discovery/rest?version=v1
https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta
No runtime discovery or schema-key dropping is used. v1beta accepts both
function schema forms; this adapter selects parametersJsonSchema to preserve
the shared IR's JSON Schema. v1 requires the narrower Google Schema form.
Lossless conversion preserves the supported schema's structure and constraints;
annotations are carried unchanged, not promised to affect model generation.
Google explicitly ignores default for validation. Its example documentation
describes root population, not an explicit rejection of nested input examples.

AI-Generated Code - GPT-6 (OpenAI)
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import httpx

from sieval.core.models._shared import copy_json_value
from sieval.core.models.dialect import DialectError
from sieval.core.types import JSONValue


@dataclass(frozen=True)
class GoogleWireContract:
    api_version: Literal["v1", "v1beta"]
    function_schema_field: Literal["parameters", "parametersJsonSchema"]
    unavailable_fields: Mapping[str, frozenset[str]]


_V1_UNAVAILABLE_FIELDS = MappingProxyType(
    {
        "request": frozenset({"cachedContent", "labels"}),
        "generationConfig": frozenset(
            {
                "audioTranscriptionConfig",
                "enableAffectiveDialog",
                "speechConfig",
                "translationConfig",
            }
        ),
        "tool": frozenset({"computerUse", "mcpServers"}),
        "functionDeclaration": frozenset(
            {"behavior", "parametersJsonSchema", "responseJsonSchema"}
        ),
        "part": frozenset({"audioTranscription", "speechMetadata"}),
        "functionResponse": frozenset({"parts", "scheduling", "willContinue"}),
        "inlineData": frozenset({"displayName"}),
        "fileData": frozenset({"displayName"}),
    }
)
GOOGLE_WIRE_CONTRACTS: Mapping[str, GoogleWireContract] = MappingProxyType(
    {
        "v1": GoogleWireContract("v1", "parameters", _V1_UNAVAILABLE_FIELDS),
        "v1beta": GoogleWireContract(
            "v1beta", "parametersJsonSchema", MappingProxyType({})
        ),
    }
)


def resolve_wire_endpoint(base_url: httpx.URL) -> tuple[GoogleWireContract, httpx.URL]:
    """Resolve a root or versioned gateway URL without mutating its client."""

    if base_url.query or base_url.fragment:
        raise ValueError("Google api_base must not contain query data or fragments")
    path = base_url.path.rstrip("/")
    segments = path.split("/")
    # Only the terminal segment selects the Google API version. Earlier
    # version-like segments may belong to a gateway's own routing scheme.
    if "models" in segments or (
        segments[-1] not in GOOGLE_WIRE_CONTRACTS
        and re.match(r"v[0-9]", segments[-1], re.IGNORECASE)
    ):
        raise ValueError(
            "Google api_base must be an endpoint root or end in /v1 or /v1beta"
        )
    version = "v1beta"
    if segments[-1] in GOOGLE_WIRE_CONTRACTS:
        version = segments.pop()
        path = "/".join(segments)
    return GOOGLE_WIRE_CONTRACTS[version], base_url.copy_with(path=f"{path}/")


_SCHEMA_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "array", "object", "null"}
)
_SCHEMA_STRINGS = frozenset({"title", "description", "format", "pattern"})
_SCHEMA_COUNTS = frozenset(
    {"minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"}
)
_SCHEMA_NUMBERS = frozenset({"minimum", "maximum"})
_SCHEMA_STRING_LISTS = frozenset({"required", "propertyOrdering", "enum"})
_SCHEMA_KEYS = (
    _SCHEMA_STRINGS
    | _SCHEMA_COUNTS
    | _SCHEMA_NUMBERS
    | _SCHEMA_STRING_LISTS
    | {"type", "properties", "items", "anyOf", "nullable", "default", "example"}
)
_INT64_MAX = 2**63 - 1


def validate_function_parameters(raw: object) -> None:
    """Require an explicit object root for function arguments, not results.

    This is SiEval's conservative admission rule for Google's object-schema
    requirement. It does not infer the root type from refs or composition, nor
    rewrite a schema to insert a type. Omitting parameters entirely is valid.
    """

    if (
        not isinstance(raw, Mapping)
        or cast(Mapping[str, object], raw).get("type") != "object"
    ):
        raise DialectError(
            "Google function parameters must describe an object with explicit "
            "type: object"
        )


def _schema_object(raw: object, path: str) -> dict[str, JSONValue]:
    if not isinstance(raw, Mapping):
        raise DialectError(f"{path}: Google v1 Schema requires an object")
    try:
        copied = copy_json_value(raw, path)
    except (TypeError, ValueError) as exc:
        raise DialectError(f"{path}: invalid Google v1 Schema: {exc}") from exc
    assert isinstance(copied, dict)
    unsupported = set(copied) - _SCHEMA_KEYS
    if unsupported:
        raise DialectError(
            f"{path}: Google v1 Schema cannot represent keyword(s) "
            f"{', '.join(sorted(unsupported))}; use /v1beta for JSON Schema"
        )
    return dict(copied)


def _check_scalar(schema: Mapping[str, JSONValue], key: str, path: str) -> None:
    value = schema[key]
    if key in _SCHEMA_STRINGS and not isinstance(value, str):
        raise DialectError(f"{path}.{key}: Google v1 Schema requires a string")
    if key == "nullable" and not isinstance(value, bool):
        raise DialectError(f"{path}.{key}: Google v1 Schema requires a boolean")
    if key in _SCHEMA_NUMBERS:
        try:
            exact = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and float(value) == value
            )
        except OverflowError:
            exact = False
        if not exact:
            raise DialectError(
                f"{path}.{key}: Google v1 Schema requires an exactly representable "
                "finite double"
            )
    if key in _SCHEMA_STRING_LISTS:
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise DialectError(f"{path}.{key}: Google v1 Schema requires strings")
        if len(value) != len(set(cast(list[str], value))):
            raise DialectError(
                f"{path}.{key}: Google v1 Schema requires unique strings"
            )
        if key == "enum" and not value:
            raise DialectError(
                f"{path}.enum: Google v1 Schema requires a non-empty enum"
            )


def lower_function_schema(
    raw: object, path: str = "tools.functions.parameters"
) -> dict[str, JSONValue]:
    """Encode a lossless subset, requiring an explicit type on every node."""

    source = _schema_object(raw, path)
    type_name = source.get("type")
    if not isinstance(type_name, str) or type_name not in _SCHEMA_TYPES:
        raise DialectError(
            f"{path}.type: Google v1 Schema requires a single explicit JSON Schema type"
        )
    result: dict[str, JSONValue] = {"type": type_name.upper()}
    for key, value in source.items():
        if key == "type":
            continue
        if key in _SCHEMA_COUNTS:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _INT64_MAX
            ):
                raise DialectError(
                    f"{path}.{key}: Google v1 Schema requires a non-negative int64"
                )
            result[key] = str(value)
        elif key == "properties":
            if not isinstance(value, Mapping):
                raise DialectError(
                    f"{path}.properties: Google v1 Schema requires an object"
                )
            property_schemas: dict[str, JSONValue] = {}
            for name, child in value.items():
                assert isinstance(name, str)  # _schema_object checked JSON keys.
                property_schemas[name] = lower_function_schema(
                    child, f"{path}.properties.{name}"
                )
            result[key] = property_schemas
        elif key == "items":
            result[key] = lower_function_schema(value, f"{path}.items")
        elif key == "anyOf":
            if not isinstance(value, list) or not value:
                raise DialectError(
                    f"{path}.anyOf: Google v1 Schema requires a non-empty array"
                )
            result[key] = [
                lower_function_schema(child, f"{path}.anyOf[{index}]")
                for index, child in enumerate(value)
            ]
        else:
            _check_scalar(source, key, path)
            if key == "enum" and type_name != "string":
                raise DialectError(
                    f"{path}.enum: Google v1 Schema only maps string enums"
                )
            result[key] = value
    return result


def inspect_function_schema(
    raw: object, path: str = "final function parameters"
) -> dict[str, JSONValue]:
    """Decode the actual v1 wire form independently of request lowering.

    The audit compares this result to the original JSON Schema. Calling the
    encoder here would let an encoder mutation also rewrite its own oracle.
    Only schema nodes are traversed; annotation values and property names are
    opaque, even when they happen to contain keys such as ``type``.
    """

    wire = _schema_object(raw, path)
    wire_type = wire.get("type")
    if not isinstance(wire_type, str) or wire_type not in {
        "STRING",
        "NUMBER",
        "INTEGER",
        "BOOLEAN",
        "ARRAY",
        "OBJECT",
        "NULL",
    }:
        raise DialectError(
            f"{path}.type: Google v1 wire Schema requires an uppercase type"
        )
    decoded: dict[str, JSONValue] = {}
    for name, value in wire.items():
        if name == "type":
            decoded[name] = wire_type.lower()
        elif name in _SCHEMA_COUNTS:
            if (
                not isinstance(value, str)
                or re.fullmatch(r"0|[1-9][0-9]{0,18}", value) is None
            ):
                raise DialectError(
                    f"{path}.{name}: Google v1 wire Schema requires an int64 string"
                )
            count = int(value)
            if count > _INT64_MAX:
                raise DialectError(
                    f"{path}.{name}: Google v1 wire Schema int64 is out of range"
                )
            decoded[name] = count
        elif name == "properties":
            if not isinstance(value, Mapping):
                raise DialectError(
                    f"{path}.properties: Google v1 wire Schema requires an object"
                )
            properties: dict[str, JSONValue] = {}
            for property_name, property_schema in value.items():
                assert isinstance(
                    property_name, str
                )  # _schema_object checked JSON keys.
                properties[property_name] = inspect_function_schema(
                    property_schema, f"{path}.properties.{property_name}"
                )
            decoded[name] = properties
        elif name == "items":
            decoded[name] = inspect_function_schema(value, f"{path}.items")
        elif name == "anyOf":
            if not isinstance(value, list) or not value:
                raise DialectError(
                    f"{path}.anyOf: Google v1 wire Schema requires a non-empty array"
                )
            decoded[name] = [
                inspect_function_schema(child, f"{path}.anyOf[{index}]")
                for index, child in enumerate(value)
            ]
        else:
            _check_scalar(wire, name, path)
            if name == "enum" and wire_type != "STRING":
                raise DialectError(
                    f"{path}.enum: Google v1 wire Schema only maps string enums"
                )
            decoded[name] = value
    return decoded


def validate_wire_body(
    body: Mapping[str, JSONValue], contract: GoogleWireContract
) -> None:
    """Validate versioned fields and parameter roots at actual schema locations.

    Free-form JSON (function arguments, results, annotations, JSON Schema)
    must not be searched by key name: a user may legitimately name a property
    ``cachedContent`` or ``parametersJsonSchema``.
    """

    def check(raw: object, shape: str, path: str) -> None:
        if isinstance(raw, Mapping):
            forbidden = contract.unavailable_fields.get(shape, frozenset()) & raw.keys()
            if forbidden:
                raise DialectError(
                    f"{path}: field(s) {', '.join(sorted(forbidden))} are unavailable "
                    f"in Google {contract.api_version}"
                )

    def sequence(raw: object) -> Sequence[object]:
        return (
            raw
            if isinstance(raw, Sequence) and not isinstance(raw, str | bytes)
            else ()
        )

    check(body, "request", "request")
    check(body.get("generationConfig"), "generationConfig", "generationConfig")
    for index, tool in enumerate(sequence(body.get("tools"))):
        check(tool, "tool", f"tools[{index}]")
        if not isinstance(tool, Mapping):
            continue
        tool_mapping = cast(Mapping[str, JSONValue], tool)
        for declaration in sequence(tool_mapping.get("functionDeclarations")):
            check(
                declaration,
                "functionDeclaration",
                f"tools[{index}].functionDeclarations",
            )
            if not isinstance(declaration, Mapping):
                continue
            declaration_mapping = cast(Mapping[str, JSONValue], declaration)
            for field in ("parameters", "response"):
                if field in declaration and f"{field}JsonSchema" in declaration:
                    raise DialectError(
                        f"{field} and {field}JsonSchema are mutually exclusive"
                    )
                if field in declaration:
                    schema = inspect_function_schema(
                        declaration_mapping[field], f"function.{field}"
                    )
                    if field == "parameters":
                        validate_function_parameters(schema)
            if "parametersJsonSchema" in declaration:
                validate_function_parameters(
                    declaration_mapping["parametersJsonSchema"]
                )
    for content in (*sequence(body.get("contents")), body.get("systemInstruction")):
        if not isinstance(content, Mapping):
            continue
        content_mapping = cast(Mapping[str, JSONValue], content)
        for part in sequence(content_mapping.get("parts")):
            if not isinstance(part, Mapping):
                continue
            check(part, "part", "contents.parts")
            part_mapping = cast(Mapping[str, JSONValue], part)
            for field in ("functionResponse", "inlineData", "fileData"):
                check(part_mapping.get(field), field, f"contents.parts.{field}")
