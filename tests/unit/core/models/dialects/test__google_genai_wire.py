"""Contract tests for versioned Google routes and lossless Schema conversion.

AI-Generated Code - GPT-5 (OpenAI)
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sieval.core.models.dialect import DialectError
from sieval.core.models.dialects import _google_genai_wire as wire
from sieval.core.models.dialects._google_genai_wire import (
    GOOGLE_WIRE_CONTRACTS,
    inspect_function_schema,
    lower_function_schema,
    resolve_wire_endpoint,
    validate_wire_body,
)


@pytest.fixture
def discovery_contract() -> dict[str, Any]:
    # This is a reduced upstream snapshot, not output from our converter.
    # CI must remain offline and must not derive expectations from local tables.
    path = Path(__file__).parent / "fixtures" / "google_discovery_20260928.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestDiscoveryContract:
    def test_schema_whitelist_and_scalar_types_match_pinned_discovery(
        self, discovery_contract: dict[str, Any]
    ) -> None:
        assert discovery_contract["revision"] == "20260928"
        fields = discovery_contract["schema_properties"]
        assert set(fields) == wire._SCHEMA_KEYS
        assert {
            value.lower()
            for value in fields["type"]["enum"]
            if value != "TYPE_UNSPECIFIED"
        } == wire._SCHEMA_TYPES
        assert {
            name for name, spec in fields.items() if spec.get("format") == "int64"
        } == wire._SCHEMA_COUNTS
        assert {
            name
            for name, spec in fields.items()
            if spec.get("type") == "string"
            and spec.get("format") != "int64"
            and name != "type"
        } == wire._SCHEMA_STRINGS
        assert {
            name for name, spec in fields.items() if spec.get("type") == "number"
        } == wire._SCHEMA_NUMBERS
        assert {
            name
            for name, spec in fields.items()
            if spec.get("type") == "array" and spec.get("items") == {"type": "string"}
        } == wire._SCHEMA_STRING_LISTS

    def test_size_limits_use_discovery_int64_strings(
        self, discovery_contract: dict[str, Any]
    ) -> None:
        for key, spec in discovery_contract["schema_properties"].items():
            if spec.get("format") != "int64":
                continue
            assert spec["type"] == "string"
            type_name = (
                "array"
                if key.endswith("Items")
                else "object"
                if key.endswith("Properties")
                else "string"
            )
            assert lower_function_schema({"type": type_name, key: 1}) == {
                "type": type_name.upper(),
                key: "1",
            }
            assert inspect_function_schema({"type": type_name.upper(), key: "1"}) == {
                "type": type_name,
                key: 1,
            }

    def test_version_differences_match_upstream_field_inventories(
        self, discovery_contract: dict[str, Any]
    ) -> None:
        v1, beta = (
            discovery_contract["versioned_fields"][version]
            for version in ("v1", "v1beta")
        )
        locations = {
            "request": "GenerateContentRequest",
            "generationConfig": "GenerationConfig",
            "tool": "Tool",
            "functionDeclaration": "FunctionDeclaration",
            "part": "Part",
            "functionResponse": "FunctionResponse",
            "inlineData": "Blob",
            "fileData": "FileData",
        }
        assert GOOGLE_WIRE_CONTRACTS["v1"].unavailable_fields == {
            location: frozenset(beta[shape]) - frozenset(v1[shape])
            for location, shape in locations.items()
        }
        assert GOOGLE_WIRE_CONTRACTS["v1beta"].unavailable_fields == {}
        parameters = discovery_contract["function_parameter_fields"]
        assert set(parameters["v1"]) == {"parameters"}
        assert set(parameters["v1beta"]) == {"parameters", "parametersJsonSchema"}
        for version in ("v1", "v1beta"):
            selected = GOOGLE_WIRE_CONTRACTS[version].function_schema_field
            assert selected in parameters[version]


class TestEndpoints:
    @pytest.mark.parametrize(
        ("path", "version", "root"),
        [
            ("", "v1beta", "/"),
            ("/", "v1beta", "/"),
            ("/v1", "v1", "/"),
            ("/v1beta/", "v1beta", "/"),
            ("/gateway/google", "v1beta", "/gateway/google/"),
            ("/gateway/google/v1/", "v1", "/gateway/google/"),
            ("/gateway/v2/google", "v1beta", "/gateway/v2/google/"),
            ("/gateway/v2/google/v1", "v1", "/gateway/v2/google/"),
        ],
    )
    def test_root_and_explicit_version_preserve_gateway_prefix(
        self, path: str, version: str, root: str
    ) -> None:
        url = httpx.URL(f"https://example.test{path}")
        contract, endpoint_root = resolve_wire_endpoint(url)
        assert contract.api_version == version
        assert endpoint_root == httpx.URL(f"https://example.test{root}")
        assert str(url) == str(httpx.URL(f"https://example.test{path}"))

    @pytest.mark.parametrize(
        "path",
        [
            "/v2",
            "/v1alpha/",
            "/gateway/v9beta",
            "/v1/models/gemini-2.5-flash:generateContent",
            "/models/gemini-2.5-flash",
            "/v1?key=secret",
            "/v1#fragment",
        ],
    )
    def test_ambiguous_endpoint_is_rejected(self, path: str) -> None:
        with pytest.raises(ValueError, match="Google api_base"):
            resolve_wire_endpoint(httpx.URL(f"https://example.test{path}"))


class TestFunctionSchema:
    @pytest.mark.parametrize(
        ("source", "wire"),
        [
            ("string", "STRING"),
            ("number", "NUMBER"),
            ("integer", "INTEGER"),
            ("boolean", "BOOLEAN"),
            ("array", "ARRAY"),
            ("object", "OBJECT"),
            ("null", "NULL"),
        ],
    )
    def test_type_enums_use_independent_golden_values(
        self, source: str, wire: str
    ) -> None:
        assert lower_function_schema({"type": source}) == {"type": wire}
        assert inspect_function_schema({"type": wire}) == {"type": source}

    def test_nested_schema_preserves_constraints_and_opaque_annotations(self) -> None:
        annotation = {"type": "not a schema", "parametersJsonSchema": {"$ref": "x"}}
        source = {
            "type": "object",
            "title": "Title",
            "description": "Description",
            "minProperties": 0,
            "maxProperties": 2**63 - 1,
            "required": ["type"],
            "propertyOrdering": ["type"],
            "properties": {
                "type": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 3,
                    "items": {
                        "type": "string",
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "nullable": True,
                        "minLength": 0,
                        "maxLength": 50,
                        "pattern": "^a",
                        "format": "enum",
                        "enum": ["a", "ab"],
                        "default": annotation,
                        "example": annotation,
                    },
                },
                "minimum": {"type": "number", "minimum": -1.5, "maximum": 2**53},
            },
        }
        wire = {
            "type": "OBJECT",
            "title": "Title",
            "description": "Description",
            "minProperties": "0",
            "maxProperties": "9223372036854775807",
            "required": ["type"],
            "propertyOrdering": ["type"],
            "properties": {
                "type": {
                    "type": "ARRAY",
                    "minItems": "0",
                    "maxItems": "3",
                    "items": {
                        "type": "STRING",
                        "anyOf": [{"type": "STRING"}, {"type": "NULL"}],
                        "nullable": True,
                        "minLength": "0",
                        "maxLength": "50",
                        "pattern": "^a",
                        "format": "enum",
                        "enum": ["a", "ab"],
                        "default": annotation,
                        "example": annotation,
                    },
                },
                "minimum": {"type": "NUMBER", "minimum": -1.5, "maximum": 2**53},
            },
        }
        assert lower_function_schema(source) == wire
        assert inspect_function_schema(wire) == source

    @pytest.mark.parametrize("decode", [False, True])
    @pytest.mark.parametrize(
        ("invalid", "message"),
        [
            ({"title": 1}, "requires a string"),
            ({"nullable": 1}, "requires a boolean"),
            ({"minimum": True}, "finite double"),
            ({"maximum": "3"}, "finite double"),
            ({"minimum": 2**53 + 1}, "finite double"),
            ({"minimum": 10**400}, "finite double"),
            ({"minimum": float("inf")}, "invalid Google v1 Schema"),
            ({"enum": [1]}, "requires strings"),
            ({"enum": "a"}, "requires strings"),
            ({"enum": []}, "non-empty enum"),
            ({"required": ["a", "a"]}, "unique strings"),
            ({"properties": []}, "properties.*requires an object"),
            ({"items": []}, "items.*requires an object"),
            ({"anyOf": []}, "anyOf.*non-empty array"),
            ({"anyOf": {}}, "anyOf.*non-empty array"),
            ({"additionalProperties": False}, "cannot represent keyword"),
            ({"default": object()}, "invalid Google v1 Schema"),
            ({1: "bad JSON key"}, "invalid Google v1 Schema"),
        ],
    )
    def test_invalid_node_is_rejected_in_encoder_and_independent_decoder(
        self, decode: bool, invalid: dict[Any, Any], message: str
    ) -> None:
        schema = {"type": "STRING" if decode else "string", **invalid}
        operation = inspect_function_schema if decode else lower_function_schema
        with pytest.raises(DialectError, match=message):
            operation(schema)

    @pytest.mark.parametrize(
        ("schema", "message"),
        [
            ({}, "single explicit JSON Schema type"),
            ({"type": ["string", "null"]}, "single explicit JSON Schema type"),
            ({"type": "STRING"}, "single explicit JSON Schema type"),
            ({"type": "integer", "enum": ["1"]}, "only maps string enums"),
            ({"type": "string", "maxLength": -1}, "non-negative int64"),
            ({"type": "string", "maxLength": True}, "non-negative int64"),
            ({"type": "string", "maxLength": "1"}, "non-negative int64"),
            ({"type": "string", "maxLength": 2**63}, "non-negative int64"),
        ],
    )
    def test_source_schema_rejects_lossy_mapping(
        self, schema: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            lower_function_schema(schema)

    @pytest.mark.parametrize(
        ("schema", "message"),
        [
            ({}, "uppercase type"),
            ({"type": "string"}, "uppercase type"),
            ({"type": "INTEGER", "enum": ["1"]}, "only maps string enums"),
            ({"type": "STRING", "maxLength": 1}, "int64 string"),
            ({"type": "STRING", "maxLength": "-1"}, "int64 string"),
            ({"type": "STRING", "maxLength": "01"}, "int64 string"),
            ({"type": "STRING", "maxLength": "9223372036854775808"}, "out of range"),
        ],
    )
    def test_wire_schema_rejects_noncanonical_types(
        self, schema: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(DialectError, match=message):
            inspect_function_schema(schema)


class TestVersionedBody:
    @pytest.mark.parametrize(
        ("version", "field", "type_name"),
        [
            ("v1", "parameters", "STRING"),
            ("v1beta", "parameters", "STRING"),
            ("v1beta", "parametersJsonSchema", "string"),
        ],
    )
    def test_function_parameter_root_must_describe_an_object(
        self, version: str, field: str, type_name: str
    ) -> None:
        body = {"tools": [{"functionDeclarations": [{field: {"type": type_name}}]}]}
        with pytest.raises(DialectError, match="function parameters.*object"):
            validate_wire_body(body, GOOGLE_WIRE_CONTRACTS[version])

    @pytest.mark.parametrize(
        "schema", [{}, None, False, [], {"type": ["object", "null"]}]
    )
    def test_json_schema_parameter_root_is_explicit_even_in_direct_wire_check(
        self, schema: Any
    ) -> None:
        body = {"tools": [{"functionDeclarations": [{"parametersJsonSchema": schema}]}]}
        with pytest.raises(DialectError, match="function parameters.*object"):
            validate_wire_body(body, GOOGLE_WIRE_CONTRACTS["v1beta"])

    @pytest.mark.parametrize(
        ("version", "field", "type_name"),
        [
            ("v1", "response", "STRING"),
            ("v1beta", "response", "STRING"),
            ("v1beta", "responseJsonSchema", "string"),
        ],
    )
    def test_function_response_schema_is_not_subject_to_parameter_root_rule(
        self, version: str, field: str, type_name: str
    ) -> None:
        body = {"tools": [{"functionDeclarations": [{field: {"type": type_name}}]}]}
        validate_wire_body(body, GOOGLE_WIRE_CONTRACTS[version])

    @pytest.mark.parametrize(
        ("body", "field"),
        [
            ({"cachedContent": "cachedContents/a"}, "cachedContent"),
            ({"labels": {"tag": "value"}}, "labels"),
            ({"generationConfig": {"speechConfig": {}}}, "speechConfig"),
            ({"tools": [{"computerUse": {}}]}, "computerUse"),
            (
                {
                    "tools": [
                        {
                            "functionDeclarations": [
                                {"parametersJsonSchema": {"type": "object"}}
                            ]
                        }
                    ]
                },
                "parametersJsonSchema",
            ),
            (
                {"tools": [{"functionDeclarations": [{"responseJsonSchema": {}}]}]},
                "responseJsonSchema",
            ),
            (
                {"tools": [{"functionDeclarations": [{"behavior": "BLOCKING"}]}]},
                "behavior",
            ),
            (
                {"contents": [{"parts": [{"audioTranscription": {}}]}]},
                "audioTranscription",
            ),
            ({"contents": [{"parts": [{"functionResponse": {"parts": []}}]}]}, "parts"),
            (
                {"contents": [{"parts": [{"inlineData": {"displayName": "a"}}]}]},
                "displayName",
            ),
            (
                {"contents": [{"parts": [{"fileData": {"displayName": "a"}}]}]},
                "displayName",
            ),
            (
                {"systemInstruction": {"parts": [{"speechMetadata": {}}]}},
                "speechMetadata",
            ),
        ],
    )
    def test_beta_only_fields_are_checked_at_their_schema_location(
        self, body: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(DialectError, match=f"{field}.*unavailable.*v1"):
            validate_wire_body(body, GOOGLE_WIRE_CONTRACTS["v1"])
        validate_wire_body(body, GOOGLE_WIRE_CONTRACTS["v1beta"])

    @pytest.mark.parametrize("version", ["v1", "v1beta"])
    def test_user_json_is_not_recursively_interpreted_as_protocol_fields(
        self, version: str
    ) -> None:
        user_json = {"cachedContent": "text", "parametersJsonSchema": {"behavior": 1}}
        body = {
            "generationConfig": {"responseJsonSchema": user_json},
            "contents": [
                {"parts": [{"functionResponse": {"name": "f", "response": user_json}}]},
                {"parts": [{"functionCall": {"name": "f", "args": user_json}}]},
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {"cachedContent": {"type": "STRING"}},
                                "default": user_json,
                            }
                        }
                    ]
                }
            ],
        }
        # Beta supports parameters too, even though SiEval emits parametersJsonSchema.
        validate_wire_body(body, GOOGLE_WIRE_CONTRACTS[version])

    @pytest.mark.parametrize("field", ["parameters", "response"])
    def test_beta_schema_representations_are_mutually_exclusive(
        self, field: str
    ) -> None:
        body = {
            "tools": [
                {
                    "functionDeclarations": [
                        {field: {"type": "OBJECT"}, f"{field}JsonSchema": {}}
                    ]
                }
            ]
        }
        with pytest.raises(DialectError, match="mutually exclusive"):
            validate_wire_body(body, GOOGLE_WIRE_CONTRACTS["v1beta"])

    def test_non_schema_values_do_not_crash_version_specific_check(self) -> None:
        # General shape validation belongs to the dialect's input/feature verifiers.
        body = {
            "tools": [None, {"functionDeclarations": [None]}],
            "contents": [None, {"parts": "not a parts array"}, {"parts": [None]}],
        }
        validate_wire_body(body, GOOGLE_WIRE_CONTRACTS["v1"])
