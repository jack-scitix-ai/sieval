"""Tests for the native Google Gemini ``generateContent`` dialect.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import json
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from sieval.core.models.capabilities import (
    CAPABILITY_KEYS,
    DialectCapabilityStatus,
    HostedToolsOptions,
    ReasoningOptions,
    Supported,
    TopLogprobsOptions,
    Unsupported,
)
from sieval.core.models.connection_factory import AsyncHTTPJSONConnection
from sieval.core.models.dialect import (
    DialectError,
    OutputContractError,
    PreparedRequest,
    RequestAudit,
    RequestAuditError,
    active_request_leaves,
)
from sieval.core.models.dialect_registry import DIALECT_SPECS
from sieval.core.models.dialects import google_genai
from sieval.core.models.dialects.google_genai import (
    CAPABILITY_DECISIONS,
    GoogleGenAIDialect,
)
from sieval.core.models.ir import (
    ChatInput,
    ChatMessage,
    CompletionInput,
    DialectOptions,
    HostedToolSpec,
    ImagePart,
    OpaqueContinuation,
    ReasoningParams,
    Request,
    SamplingParams,
    SchedulingParams,
    ScoringParams,
    SessionParams,
    StructuredOutputParams,
    TextPart,
    ToolCallPart,
    ToolParams,
    ToolResultPart,
)


def _chat(*messages: ChatMessage) -> ChatInput:
    return ChatInput(messages or (ChatMessage("user", (TextPart("Hello"),)),))


def _request(**changes: object) -> Request:
    values: dict[str, object] = {"input": _chat()}
    values.update(changes)
    return Request(**cast(Any, values))


def _response(
    parts: list[dict[str, object]] | None = None,
    *,
    finish_reason: object = "STOP",
    usage: object | None = None,
    model_version: object = "gemini-2.5-flash",
    candidate_extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "index": 0,
        "content": {
            "role": "model",
            "parts": parts if parts is not None else [{"text": "Hi"}],
        },
        "finishReason": finish_reason,
    }
    if candidate_extra:
        candidate.update(candidate_extra)
    return {
        "candidates": [candidate],
        "usageMetadata": usage
        if usage is not None
        else {
            "promptTokenCount": 4,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
            "thoughtsTokenCount": 1,
            "cachedContentTokenCount": 2,
        },
        "modelVersion": model_version,
    }


def _connection(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = "https://generativelanguage.googleapis.com/",
    credential: str | None = "google-secret",
) -> AsyncHTTPJSONConnection:
    return AsyncHTTPJSONConnection(
        httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler)),
        credential,
    )


async def _run(
    handler: Callable[[httpx.Request], httpx.Response],
    request: Request | None = None,
    *,
    model: str = "gemini-2.5-flash",
    base_url: str = "https://generativelanguage.googleapis.com/",
    credential: str | None = "google-secret",
):
    connection = _connection(handler, base_url=base_url, credential=credential)
    try:
        return await GoogleGenAIDialect(connection, model).arun(request or _request())
    finally:
        await connection.aclose()


def _sse(*chunks: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    for chunk in chunks:
        payload = json.dumps(chunk, separators=(",", ":"))
        split_at = payload.find(",", len(payload) // 2)
        if split_at < 0:
            lines.extend((f"data: {payload}", ""))
        else:
            lines.extend(
                (
                    f"data: {payload[: split_at + 1]}",
                    f"data:{payload[split_at + 1 :]}",
                    "",
                )
            )
    return "\n".join(lines).encode()


def _no_io(_: httpx.Request) -> httpx.Response:
    raise AssertionError("request rejection must happen before HTTP I/O")


class TestCapabilities:
    def test_decision_row_and_registry_are_complete(self) -> None:
        assert set(CAPABILITY_DECISIONS) == set(CAPABILITY_KEYS)
        assert isinstance(CAPABILITY_DECISIONS["input_scoring"], Unsupported)
        for key in (
            "sampled_logprobs",
            "top_logprobs",
            "reasoning",
            "function_tools",
            "hosted_tools",
            "structured_output",
            "opaque_continuation",
            "multimodal_input",
        ):
            assert isinstance(CAPABILITY_DECISIONS[key], Supported)

        spec = DIALECT_SPECS["google_genai"]
        assert spec.implementation_status.value == "active"
        assert spec.request_seed_support.value == "supported"
        assert spec.capability_outcomes["reasoning"] is (
            DialectCapabilityStatus.SUPPORTED
        )

    @pytest.mark.parametrize(
        ("key", "options", "message"),
        [
            ("top_logprobs", TopLogprobsOptions(21), "20"),
            ("reasoning", ReasoningOptions(summary="detailed"), "summary"),
            ("reasoning", ReasoningOptions(effort="xhigh"), "thinking level"),
            (
                "hosted_tools",
                HostedToolsOptions(("web_search",)),
                "google_search",
            ),
        ],
    )
    def test_capability_domains_are_provider_validated(
        self, key: str, options: object, message: str
    ) -> None:
        decision = cast(Supported, CAPABILITY_DECISIONS[cast(Any, key)])
        with pytest.raises(ValueError, match=message):
            decision.binding.validate_config(cast(Any, options))


class TestRequestLowering:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("base_url", "model", "expected_path"),
        [
            (
                "https://generativelanguage.googleapis.com/",
                "gemini-2.5-flash",
                "/v1beta/models/gemini-2.5-flash:generateContent",
            ),
            (
                "https://generativelanguage.googleapis.com/v1beta/",
                "models/gemini-2.5-flash",
                "/v1beta/models/gemini-2.5-flash:generateContent",
            ),
        ],
    )
    async def test_native_request_uses_final_url_header_and_wire_body(
        self, base_url: str, model: str, expected_path: str
    ) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["request"] = request
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage("system", (TextPart("Be concise"),)),
                ChatMessage("user", (TextPart("Hello"),)),
            ),
            sampling=SamplingParams(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_tokens=64,
                stop=("END",),
                seed=7,
                frequency_penalty=0.1,
                presence_penalty=-0.2,
            ),
        )
        result = await _run(handler, req, model=model, base_url=base_url)

        request = cast(httpx.Request, seen["request"])
        assert request.url.path == expected_path
        assert request.headers["x-goog-api-key"] == "google-secret"
        assert seen["body"] == {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "Be concise"}]},
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 64,
                "stopSequences": ["END"],
                "seed": 7,
                "frequencyPenalty": 0.1,
                "presencePenalty": -0.2,
            },
        }
        assert result.texts == ("Hi",)
        assert result.finish_reasons == ("STOP",)
        assert result.request_params == {
            "method": "POST",
            "path": expected_path,
            "body": {
                "generationConfig": cast(dict[str, object], seen["body"])[
                    "generationConfig"
                ]
            },
        }
        assert "google-secret" not in json.dumps(result.request_params)

    @pytest.mark.anyio
    async def test_missing_credential_does_not_emit_auth_header(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "x-goog-api-key" not in request.headers
            return httpx.Response(200, json=_response())

        await _run(handler, credential=None)

    @pytest.mark.anyio
    async def test_prepared_context_is_detached_from_mutable_request_inputs(
        self,
    ) -> None:
        function_specs: list[Mapping[str, object]] = [{"name": "lookup"}]

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["tools"] == [
                {"functionDeclarations": [{"name": "lookup"}]}
            ]
            return httpx.Response(
                200,
                json=_response(
                    parts=[
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "lookup",
                                "args": {},
                            }
                        }
                    ]
                ),
            )

        connection = _connection(handler)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request(
                tools=ToolParams(functions=cast(Any, function_specs)),
            )
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)

            function_specs.clear()

            result = await dialect.execute(prepared)
        finally:
            await connection.aclose()

        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "lookup"

    @pytest.mark.anyio
    async def test_multimodal_parts_preserve_order(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        TextPart("before"),
                        ImagePart(data="aGVsbG8=", media_type="image/png"),
                        TextPart("after"),
                        ImagePart(url="gs://bucket/image.jpg", media_type="image/jpeg"),
                    ),
                )
            )
        )
        await _run(handler, req)

        body = cast(dict[str, Any], seen["body"])
        assert body["contents"][0]["parts"] == [
            {"text": "before"},
            {"inlineData": {"data": "aGVsbG8=", "mimeType": "image/png"}},
            {"text": "after"},
            {
                "fileData": {
                    "fileUri": "gs://bucket/image.jpg",
                    "mimeType": "image/jpeg",
                }
            },
        ]

    @pytest.mark.anyio
    async def test_url_image_cannot_bypass_image_only_media_types(self) -> None:
        req = _request(
            input=_chat(
                ChatMessage(
                    "user",
                    (
                        ImagePart(
                            url="gs://bucket/document.pdf", media_type="application/pdf"
                        ),
                    ),
                )
            )
        )

        with pytest.raises(DialectError, match="unsupported Google image media type"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_function_result_recovers_name_from_prior_call(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_response())

        req = _request(
            input=_chat(
                ChatMessage(
                    "assistant",
                    (ToolCallPart("call_1", "weather", {"city": "Paris"}),),
                ),
                ChatMessage(
                    "tool",
                    (ToolResultPart("call_1", {"temperature": 20}),),
                ),
                ChatMessage("user", (TextPart("Continue"),)),
            )
        )
        await _run(handler, req)

        assert cast(dict[str, Any], seen["body"])["contents"] == [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "call_1",
                            "name": "weather",
                            "args": {"city": "Paris"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "call_1",
                            "name": "weather",
                            "response": {"temperature": 20},
                        }
                    }
                ],
            },
            {"role": "user", "parts": [{"text": "Continue"}]},
        ]

    @pytest.mark.anyio
    async def test_error_tool_result_is_explicit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][1]["parts"][0]["functionResponse"]["response"] == {
                "error": "failed"
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage("assistant", (ToolCallPart("c", "tool", {}),)),
                    ChatMessage("tool", (ToolResultPart("c", "failed", True),)),
                )
            ),
        )

    @pytest.mark.anyio
    async def test_function_and_hosted_tools_lower_with_observable_invocations(
        self,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["tools"] == [
                {
                    "functionDeclarations": [
                        {
                            "name": "weather",
                            "description": "Get weather",
                            "parametersJsonSchema": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ]
                },
                {"googleSearch": {}},
            ]
            assert body["toolConfig"] == {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["weather"],
                },
                "includeServerSideToolInvocations": True,
            }
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                tools=ToolParams(
                    functions=(
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            },
                        },
                    ),
                    choice={"type": "function", "name": "weather"},
                    hosted=(HostedToolSpec("google_search"),),
                )
            ),
        )

    @pytest.mark.anyio
    async def test_safety_settings_are_audited_root_options(self) -> None:
        settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            }
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["safetySettings"] == settings
            return httpx.Response(200, json=_response())

        result = await _run(
            handler,
            _request(
                dialect_options=DialectOptions(
                    "google_genai", {"safetySettings": settings}
                )
            ),
        )
        assert result.request_params is not None
        assert result.request_params["body"] == {"safetySettings": settings}


class TestPreIORejections:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("req", "message"),
        [
            (_request(input=CompletionInput("prompt")), "ChatInput"),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("first"),)),
                        ChatMessage("system", (TextPart("late"),)),
                    )
                ),
                "must precede",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("developer", (TextPart("higher priority"),)),
                        ChatMessage("user", (TextPart("question"),)),
                    )
                ),
                "developer message authority",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage("user", (TextPart("complete this"),)),
                        ChatMessage("assistant", (TextPart("partial"),)),
                    )
                ),
                "assistant-prefill",
            ),
            (
                _request(
                    input=_chat(ChatMessage("system", (TextPart("instructions"),)))
                ),
                "end with a user or tool-result turn",
            ),
            (
                _request(
                    input=_chat(
                        ChatMessage(
                            "user",
                            (
                                ImagePart(
                                    data="x", media_type="image/png", detail="high"
                                ),
                            ),
                        )
                    )
                ),
                "detail",
            ),
            (
                _request(
                    input=_chat(ChatMessage("tool", (ToolResultPart("missing", {}),)))
                ),
                "no prior call",
            ),
            (
                _request(
                    tools=ToolParams(
                        functions=({"name": "one"},),
                        choice={"type": "function", "name": "two"},
                    )
                ),
                "undeclared",
            ),
            (
                _request(
                    structured_output=StructuredOutputParams(schema={"type": "object"})
                ),
                "requires format",
            ),
            (
                _request(
                    dialect_options=DialectOptions(
                        "google_genai", {"generationConfig": {"temperature": 2}}
                    )
                ),
                "provider-neutral",
            ),
            (
                _request(sampling=SamplingParams(stop=("1", "2", "3", "4", "5", "6"))),
                "five",
            ),
        ],
    )
    async def test_invalid_shapes_fail_before_io(
        self, req: Request, message: str
    ) -> None:
        with pytest.raises((DialectError, RequestAuditError), match=message):
            await _run(_no_io, req)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "sampling",
        [
            SamplingParams(temperature=-0.1),
            SamplingParams(temperature=2.1),
            SamplingParams(top_p=1.1),
            SamplingParams(top_k=0),
            SamplingParams(max_tokens=0),
        ],
    )
    async def test_provider_sampling_domains_fail_before_io(
        self, sampling: SamplingParams
    ) -> None:
        with pytest.raises(RequestAuditError):
            await _run(_no_io, _request(sampling=sampling))

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "req",
        [
            _request(
                sampling=SamplingParams(n=2),
                scoring=ScoringParams(sampled_logprobs=True),
            ),
            _request(
                sampling=SamplingParams(n=2),
                tools=ToolParams(functions=({"name": "f"},)),
            ),
            _request(
                sampling=SamplingParams(n=2),
                tools=ToolParams(hosted=(HostedToolSpec("google_search"),)),
            ),
            _request(
                sampling=SamplingParams(n=2),
                structured_output=StructuredOutputParams(format="json_object"),
            ),
        ],
    )
    async def test_multiple_candidates_reject_singular_channels_before_io(
        self, req: Request
    ) -> None:
        with pytest.raises(ValueError, match="singular response channels"):
            await _run(_no_io, req)


class TestReasoningAndStructuredOutput:
    @pytest.mark.anyio
    async def test_visible_thoughts_and_full_history_round_trip(self) -> None:
        first_body: dict[str, Any] = {}

        def first_handler(request: httpx.Request) -> httpx.Response:
            first_body.update(json.loads(request.content))
            return httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "text": "analysis",
                            "thought": True,
                            "thoughtSignature": "sig_1",
                        },
                        {"text": "answer"},
                    ]
                ),
            )

        first = await _run(
            first_handler,
            _request(reasoning=ReasoningParams(effort="high", summary="auto")),
        )
        assert first.reasoning is not None
        assert first.reasoning[0] is not None
        assert first.reasoning[0].text == "analysis"
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)

        def second_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][:2] == [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": "analysis",
                            "thought": True,
                            "thoughtSignature": "sig_1",
                        },
                        {"text": "answer"},
                    ],
                },
            ]
            assert body["contents"][2] == {
                "role": "user",
                "parts": [{"text": "next"}],
            }
            return httpx.Response(
                200,
                json=_response([{"text": "next answer", "thoughtSignature": "sig_2"}]),
            )

        second = await _run(
            second_handler,
            _request(
                input=_chat(ChatMessage("user", (TextPart("next"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
        )
        assert second.reasoning is not None
        assert second.reasoning[0] is not None
        assert second.reasoning[0].opaque_roundtrip

    @pytest.mark.anyio
    async def test_opaque_history_preserves_system_instruction(self) -> None:
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "text": "analysis",
                            "thought": True,
                            "thoughtSignature": "sig_1",
                        },
                        {"text": "answer"},
                    ]
                ),
            ),
            _request(
                input=_chat(
                    ChatMessage("system", (TextPart("Always answer as JSON"),)),
                    ChatMessage("user", (TextPart("first"),)),
                )
            ),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        envelope = json.loads(opaque)
        assert envelope["systemInstruction"] == {
            "parts": [{"text": "Always answer as JSON"}]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["systemInstruction"] == envelope["systemInstruction"]
            return httpx.Response(
                200,
                json=_response([{"text": "next", "thoughtSignature": "sig_2"}]),
            )

        await _run(
            handler,
            _request(
                input=_chat(ChatMessage("user", (TextPart("next"),))),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
        )

    @pytest.mark.anyio
    async def test_opaque_history_rejects_conflicting_system_instruction(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "systemInstruction": {"parts": [{"text": "original"}]},
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"text": "answer", "thoughtSignature": "signature"}],
                    }
                ],
            }
        )
        req = _request(
            input=_chat(
                ChatMessage("system", (TextPart("replacement"),)),
                ChatMessage("user", (TextPart("next"),)),
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("google_genai", opaque)
            ),
        )

        with pytest.raises(DialectError, match="systemInstruction conflicts"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_requested_summary_must_be_nonempty(self) -> None:
        with pytest.raises(OutputContractError, match="omitted requested thoughts"):
            await _run(
                lambda _: httpx.Response(200, json=_response()),
                _request(reasoning=ReasoningParams(summary="auto")),
            )

    @pytest.mark.anyio
    async def test_requested_summary_rejects_whitespace_only_thoughts(self) -> None:
        with pytest.raises(OutputContractError, match="omitted requested thoughts"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [{"text": " \n\t ", "thought": True}, {"text": "answer"}]
                    ),
                ),
                _request(reasoning=ReasoningParams(summary="auto")),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', "not-json"])
    async def test_structured_output_requires_strict_json(self, text: str) -> None:
        with pytest.raises(OutputContractError, match="valid JSON"):
            await _run(
                lambda _: httpx.Response(200, json=_response([{"text": text}])),
                _request(
                    structured_output=StructuredOutputParams(
                        format="json_schema", schema={"type": "object"}
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_structured_output_uses_response_json_schema(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["generationConfig"] == {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            }
            return httpx.Response(200, json=_response([{"text": '{"answer":42}'}]))

        result = await _run(
            handler,
            _request(
                structured_output=StructuredOutputParams(
                    format="json_schema", schema=schema
                )
            ),
        )
        assert result.structured_output is not None
        assert result.structured_output.value == {"answer": 42}


class TestResponseLifting:
    @pytest.mark.anyio
    async def test_multiple_text_candidates_preserve_choice_order(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "index": 0,
                            "content": {"role": "model", "parts": [{"text": "A"}]},
                            "finishReason": "STOP",
                        },
                        {
                            "index": 1,
                            "content": {"role": "model", "parts": [{"text": "B"}]},
                            "finishReason": "MAX_TOKENS",
                        },
                    ]
                },
            ),
            _request(sampling=SamplingParams(n=2)),
        )

        assert result.texts == ("A", "B")
        assert result.finish_reasons == ("STOP", "MAX_TOKENS")

    @pytest.mark.anyio
    async def test_multiple_candidates_reject_singular_response_metadata(self) -> None:
        with pytest.raises(OutputContractError, match="singular citation"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "A"}],
                                },
                                "finishReason": "STOP",
                                "citationMetadata": {
                                    "citationSources": [{"uri": "https://a.test"}]
                                },
                            },
                            {
                                "index": 1,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "B"}],
                                },
                                "finishReason": "STOP",
                            },
                        ]
                    },
                ),
                _request(sampling=SamplingParams(n=2)),
            )

    @pytest.mark.anyio
    async def test_missing_function_call_id_is_stable_and_replayable(self) -> None:
        parts = [
            {
                "functionCall": {"name": "weather", "args": {"city": "Paris"}},
                "thoughtSignature": "signature",
            }
        ]
        tools = ToolParams(functions=({"name": "weather"},))
        request = _request(tools=tools)
        first = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            request,
        )
        repeated = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            request,
        )

        assert first.tool_calls is not None
        call_id = first.tool_calls[0].call_id
        assert call_id.startswith("google_call_")
        assert repeated.tool_calls is not None
        assert repeated.tool_calls[0].call_id == call_id
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)
        assert repeated.reasoning is not None and repeated.reasoning[0] is not None
        assert repeated.reasoning[0].opaque_roundtrip == opaque
        envelope = json.loads(opaque)
        assert envelope["contents"][-1]["parts"][0]["functionCall"]["id"] == call_id

        changed = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            _request(
                input=_chat(ChatMessage("user", (TextPart("Different input"),))),
                tools=tools,
            ),
        )
        assert changed.tool_calls is not None
        assert changed.tool_calls[0].call_id != call_id

        def handler(http_request: httpx.Request) -> httpx.Response:
            body = json.loads(http_request.content)
            assert body["contents"][-1]["parts"] == [
                {
                    "functionResponse": {
                        "id": call_id,
                        "name": "weather",
                        "response": {"temperature": 20},
                    }
                }
            ]
            return httpx.Response(200, json=_response())

        await _run(
            handler,
            _request(
                input=_chat(
                    ChatMessage(
                        "tool",
                        (ToolResultPart(call_id, {"temperature": 20}),),
                    )
                ),
                session=SessionParams(
                    opaque_continuation=OpaqueContinuation("google_genai", opaque)
                ),
            ),
        )

    @pytest.mark.anyio
    async def test_opaque_history_restores_answered_function_calls(self) -> None:
        opaque = json.dumps(
            {
                "version": 1,
                "contents": [
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"id": "call_1", "name": "f"},
                                "thoughtSignature": "signature",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "id": "call_1",
                                    "name": "f",
                                    "response": {"ok": True},
                                }
                            }
                        ],
                    },
                ],
            }
        )
        req = _request(
            input=_chat(
                ChatMessage("tool", (ToolResultPart("call_1", {"again": True}),))
            ),
            session=SessionParams(
                opaque_continuation=OpaqueContinuation("google_genai", opaque)
            ),
        )

        with pytest.raises(DialectError, match="duplicate Google function result"):
            await _run(_no_io, req)

    @pytest.mark.anyio
    async def test_usage_logprobs_citations_grounding_and_server_tools(self) -> None:
        logprobs = {
            "chosenCandidates": [{"token": "Hi", "logProbability": -0.1, "tokenId": 7}],
            "topCandidates": [
                {
                    "candidates": [
                        {"token": "Hi", "logProbability": -0.1, "tokenId": 7},
                        {"token": "Hey", "logProbability": -1.2, "tokenId": 8},
                    ]
                }
            ],
        }
        candidate_extra = {
            "logprobsResult": logprobs,
            "citationMetadata": {"citationSources": [{"uri": "https://cite"}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://web", "title": "Web"}},
                    {
                        "image": {
                            "sourceUri": "https://image-source",
                            "title": "Image",
                        }
                    },
                ],
                "searchEntryPoint": {"renderedContent": "<div>sources</div>"},
            },
        }
        parts = [
            {
                "toolCall": {
                    "id": "server_1",
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "args": {"q": "weather"},
                }
            },
            {
                "toolResponse": {
                    "id": "server_1",
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "response": {"ok": True},
                }
            },
            {"text": "Hi"},
        ]
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(parts, candidate_extra=candidate_extra),
            ),
            _request(
                scoring=ScoringParams(sampled_logprobs=True, top_logprobs=2),
                tools=ToolParams(hosted=(HostedToolSpec("google_search"),)),
            ),
        )

        assert result.logprobs is not None and result.logprobs[0].token_id == 7
        assert result.top_logprobs is not None
        assert [item.token for item in result.top_logprobs[0]] == ["Hi", "Hey"]
        assert result.citations is not None
        assert result.citations[0].url == "https://cite"
        assert result.grounding is not None
        assert [chunk.uri for chunk in result.grounding.chunks] == [
            "https://web",
            "https://image-source",
        ]
        assert result.server_tool_uses is not None
        assert result.server_tool_uses[0].tool_type == "GOOGLE_SEARCH_WEB"
        assert result.server_tool_uses[0].result == {"ok": True}
        assert result.usage is not None
        assert result.usage.total_tokens == 6
        assert result.usage.reported_total_tokens == 7
        assert result.usage.reasoning_tokens == 1
        assert result.usage.cached_tokens == 2

    @pytest.mark.anyio
    async def test_missing_server_tool_ids_are_synthesized_and_paired(self) -> None:
        parts = [
            {
                "toolCall": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "args": {"q": "weather"},
                },
                "thoughtSignature": "signature",
            },
            {
                "toolResponse": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "response": {"ok": True},
                }
            },
            {"text": "sunny"},
        ]
        result = await _run(
            lambda _: httpx.Response(200, json=_response(parts)),
            _request(tools=ToolParams(hosted=(HostedToolSpec("google_search"),))),
        )

        assert result.server_tool_uses is not None
        tool_id = result.server_tool_uses[0].tool_use_id
        assert tool_id.startswith("google_server_")
        assert result.server_tool_uses[0].result == {"ok": True}
        assert result.reasoning is not None and result.reasoning[0] is not None
        opaque = json.loads(cast(str, result.reasoning[0].opaque_roundtrip))
        opaque_parts = opaque["contents"][-1]["parts"]
        assert opaque_parts[0]["toolCall"]["id"] == tool_id
        assert opaque_parts[1]["toolResponse"]["id"] == tool_id

    @pytest.mark.anyio
    async def test_usage_computes_total_without_claiming_provider_report(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    usage={"promptTokenCount": 3, "candidatesTokenCount": 4}
                ),
            )
        )
        assert result.usage is not None
        assert result.usage.total_tokens == 7
        assert result.usage.reported_total_tokens is None

    @pytest.mark.anyio
    async def test_matching_provider_total_is_not_redundantly_persisted(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    usage={
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 7,
                    }
                ),
            )
        )
        assert result.usage is not None
        assert result.usage.total_tokens == 7
        assert result.usage.reported_total_tokens is None

    @pytest.mark.anyio
    async def test_requested_top_logprobs_require_exact_cardinality(self) -> None:
        logprobs = {
            "chosenCandidates": [{"token": "Hi", "logProbability": -0.1}],
            "topCandidates": [
                {"candidates": [{"token": "Hi", "logProbability": -0.1}]}
            ],
        }
        with pytest.raises(OutputContractError, match="different number"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(candidate_extra={"logprobsResult": logprobs}),
                ),
                _request(
                    scoring=ScoringParams(
                        sampled_logprobs=True,
                        top_logprobs=2,
                    )
                ),
            )

    @pytest.mark.anyio
    async def test_nonempty_text_requires_chosen_logprobs(self) -> None:
        logprobs = {"chosenCandidates": [], "topCandidates": []}
        with pytest.raises(OutputContractError, match="without chosen logprobs"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(candidate_extra={"logprobsResult": logprobs}),
                ),
                _request(scoring=ScoringParams(sampled_logprobs=True)),
            )

    @pytest.mark.anyio
    async def test_duplicate_provider_function_call_ids_fail_loudly(self) -> None:
        with pytest.raises(OutputContractError, match="duplicate Google function call"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {"functionCall": {"id": "same", "name": "f", "args": {}}},
                            {"functionCall": {"id": "same", "name": "g", "args": {}}},
                        ]
                    ),
                ),
                _request(tools=ToolParams(functions=({"name": "f"}, {"name": "g"}))),
            )

    @pytest.mark.anyio
    async def test_provider_cannot_reuse_function_call_id_across_rounds(self) -> None:
        tools = ToolParams(functions=({"name": "f"},))
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "f",
                                "args": {},
                            },
                            "thoughtSignature": "signature_1",
                        }
                    ]
                ),
            ),
            _request(tools=tools),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)

        with pytest.raises(OutputContractError, match="duplicate Google function call"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "functionCall": {
                                    "id": "call_1",
                                    "name": "f",
                                    "args": {"again": True},
                                },
                                "thoughtSignature": "signature_2",
                            }
                        ]
                    ),
                ),
                _request(
                    input=_chat(
                        ChatMessage(
                            "tool",
                            (ToolResultPart("call_1", {"ok": True}),),
                        )
                    ),
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                    tools=tools,
                ),
            )

    @pytest.mark.anyio
    async def test_provider_cannot_reuse_server_tool_id_across_rounds(self) -> None:
        tools = ToolParams(hosted=(HostedToolSpec("google_search"),))
        first = await _run(
            lambda _: httpx.Response(
                200,
                json=_response(
                    [
                        {
                            "toolCall": {
                                "id": "server_1",
                                "toolType": "GOOGLE_SEARCH_WEB",
                                "args": {"q": "first"},
                            },
                            "thoughtSignature": "signature_1",
                        }
                    ]
                ),
            ),
            _request(tools=tools),
        )
        assert first.reasoning is not None and first.reasoning[0] is not None
        opaque = cast(str, first.reasoning[0].opaque_roundtrip)

        with pytest.raises(OutputContractError, match="duplicate Google server tool"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    json=_response(
                        [
                            {
                                "toolCall": {
                                    "id": "server_1",
                                    "toolType": "GOOGLE_SEARCH_WEB",
                                    "args": {"q": "second"},
                                },
                                "thoughtSignature": "signature_2",
                            }
                        ]
                    ),
                ),
                _request(
                    session=SessionParams(
                        opaque_continuation=OpaqueContinuation("google_genai", opaque)
                    ),
                    tools=tools,
                ),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (_response(finish_reason="FINISH_REASON_UNSPECIFIED"), "not terminal"),
            (_response(candidate_extra={"index": 1}), "not ordered"),
            (
                _response(
                    parts=[{"executableCode": {"language": "PYTHON", "code": "1"}}]
                ),
                "unsupported field",
            ),
            (
                _response(
                    parts=[{"functionCall": {"id": "c", "name": "f", "args": {}}}]
                ),
                "unrequested function",
            ),
        ],
    )
    async def test_unsupported_response_shapes_fail_loudly(
        self, payload: dict[str, object], message: str
    ) -> None:
        with pytest.raises(OutputContractError, match=message):
            await _run(lambda _: httpx.Response(200, json=payload))

    @pytest.mark.anyio
    async def test_missing_requested_logprobs_is_an_output_contract_error(self) -> None:
        with pytest.raises(OutputContractError, match="omitted requested logprobs"):
            await _run(
                lambda _: httpx.Response(200, json=_response()),
                _request(scoring=ScoringParams(sampled_logprobs=True)),
            )


class TestStreamingAndAuditEvidence:
    @pytest.mark.anyio
    async def test_execute_rejects_absolute_operation_before_credential_leak(
        self,
    ) -> None:
        connection = _connection(_no_io)
        try:
            dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
            req = _request()
            audit = RequestAudit(active_request_leaves(req))
            dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
            prepared = dialect.prepare(req, audit)
            audit.finish(prepared)
            attacker_operation = (
                "https://attacker.invalid/v1beta/models/"
                "gemini-2.5-flash:generateContent"
            )

            with pytest.raises(DialectError, match="wrong prepared operation"):
                await dialect.execute(
                    PreparedRequest(
                        attacker_operation,
                        prepared.body,
                        prepared.context,
                    )
                )
        finally:
            await connection.aclose()

    @pytest.mark.anyio
    async def test_streaming_uses_stream_operation_and_merges_deltas(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Hel"}],
                                },
                            }
                        ],
                        "modelVersion": "gemini-2.5-flash",
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {"role": "model", "parts": [{"text": "lo"}]},
                                "finishReason": "MAX_TOKENS",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 2,
                            "candidatesTokenCount": 2,
                            "totalTokenCount": 4,
                        },
                        "modelVersion": "gemini-2.5-flash",
                    },
                ),
                headers={"content-type": "text/event-stream"},
            )

        result = await _run(
            handler,
            _request(scheduling=SchedulingParams(stream=True)),
        )
        assert seen["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:streamGenerateContent?alt=sse"
        )
        assert result.texts == ("Hello",)
        assert result.finish_reasons == ("MAX_TOKENS",)
        assert result.usage is not None
        assert result.usage.total_tokens == 4
        assert result.request_params is not None
        assert result.request_params["query"] == {"alt": "sse"}

    @pytest.mark.anyio
    async def test_streaming_does_not_persist_nonterminal_usage(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "Hel"}],
                                },
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 2,
                            "candidatesTokenCount": 1,
                            "totalTokenCount": 3,
                        },
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "lo"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.texts == ("Hello",)
        assert result.usage is None

    @pytest.mark.anyio
    async def test_streaming_preserves_url_context_citations(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "answer"}],
                                },
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "finishReason": "STOP",
                                "urlContextMetadata": {
                                    "urlMetadata": [
                                        {"retrievedUrl": "https://example.test/source"}
                                    ]
                                },
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(
                scheduling=SchedulingParams(stream=True),
                tools=ToolParams(hosted=(HostedToolSpec("url_context"),)),
            ),
        )

        assert result.citations is not None
        assert [item.url for item in result.citations] == [
            "https://example.test/source"
        ]

    @pytest.mark.anyio
    async def test_streaming_accumulates_grounding_chunks_in_event_order(self) -> None:
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "grounded "}],
                                },
                                "groundingMetadata": {
                                    "groundingChunks": [
                                        {
                                            "web": {
                                                "uri": "https://first.test",
                                                "title": "First",
                                            }
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "answer"}],
                                },
                                "finishReason": "STOP",
                                "groundingMetadata": {
                                    "groundingChunks": [
                                        {
                                            "retrievedContext": {
                                                "uri": "https://second.test",
                                                "title": "Second",
                                            }
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(
                scheduling=SchedulingParams(stream=True),
                tools=ToolParams(hosted=(HostedToolSpec("google_search"),)),
            ),
        )

        assert result.grounding is not None
        assert [chunk.uri for chunk in result.grounding.chunks] == [
            "https://first.test",
            "https://second.test",
        ]

    def test_streaming_grounding_metadata_preserves_all_incremental_evidence(
        self,
    ) -> None:
        search_entry_point = {"renderedContent": "<div>sources</div>"}
        retrieval_metadata = {"googleSearchDynamicRetrievalScore": 0.75}
        merged = google_genai._merge_stream_grounding_metadata(
            None,
            {
                "groundingChunks": [{"web": {"uri": "https://first.test"}}],
                "groundingSupports": [{"groundingChunkIndices": [0]}],
                "searchEntryPoint": search_entry_point,
                "retrievalMetadata": retrieval_metadata,
                "googleMapsWidgetContextToken": "maps-token",
            },
            "groundingMetadata",
        )
        merged = google_genai._merge_stream_grounding_metadata(
            merged,
            {
                "groundingChunks": [{"web": {"uri": "https://second.test"}}],
                "groundingSupports": [{"groundingChunkIndices": [1]}],
                "searchEntryPoint": {"renderedContent": "<div>sources</div>"},
                "retrievalMetadata": {"googleSearchDynamicRetrievalScore": 0.75},
                "googleMapsWidgetContextToken": "maps-token",
            },
            "groundingMetadata",
        )

        assert merged == {
            "groundingChunks": [
                {"web": {"uri": "https://first.test"}},
                {"web": {"uri": "https://second.test"}},
            ],
            "groundingSupports": [
                {"groundingChunkIndices": [0]},
                {"groundingChunkIndices": [1]},
            ],
            "searchEntryPoint": search_entry_point,
            "retrievalMetadata": retrieval_metadata,
            "googleMapsWidgetContextToken": "maps-token",
        }

    @pytest.mark.anyio
    async def test_streaming_accepts_identical_repeated_singular_metadata(
        self,
    ) -> None:
        citation_metadata = {"citationSources": [{"uri": "https://citation.test"}]}
        url_context_metadata = {
            "urlMetadata": [{"retrievedUrl": "https://context.test"}]
        }
        first_candidate_metadata = {
            "citationMetadata": citation_metadata,
            "urlContextMetadata": url_context_metadata,
        }
        second_candidate_metadata = {
            "citationMetadata": {"citationSources": [{"uri": "https://citation.test"}]},
            "urlContextMetadata": {
                "urlMetadata": [{"retrievedUrl": "https://context.test"}]
            },
        }
        result = await _run(
            lambda _: httpx.Response(
                200,
                content=_sse(
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "answer"}],
                                },
                                **first_candidate_metadata,
                            }
                        ]
                    },
                    {
                        "candidates": [
                            {
                                "index": 0,
                                "finishReason": "STOP",
                                **second_candidate_metadata,
                            }
                        ]
                    },
                ),
                headers={"content-type": "text/event-stream"},
            ),
            _request(scheduling=SchedulingParams(stream=True)),
        )

        assert result.citations is not None
        assert [item.url for item in result.citations] == [
            "https://citation.test",
            "https://context.test",
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("metadata_name", "first_value", "second_value"),
        [
            (
                "citationMetadata",
                {"citationSources": [{"uri": "https://first.test"}]},
                {"citationSources": [{"uri": "https://second.test"}]},
            ),
            (
                "urlContextMetadata",
                {"urlMetadata": [{"retrievedUrl": "https://first.test"}]},
                {"urlMetadata": [{"retrievedUrl": "https://second.test"}]},
            ),
            (
                "groundingMetadata",
                {"searchEntryPoint": {"renderedContent": "first"}},
                {"searchEntryPoint": {"renderedContent": "second"}},
            ),
        ],
    )
    async def test_streaming_rejects_conflicting_singular_metadata(
        self,
        metadata_name: str,
        first_value: Mapping[str, object],
        second_value: Mapping[str, object],
    ) -> None:
        with pytest.raises(OutputContractError, match="changed .* between events"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=_sse(
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": "answer"}],
                                    },
                                    metadata_name: first_value,
                                }
                            ]
                        },
                        {
                            "candidates": [
                                {
                                    "index": 0,
                                    "finishReason": "STOP",
                                    metadata_name: second_value,
                                }
                            ]
                        },
                    ),
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    @pytest.mark.parametrize("invalid_event", ["junk", [], {}])
    async def test_streaming_rejects_non_response_data_events(
        self, invalid_event: object
    ) -> None:
        terminal = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        content = (
            f"data: {json.dumps(invalid_event)}\n\ndata: {json.dumps(terminal)}\n\n"
        ).encode()

        with pytest.raises(OutputContractError, match="non-empty response object"):
            await _run(
                lambda _: httpx.Response(
                    200,
                    content=content,
                    headers={"content-type": "text/event-stream"},
                ),
                _request(scheduling=SchedulingParams(stream=True)),
            )

    @pytest.mark.anyio
    async def test_streaming_logprobs_are_accumulated(self) -> None:
        def logprob(token: str, token_id: int) -> dict[str, object]:
            return {"token": token, "logProbability": -0.1, "tokenId": token_id}

        def handler(_: httpx.Request) -> httpx.Response:
            chunks: list[dict[str, object]] = []
            for index, token in enumerate(("A", "B")):
                candidate: dict[str, object] = {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": token}]},
                    "logprobsResult": {
                        "chosenCandidates": [logprob(token, index)],
                        "topCandidates": [{"candidates": [logprob(token, index)]}],
                    },
                }
                if index == 1:
                    candidate["finishReason"] = "STOP"
                chunks.append({"candidates": [candidate]})
            return httpx.Response(200, content=_sse(*chunks))

        result = await _run(
            handler,
            _request(
                scheduling=SchedulingParams(stream=True),
                scoring=ScoringParams(sampled_logprobs=True, top_logprobs=1),
            ),
        )
        assert result.logprobs is not None
        assert [item.token for item in result.logprobs] == ["A", "B"]

    def test_input_verifier_detects_lowering_mutation(self) -> None:
        connection = _connection(_no_io)
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
        req = _request(
            input=_chat(ChatMessage("user", (TextPart("before"), TextPart("after"))))
        )
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
        prepared = dialect.prepare(req, audit)
        body = prepared.thaw_body()
        body["contents"] = [{"role": "user", "parts": [{"text": "afterbefore"}]}]

        with pytest.raises(RequestAuditError, match="contents"):
            audit.finish(PreparedRequest(prepared.operation, body, prepared.context))

    def test_operation_verifier_detects_stream_routing_mutation(self) -> None:
        connection = _connection(_no_io)
        dialect = GoogleGenAIDialect(connection, "gemini-2.5-flash")
        req = _request(scheduling=SchedulingParams(stream=True))
        audit = RequestAudit(active_request_leaves(req))
        dialect.validate_request(req, audit, cast(Any, SimpleNamespace()))
        prepared = dialect.prepare(req, audit)
        wrong = PreparedRequest(
            prepared.operation.replace("streamGenerateContent", "generateContent"),
            prepared.body,
            prepared.context,
        )

        with pytest.raises(RequestAuditError, match="operation"):
            audit.finish(wrong)

    @pytest.mark.anyio
    async def test_malformed_sse_fails_loudly(self) -> None:
        with pytest.raises(OutputContractError, match="malformed SSE"):
            await _run(
                lambda _: httpx.Response(200, content=b"not-sse\n\n"),
                _request(scheduling=SchedulingParams(stream=True)),
            )
