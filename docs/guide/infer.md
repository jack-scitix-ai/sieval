# Inference Service Management

SiEval orchestrates inference backends (vLLM, SGLang) with recipe-driven auto-resolve.

## Quick Reference

```bash
sieval infer start /path/to/model     # auto-resolve and launch
sieval infer list                      # show running services
sieval infer show <name>               # detailed service info (includes status/phase/conditions)
sieval infer logs <name> -f            # stream engine logs
sieval infer stop <name>               # graceful shutdown
```

## Starting a Service

```bash
# Auto-resolve: detect architecture, match recipe, launch
sieval infer start /path/to/Qwen3-8B

# Explicit YAML config (recipe auto-resolved from checkpoint if omitted)
sieval infer start config.yaml

# Dry-run: print launch command without executing
sieval infer start /path/to/Qwen3-8B --dry-run

# Pass extra engine arguments after --
sieval infer start /path/to/Qwen3-8B -- --served-model-name my-model

# Detach: return immediately without waiting for ready
sieval infer start /path/to/Qwen3-8B --detach

# Serve a base checkpoint: skip the recipe's instruct-only serving params
sieval infer start /path/to/Qwen3-8B-Base --model-type gen
```

### Model type and the capability layer

A recipe splits its params into a **hardware** layer (dtype, memory,
parallelism, context) and a **capability** layer (reasoning parser, tool-call
parser, tool choice). The capability layer is selected by model type: `chat`
resolves the instruct params, `gen` resolves none — a base checkpoint has no
chat template and no tool-calling surface for those flags to act on.

In YAML mode the type comes from the config: the model's `type:` if declared,
otherwise it is inferred from the tasks pointing at that model (a PPL or CLP
task requires `gen`). `sieval run` does the same. Checkpoint mode has no config
and no task context, so `--model-type` declares it there; unset, it defaults to
`chat`. Passing `--model-type` in YAML mode is rejected rather than silently
overriding the config.

The instruct flags are inert on a base checkpoint — the engine accepts and
ignores them — so this affects what `infer_plans.yaml` records as the params
used, not whether the service starts.

## Handles and engine logs

Each launched service has a handle file under `~/.sieval/handles`:

| Launched by | Handle | Name in `infer list` / `show` / `stop` |
| --- | --- | --- |
| `sieval infer start` | `<model>.json` | `<model>` |
| `sieval run` | `run-<pid>/<model>.json` | `run-<pid>/<model>` |

So `sieval run` never collides with an `infer start` of the same model. A run
removes its handles when it ends; if it was killed, the next `infer list`
cleans them up, except a handle whose engine is still alive, which stays
listed so `infer stop` can reach it.

Engine logs:

| Launched by | Log |
| --- | --- |
| `sieval run` with a `result_dir` | `<result_dir>/logs/<model>/<role>.log` |
| otherwise | `~/.sieval/logs/<model>/<role>-<deployment_id>.log` |

Each launch prunes `~/.sieval/logs/<model>/` to the last N logs per role,
counting the new one; other models' logs and `result_dir` logs are untouched.
Set N with `--keep-logs` (on `infer start` and `run`), else `SIEVAL_KEEP_LOGS`,
else 3. `0` deletes the log when the service stops; a negative N keeps all.

## YAML Infer Configuration

Models with a `path` field (and no `api_base`) or an `infer` section in the YAML config are automatically launched by `sieval run` and stopped after evaluation completes.

### Anthropic Messages

To call the native Anthropic Messages API, select the dialect explicitly and
provide an external endpoint and credential:

```yaml
models:
  claude:
    name: claude-sonnet-4-5
    dialect: anthropic_messages
    api_base: https://api.anthropic.com/v1
    api_key: "<your Anthropic API key>"
    args:
      max_tokens: 1024
```

The dialect sends JSON or SSE requests to `POST /v1/messages`. Every request
must specify `max_tokens`; set a model-wide default under `models.<name>.args`,
as above, or override it for one task under `tasks.<name>.infer_args`.

Anthropic Messages has no per-request `seed`. Deterministic mode therefore does
not inject one for this dialect, and an explicit `seed` is rejected before any
network I/O.

Raw request passthrough is intentionally unavailable. `dialect_options` cannot
inject arbitrary Anthropic body fields or HTTP headers; in particular, there is
currently no way to send the `anthropic-beta` header. Features that require a
beta header are unsupported until they receive an explicit, audited mapping.

The YAML loader does not expand environment-variable expressions in `api_key`.
Replace the placeholder above through your configuration/secret-management
workflow; omit `api_key` only when the target endpoint does not require
authentication.

### Google GenAI

To call the native Gemini `generateContent` REST API, select the dialect
explicitly and provide the Google AI endpoint and API key:

```yaml
models:
  gemini:
    name: gemini-3.8-flash
    type: chat
    dialect: google_genai
    api_base: https://generativelanguage.googleapis.com
    api_key: "<your Gemini API key>"
```

The model name may be either `gemini-...` or `models/gemini-...`. The dialect
supports the Google AI Gemini API's `v1` and `v1beta` versions. An endpoint root
defaults to `v1beta`; an explicit `/v1` or `/v1beta` suffix selects that version:

```yaml
# Stable API; use /v1beta explicitly for the beta version.
api_base: https://generativelanguage.googleapis.com/v1
```

Requests use `POST /<version>/models/<model>:generateContent`, or
`POST /<version>/models/<model>:streamGenerateContent?alt=sse` for streaming.
Gateway path prefixes are preserved. Unknown API versions and full resource
URLs are rejected at dialect construction before I/O. The final HTTP URL is
checked against the selected version and recorded in request evidence; explicit
`/v1` and `/v1beta` endpoints also have distinct runtime/provenance fingerprints.
Vertex AI resource paths and OAuth authentication are not mapped by this dialect.

Function parameter schemas have a version-specific wire contract, based on
Google's [v1 Discovery schema](https://generativelanguage.googleapis.com/$discovery/rest?version=v1)
and [v1beta Discovery schema](https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta)
(revision `20260928`). `v1beta` permits both `parameters` and
`parametersJsonSchema`, mutually exclusively; SiEval selects
`parametersJsonSchema` to preserve the supplied JSON Schema. For either version,
function parameters must describe an object. SiEval requires an explicit root
`type: object`: scalar/array roots, type unions, and roots whose type is only
implied by `$ref` or composition are rejected before I/O. This is SiEval's
conservative supported subset, not a claim that Google rejects every referenced
object schema. A function with no parameters may omit `parameters`, or declare
`{"type": "object", "properties": {}}`.

`v1` exposes only `parameters`, using Google's OpenAPI `Schema` subset.
SiEval converts explicit
types, nested properties/items/`anyOf`, string enums, and supported scalar
constraints without discarding keywords. Unsupported constructs such as
`additionalProperties`, `$ref`/`$defs`, type arrays, and numeric enums are rejected
before I/O; every converted schema node must declare an explicit `type`.
Use `/v1beta` for other schema constructs, retaining the explicit object root.
The `v1` conversion encodes Schema types as uppercase enums and size limits as
int64 strings. Structured output
continues to use `generationConfig.responseJsonSchema` on both API versions.
Lossless conversion here means preserving supported structural and validation
information, not promising that Google uses every annotation during generation.
`default` and `example` are preserved unchanged; Google explicitly ignores
`default` for validation. Nested `example` is not rejected solely on the basis
of Google's root-population description. Fixed Discovery contract tests run
offline; neither evaluation nor tests fetch a changing schema at runtime.

The credential is sent only in the `x-goog-api-key` header. As with the
Anthropic example, YAML does not expand environment-variable expressions and
the provider-neutral HTTP connection does not infer a Google-specific
environment variable. Omit `api_key` only for a compatible endpoint that does
not require authentication.

The wire adapter implements text and image input, function tools,
`google_search` and `url_context` hosted tools, JSON output, thinking controls,
logprobs, streaming, deterministic request seeds, and opaque continuation.
Function declarations require a non-empty description and a Google-compatible
name of at most 128 characters. When hosted tools are combined with an explicit
`tools.choice: auto`, the dialect sends Google's `VALIDATED` mode because
server-side tool invocation circulation does not support `AUTO`.
Google documents combining custom functions with hosted tools as a Gemini 3
feature. Known Gemini 1.x/2.x model identifiers therefore reject this combination
before I/O, without disabling standalone function tools, Google Search, or URL
Context where supported. Unknown provider aliases do not acquire a model-family
constraint by name; model support still requires capability admission.
Historical function calls are validated turn by turn: every model turn that
emits calls must be followed immediately by one user/tool turn containing
exactly one matching `FunctionResponse` for each call. Matching the calls and
responses only across the conversation as a whole is rejected before I/O, as
required by Google's
[function-calling guidance](https://ai.google.dev/gemini-api/docs/generate-content/function-calling).
Gemini 3 responses must also provide Google's exact non-empty function-call ID,
which is replayed unchanged in the matching `FunctionResponse`; a missing ID is
an output-contract error. The first function call in each Gemini 3 step must
also preserve its provider-issued `thoughtSignature`; unsigned canonical tool
history is rejected for the current turn, while completed earlier turns remain
accepted. Older Gemini families retain a deterministic local fallback for
responses from endpoints that omit the formerly optional ID.

Gemini 3 tool-context circulation has an additional state contract: when
`includeServerSideToolInvocations` is enabled, each returned `toolCall` and
`toolResponse` must carry its own non-empty `thoughtSignature`. Missing signatures
fail response validation rather than producing incomplete replay state. These
parts and signatures must also survive opaque replay and final HTTP serialization,
even if a later request no longer enables hosted tools. This does not require
signatures on every text part or every parallel custom function call. See Google's
[tool-combination contract](https://ai.google.dev/gemini-api/docs/generate-content/tool-combination).

Hosted-tool configuration follows the Google Discovery schema. `url_context`
accepts only an empty object. `google_search` accepts `timeRangeFilter` and
`searchTypes`; time ranges require `startTime` and `endTime` together as
RFC3339 timestamps with explicit UTC offsets, and the start must not be after
the end. `webSearch` and `imageSearch` are empty marker objects. Unknown
configuration keys are rejected locally rather than being deferred to a
provider 400. The schema source is Google's public
[Generative Language Discovery document](https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta).
Ordinary YAML bindings do not yet supply Google model-profile outcomes, so
model-dependent capabilities such as images, reasoning, tools, structured
output, logprobs, and opaque continuation fail reconciliation with
`model_outcome_missing` instead of being assumed to work on every Gemini model
or compatible endpoint. Plain text generation, streaming, seeds, and audited
provider options are currently reachable without those profile outcomes.

Per-image `detail`, assistant prefill, input scoring, and
`previous_response_id` are rejected before I/O. Google has no distinct
developer-message role, so a `developer` message is also rejected instead of
being silently weakened to a system instruction.

Gemini 3 rejects `candidateCount` greater than one, so the dialect rejects
`sampling.n > 1` for Gemini 3 model identifiers before network I/O. Earlier
Gemini families retain the provider's multiple-candidate behavior where no
singular response channel makes it ambiguous.

The dialect also applies Google's documented model-family request domains.
Gemini 2.5 models reject `reasoning.effort` and validate positive
`reasoning.budget_tokens` against the published range for Pro, Flash, or
Flash-Lite. Gemini 3 models validate `reasoning.effort` against the model's
published thinking-level table; for example, 3.1 Pro and 3.8 Flash reject
`minimal`, 3 Pro Preview accepts only `low` and `high`, and 3.1 Flash-Lite
Image accepts only `minimal` and `high`. Gemini 3 continues to accept
`reasoning.budget_tokens` for provider backward compatibility, although Google
recommends `reasoning.effort` / `thinkingLevel` for more predictable behavior.
Google's current Gemini 3 migration guidance says to remove `temperature`,
`top_p`, and `top_k`, so explicit values are rejected for Gemini 3 rather than
being recorded as effective sampling controls. These policies follow the
[Google thinking guide](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
and the
[current-model migration guide](https://ai.google.dev/gemini-api/docs/latest-model).

`reasoning.summary: auto` requests visible thoughts with `includeThoughts`.
Google documents those summaries as optional when no thought is available, so
their absence is valid; returned `thoughtSignature` values and signed history
are preserved without rewriting provider parts in the opaque continuation
channel for later turns.
When `finishReason` is `MAX_TOKENS`, partial text is preserved, requested
channels may be absent, and incomplete JSON is not misreported as a malformed
structured response; downstream truncation accounting receives the provider's
finish reason unchanged.

Provider-only root controls can be supplied through matching
`dialect_options`, for example `safetySettings`, `serviceTier`, or `store`.
`cachedContent` and `labels` are available only on `v1beta` and are rejected on
`v1`. Options are audited against the final JSON request. Canonical fields
such as `contents`, `generationConfig`, `tools`, `model`, and `stream` cannot be
overridden there, and there is no arbitrary HTTP-header passthrough.

## Environment Variables

Custom environment variables can be passed through the YAML config's `infer.env` section. Values are injected into the inference engine process.
