# Live Interactions validation — October 28, 2026

The final implementation was exercised against Google's real
`generativelanguage.googleapis.com` service with `gemini-3.6-flash`
(`3.6-flash-07-2026` in model metadata). Both `POST /v1beta/interactions` and
`POST /v1/interactions` returned successful responses.

The validated dialect's SHA-256 is
`2886c0eb783bd3e489af0604268e3eb7bdfd202120172d6307dd300b565ba967`.
Thirty verification scenarios passed; their final successful matrix represents
33 generation requests, excluding earlier diagnostic/probe iterations.
Requests used output limits of 128, 512, 1024 or 2048 tokens. Every transmitted
request was checked for Boolean `store: false` and the absence of
`previous_interaction_id`, `background` and `agent`.

## Covered paths

| Area | Actual-service verification |
| --- | --- |
| Text | Non-streaming and streaming; `completed` and budget-limited `incomplete` |
| Sampling | Temperature, top-p and seed accepted; an explicit stop sequence stops visible output |
| Reasoning | Minimal, low, medium and high effort; summary `none` and `auto`; multiple summary Content items concatenate verbatim |
| Opaque continuation | Two-turn non-streaming and streaming replay, exact native-history prefix and system-instruction preservation |
| Persistence | A live `ReasoningOutput` is serialized to a private temporary file, reloaded with type metadata, and used for the next actual request |
| Function choice | Auto, required, none and a specifically named function |
| Function results | String, object, annotated text Content, type-only citation annotation, image Content and `is_error: true`; names inferred from matching calls |
| Multiple calls | Two streamed function calls with distinct IDs, two separate tool-result messages and a streamed final answer |
| Structured output | Schema-shaped JSON in non-streaming and streaming replies |
| Images | Inline PNG, an HTTPS image URI and PNG returned by a function |
| Hosted tools | Google Search and URL context, each non-streaming and streaming, with actual call/result evidence and citations |
| Public binding | Ordinary YAML binding followed by `Model.agenerate(ChatInput(...))`, non-streaming and streaming |
| Storage guards | Invalid store values and stateful/background options rejected before HTTP; these checks deliberately issue no provider request |

Feature-specific requests exercise the standalone typed dialect protocol.
The ordinary-YAML checks exercise public chat binding without injecting a
model-profile capability outcome. These tests do not bypass or claim to remove
the model-profile requirements for capability-dependent `Model` requests.

## Accounting evidence

The following counters come from actual responses, not synthetic fixtures.
All four observed totals equal input + tool-use + output + thoughts.

| Response | Raw input | Raw tool-use | Raw output | Raw thoughts | Raw total | IR input | IR output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text, non-streaming | 16 | 0 | 1 | 109 | 126 | 16 | 110 |
| Thought summaries, streaming | 31 | 0 | 621 | 639 | 1291 | 31 | 1260 |
| URL context, streaming | 95 | 213 | 87 | 303 | 698 | 308 | 390 |
| Google Search, streaming | 156 | 0 | 78 | 239 | 473 | 156 | 317 |

This independently supports adding thoughts to the IR output count and tool-use
tokens to the IR input count for these observed replies. Returned cache counts
are preserved as breakdowns, not added again.

## Findings and limits

- `thinking_summaries: auto` sometimes returns only a thought signature, with
  no summary delta. The overly strict presence check was removed; visible text,
  calls, citations and opaque signatures remain available. Four live SSE
  fixtures cover this behavior.
- After an error result, one provider reply emitted a new function call despite
  `tool_choice: none`. The client correctly rejected it. The negative live
  fixture preserves that guard; ordinary none-choice requests also passed.
- Streaming terminal interactions omitted `steps`. Terminal-step reconciliation
  and metadata-only usage fallback remain deterministic compatibility tests,
  not behaviors forced or observed by this live run.
- Accepted sampling parameters are not proof of statistical seed determinism.
  Validation is specific to the tested model and service versions.
- Captures contain no credentials or request headers; signatures and call IDs
  in committed fixtures are placeholders, not usable continuation credentials.

## Local automated verification

- Unit + integration: **10563 passed, 29 skipped** in the verification environment.
- Interactions unit + resume integration: **799 passed**, dialect coverage **97%**.
- Repository Ruff lint and formatting, scoped type checks, and diff checks pass.
- Quick preflight: **26 PASS, 1 SKIP**; the skip is deep HTTP link reachability.

This record is not a claim of a completed rebase, clean-install gate, global
coverage gate, mutation score or hosted CI result.
