# Google Interactions response fixtures

Formatting must not change SSE event separators, including the final blank
lines in the live captures. Only `end-of-file-fixer` excludes `.sse` files in
this directory; all other hook selection remains unchanged.

`stream_thought_summary_completed.sse` and
`stream_thought_summary_incomplete.sse` come from successful live requests on
October 27, 2026, to `POST /v1beta/interactions`, using `gemini-3.6-flash`,
`stream: true`, `store: false`, and `thinking_summaries: auto`. The incomplete
and completed cases used output budgets of 1024 and 2048 respectively.

Only thought signatures are replaced with `sig-redacted`; the generated text,
including all spaces and newlines, is preserved. The two streaming captures
contain no request headers, API keys, or stored interaction IDs. Both terminal
events omit `steps`.

Each capture has two thought-summary Content items. Their text already ends
with three newlines. The scalar reasoning projection concatenates that text
without adding separators, matching the official streaming example's
`print(event.delta.content.text, end="")` behavior:
<https://ai.google.dev/gemini-api/docs/streaming#streaming-with-thinking>.
The items themselves remain distinct, as the OpenAPI description specifies:
<https://ai.google.dev/static/api/interactions.openapi.json>.

Tests that derive a non-streaming response or a terminal `steps` snapshot from
these items are synthetic compatibility tests, not additional live captures.

Four additional streaming fixtures were captured from HTTP 200 responses on
October 28, 2026, against the same model and endpoint, with `store: false` and
`thinking_summaries: auto`:

- `stream_signature_only_completed.sse`: completed text with a thought
  signature but no thought-summary delta.
- `stream_multiple_functions_signature_only.sse`: two function calls and
  `requires_action`, with a thought signature but no summary delta.
- `stream_url_context_signature_only.sse`: URL-context call/result and citations,
  followed by a signed thought without a summary delta.
- `stream_google_search_signature_only.sse`: Google Search call/result and
  citation deltas, followed by a signed thought without a summary delta.

These replies exposed an overly strict requested-summary presence check, which
the capture-based regression tests now prevent. Optional summary omission does
not waive validation of provided content or reconciliation with terminal steps.
The JSON event frames are reserialized; their text and whitespace are unchanged.
Signatures and call identifiers are replaced with consistent placeholders. These
four fixtures contain no request headers, API keys or real interaction IDs.

`function_choice_none_provider_violation.json` records another HTTP 200 response
from October 28, 2026. After an error function result, the provider emitted a new
`get_code` call despite the request's `tool_choice: none`. This is a negative
response-contract fixture: SiEval must reject the disallowed call rather than
silently executing it. Its signature and call identifier are also placeholders.

The final live capability matrix, accounting examples and validation limits are
recorded in [live_validation.md](live_validation.md).
