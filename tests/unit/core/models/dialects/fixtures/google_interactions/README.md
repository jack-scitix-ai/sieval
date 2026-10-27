# Google Interactions response fixtures

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
