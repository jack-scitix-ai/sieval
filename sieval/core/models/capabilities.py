"""Capability catalog for the Model IR.

Each ``Capability`` represents a feature a Transport can declare support for.
``Model.assert_capability()`` checks these at setup time — a Request using a
feature the Transport lacks is rejected immediately, never silently ignored.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from enum import Flag, auto


class Capability(Flag):
    # ── Input modality ────────────────────────────────────────
    Completion = auto()  # input: str (base model)
    Chat = auto()  # input: list[Message]

    # ── Tools ─────────────────────────────────────────────────
    FunctionCalling = auto()  # client-side tool call (formerly Tools)
    ServerTools = auto()  # provider-hosted tools master switch
    WebSearch = auto()
    WebFetch = auto()
    HostedCodeExecution = auto()
    FileSearch = auto()
    ComputerUse = auto()

    # ── Reasoning ─────────────────────────────────────────────
    Reasoning = auto()
    ReasoningOptional = auto()  # can be turned off (OpenAI o-series, older Claude)
    ReasoningAlwaysOn = auto()  # forced on (Claude Fable 5 / Mythos 5)
    ReasoningEffort = auto()  # supports effort enum
    ReasoningBudget = auto()  # supports exact budget_tokens
    ReasoningMode = auto()  # supports mode (OpenAI standard/pro)
    ReasoningContext = auto()  # supports cross-turn reasoning state
    ReasoningTaskBudget = auto()  # supports cross-call total budget (Anthropic)

    # ── Logprobs / scoring ────────────────────────────────────
    TopKLogprobs = auto()
    InputScoring = auto()  # prompt-side logprobs (BPB/PPL)
    SampledLogprobs = auto()
    SampledLogprobsWithTokenIds = auto()  # token_id populated (sglang native)

    # ── Other ─────────────────────────────────────────────────
    StructuredOutput = auto()
    Prefill = auto()
    FIM = auto()
