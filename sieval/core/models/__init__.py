"""Model backends, the provider-agnostic IR, and Transport frontends.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from .capabilities import Capability
from .chat_model import ChatModel
from .exceptions import CapabilityError
from .gen_model import GenModel
from .ir import (
    Citation,
    GroundingChunk,
    GroundingMetadata,
    InputScoringResult,
    ReasoningOutput,
    ReasoningParams,
    Request,
    Response,
    SamplingParams,
    ServerToolSpec,
    ServerToolType,
    ServerToolUse,
    TokenLogprob,
    TopKEntry,
    UsageStats,
)
from .model import Model, ModelCallMeta, ModelMeta, ModelOutput, ModelUsage
from .sglang_gen_model import SglangGenModel
from .transport import Transport

__all__ = [
    "Capability",
    "CapabilityError",
    "ChatModel",
    "Citation",
    "GenModel",
    "GroundingChunk",
    "GroundingMetadata",
    "InputScoringResult",
    "Model",
    "ModelCallMeta",
    "ModelMeta",
    "ModelOutput",
    "ModelUsage",
    "ReasoningOutput",
    "ReasoningParams",
    "Request",
    "Response",
    "SamplingParams",
    "ServerToolSpec",
    "ServerToolType",
    "ServerToolUse",
    "SglangGenModel",
    "TokenLogprob",
    "TopKEntry",
    "Transport",
    "UsageStats",
]
