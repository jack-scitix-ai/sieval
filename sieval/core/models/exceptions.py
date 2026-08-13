"""Model-layer exceptions.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""


class CapabilityError(ValueError):
    """Raised when a request requires an unavailable model capability.

    This replaces the previous pattern of silently ignoring unsupported
    parameters (for example, input scoring on a chat-only dialect).
    """
