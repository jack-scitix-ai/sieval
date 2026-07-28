"""Model-layer exceptions.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""


class CapabilityError(ValueError):
    """Raised at setup when a Request requires a Capability the Transport lacks.

    This replaces the previous pattern of silently ignoring unsupported
    parameters (e.g. ``echo=True`` on ChatModel). Fail loud at setup, never
    silently return wrong results.
    """
