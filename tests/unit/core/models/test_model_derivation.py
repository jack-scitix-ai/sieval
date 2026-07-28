"""
Unit tests for Model.with_args and meta() derivation logic.

Covers parent limiter wiring, nested derivation, and meta() field presence —
paths not exercised by test_model.py. (The as_type coverage that used to live
here was removed with the method itself; RFC #25 dropped cross-kind model
conversion.)

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import pytest

from sieval.core.models import ChatModel, GenModel, Request, Response
from sieval.core.models.transports import (
    OpenAIChatTransport,
    OpenAICompletionsTransport,
)


# ---------------------------------------------------------------------------
# Stub implementations — no real API calls
# ---------------------------------------------------------------------------
class _NeverCallTransport:
    """Transport double that fails loudly if any wire call is attempted."""

    def __init__(self, capabilities):
        self._capabilities = frozenset(capabilities)

    @property
    def capabilities(self):
        return self._capabilities

    async def arun(self, req: Request) -> Response:
        raise RuntimeError("Stub must not be called in unit tests")


class StubGenModel(GenModel):
    """GenModel stub that never hits a real API."""

    def _build_default_transport(self):
        return _NeverCallTransport(OpenAICompletionsTransport.CAPABILITIES)


class StubChatModel(ChatModel):
    """ChatModel stub that never hits a real API."""

    def _build_default_transport(self):
        return _NeverCallTransport(OpenAIChatTransport.CAPABILITIES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_gen():
    """GenModel with a concurrency limiter (total_tokens=64)."""
    return StubGenModel(model="base-gen", api_key="fake", concurrency_limit=64)


@pytest.fixture
def base_gen_no_limit():
    """GenModel without any concurrency limiter."""
    return StubGenModel(model="base-gen-unlimited", api_key="fake")


# ===================================================================
# TestModelDerivation
# ===================================================================
class TestModelDerivation:
    # ------------------------------------------------------------------
    # with_args — parent limiter wiring
    # ------------------------------------------------------------------

    def test_with_args_creates_child_with_parent_limiter(self, base_gen):
        """with_args(concurrency_limit=32) → child._parent_limiter is base _limiter."""
        child = base_gen.with_args(concurrency_limit=32)

        assert child._limiter is not None
        assert child._limiter.total_tokens == 32
        # Parent's own limiter becomes the child's _parent_limiter
        assert child._parent_limiter is base_gen._limiter

    def test_with_args_without_concurrency_limit_shares_limiters(self, base_gen):
        """with_args() without concurrency_limit keeps the same limiter refs."""
        child = base_gen.with_args(temperature=0.7)

        # Shares the exact same limiter object — no new limiter created
        assert child._limiter is base_gen._limiter
        # parent_limiter is also unchanged (base_gen has no parent, so None)
        assert child._parent_limiter is base_gen._parent_limiter

    def test_with_args_without_concurrency_limit_no_base_limiter(
        self, base_gen_no_limit
    ):
        """with_args() on an unlimited model → child also has no limiter."""
        child = base_gen_no_limit.with_args(temperature=0.5)

        assert child._limiter is None
        assert child._parent_limiter is None

    def test_with_args_new_limiter_total_tokens(self, base_gen):
        """Child limiter capacity equals the requested concurrency_limit."""
        child = base_gen.with_args(concurrency_limit=16)

        assert child._limiter.total_tokens == 16

    def test_with_args_shares_transport(self, base_gen):
        """Derived models reuse the same Transport (same client, same wire)."""
        child = base_gen.with_args(temperature=0.7)

        assert child._transport is base_gen._transport
        assert child.capabilities == base_gen.capabilities

    # ------------------------------------------------------------------
    # Nested derivation
    # ------------------------------------------------------------------

    def test_nested_derivation_child_has_parent_limiter(self, base_gen):
        """base → child1 (limit=32) → child1.with_args(no limit): child2 inherits."""
        child1 = base_gen.with_args(concurrency_limit=32)
        # Further derivation without new concurrency_limit is allowed
        child2 = child1.with_args(temperature=0.1)

        # child2 shares child1's limiter
        assert child2._limiter is child1._limiter
        # child2 also shares child1's parent_limiter (which points to base_gen._limiter)
        assert child2._parent_limiter is not None
        assert child2._parent_limiter is base_gen._limiter

    def test_nested_derivation_with_new_limit_raises(self, base_gen):
        """base → child1 (limit=32) → child1.with_args(limit=16) must raise."""
        child1 = base_gen.with_args(concurrency_limit=32)

        with pytest.raises(ValueError, match="multi-level"):
            child1.with_args(concurrency_limit=16)

    # ------------------------------------------------------------------
    # meta()
    # ------------------------------------------------------------------

    def test_model_meta_contains_model_field(self, base_gen):
        """meta() must return a dict with a 'model' key."""
        m = base_gen.meta()

        assert "model" in m
        assert m["model"] == "base-gen"

    def test_model_meta_contains_api_base(self, base_gen):
        """meta() 'api_base' is None when no api_base was given."""
        m = base_gen.meta()

        assert "api_base" in m
        assert m["api_base"] is None

    def test_model_meta_contains_default_params(self, base_gen):
        """meta() 'default_params' reflects kwargs passed at construction."""
        model = StubGenModel(
            model="test-params", api_key="fake", temperature=0.5, top_p=0.9
        )
        m = model.meta()

        assert "default_params" in m
        assert m["default_params"]["temperature"] == 0.5
        assert m["default_params"]["top_p"] == 0.9

    def test_model_meta_api_base_set(self):
        """meta() 'api_base' reflects the value passed at construction."""
        model = StubGenModel(
            model="remote", api_key="fake", api_base="http://localhost:9000"
        )
        m = model.meta()

        assert m["api_base"] == "http://localhost:9000"

    def test_model_meta_after_with_args_inherits_model_name(self, base_gen):
        """Derived model's meta() still reports the original model name."""
        child = base_gen.with_args(temperature=0.3)
        m = child.meta()

        assert m["model"] == "base-gen"

    def test_model_meta_after_with_args_reflects_overridden_kwargs(self, base_gen):
        """Derived model's meta() 'default_params' shows overridden kwargs."""
        child = base_gen.with_args(temperature=0.3)
        m = child.meta()

        assert m["default_params"].get("temperature") == 0.3


# ===================================================================
# TestModelExtra
# ===================================================================
class TestModelExtra:
    """extra: init, property, with_args propagation, meta() exposure."""

    def test_default_extra_is_empty(self):
        """Model without extra returns empty dict."""
        model = StubGenModel(model="test", api_key="fake")
        assert model.extra == {}

    def test_init_stores_extra(self):
        """Model with extra stores and returns it."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        assert model.extra == extra

    def test_with_args_preserves_extra(self):
        """with_args() without extra preserves the original."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        child = model.with_args(temperature=0.5)
        assert child.extra == extra

    def test_with_args_overrides_extra(self):
        """with_args(extra=...) replaces the original."""
        old = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        new = {"sequence_wrappers": {"rna": "<rna>{seq}</rna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=old)
        child = model.with_args(extra=new)
        assert child.extra == new

    def test_with_args_sets_extra_on_plain_model(self):
        """with_args(extra=...) on a model without extra sets it."""
        model = StubGenModel(model="test", api_key="fake")
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        child = model.with_args(extra=extra)
        assert child.extra == extra
        # Original unchanged
        assert model.extra == {}

    def test_meta_includes_extra(self):
        """meta() includes extra when configured."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        m = model.meta()
        assert m["extra"] == extra

    def test_meta_omits_extra_when_empty(self):
        """meta() omits extra key when no extra is configured."""
        model = StubGenModel(model="test", api_key="fake")
        m = model.meta()
        assert "extra" not in m

    def test_extra_not_in_kwargs(self):
        """extra must NOT leak into _kwargs (would be sent to API)."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        assert "extra" not in model._kwargs

    def test_extra_not_in_meta_default_params(self):
        """meta()['default_params'] must not contain extra."""
        extra = {"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}}
        model = StubGenModel(model="test", api_key="fake", extra=extra)
        assert "extra" not in model.meta()["default_params"]

    def test_with_args_extra_not_in_child_kwargs(self):
        """with_args(extra=...) must not leak into child _kwargs."""
        model = StubGenModel(model="test", api_key="fake")
        child = model.with_args(
            extra={"sequence_wrappers": {"dna": "<dna>{seq}</dna>"}},
            temperature=0.5,
        )
        assert "extra" not in child._kwargs
        assert child._kwargs.get("temperature") == 0.5
