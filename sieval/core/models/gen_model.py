"""One-cycle ``openai_completions`` compatibility wrapper.

The historical constructor creates a private pool, but the inherited execution
path is the same deployment/pool/runtime-plan binding used by canonical models.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from typing import cast

import anyio

from sieval.core.types import JSONValue

from .dialect import Dialect
from .dialects.openai_completions import OpenAICompletionsDialect
from .ir import CompletionInput, ModelInput
from .model import Model, _legacy_runtime_plan, build_legacy_openai_binding
from .reconcile import RuntimeBindingPlan


class GenModel(Model):
    """Deprecated constructor wrapper selecting ``openai_completions``."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        concurrency_limit: int | None = None,
        parent_limiter: anyio.CapacityLimiter | None = None,
        extra: dict[str, JSONValue] | None = None,
        transport: Dialect | None = None,
        **kwargs: object,
    ) -> None:
        binding = build_legacy_openai_binding(
            dialect_id="openai_completions",
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_retries=max_retries,
            concurrency_limit=concurrency_limit,
            parent_limiter=parent_limiter,
        )
        self._client = binding.pool.connection
        self._model = model
        self._api_base = api_base
        dialect = (
            transport if transport is not None else self._build_default_transport()
        )
        self._initialize(
            deployment=binding.deployment,
            pool=binding.pool,
            runtime_plan=binding.runtime_plan,
            dialect=dialect,
            local_limiter=(
                binding.local_limiter
                if binding.local_limiter is not None
                else cast(anyio.CapacityLimiter | None, binding.pool.shared_limiter)
            ),
            parent_limiter=binding.parent_limiter,
            builder_defaults=kwargs,
            extra=extra,
            api_base=api_base,
            lifecycle_owner=self,
        )

    def _build_default_transport(self) -> Dialect:
        """Deprecated hook retained for downstream test/subclass adapters."""

        return OpenAICompletionsDialect(self._client, self._model)

    def _coerce_input(self, prompt: object) -> ModelInput:
        if isinstance(prompt, CompletionInput):
            return prompt
        if isinstance(prompt, str):
            return CompletionInput(prompt)
        raise TypeError("GenModel prompt must be text or CompletionInput")

    def as_type(
        self,
        model_type: type[Model],
        runtime_plan: RuntimeBindingPlan | None = None,
    ) -> Model:
        """Reconstruct a truthful compatibility wrapper for a target plan.

        The optional second argument is the canonical composition path.  The
        historical one-argument API remains available only for a wrapper that
        owns its private legacy pool; session-bound wrappers cannot invent a
        replacement reconciled plan.
        """

        from .chat_model import ChatModel

        if model_type is ChatModel:
            target_dialect = "openai_chat"
        elif model_type is GenModel:
            target_dialect = "openai_completions"
        else:
            raise TypeError("model_type must be exactly ChatModel or GenModel")
        if runtime_plan is None:
            if self._lifecycle_owner is None:
                raise ValueError(
                    "session-bound wrappers require an explicitly reconciled "
                    "RuntimeBindingPlan"
                )
            runtime_plan = _legacy_runtime_plan(
                dialect_id=target_dialect,
                requested_model_id=self._model,
                deployment=self._deployment,
                identity=self._pool.identity,
            )
        return self._as_wrapper(model_type, target_dialect, runtime_plan)
