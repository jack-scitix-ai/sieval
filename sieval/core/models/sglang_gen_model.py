"""Explicit one-cycle bypass for SGLang's native ``/generate`` protocol.

``sglang_native`` has no executable PR-1 binder, so this facade intentionally
does not fabricate a ``RuntimeBindingPlan`` or enter the canonical dialect
registry.  It preserves the existing native wire implementation until PR 5
activates that dialect, while exposing a truthful ``sglang_legacy`` identity to
the temporary task compatibility checks.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

from types import TracebackType
from typing import Any, Self, cast
from uuid import uuid4

import anyio
from openai import AsyncOpenAI

from sieval.core.types import JSONValue

from .capabilities import Capability
from .deployment import (
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    Engine,
    ServingFacts,
)
from .dialect import validate_request_invariants
from .dialect_registry import compatibility_factory_for
from .ir import CompletionInput, ModelInput, Request, Response
from .model import Model
from .transports.sglang import SglangTransport


class SglangGenModel(Model):
    """Legacy native SGLang facade, deliberately outside canonical binding."""

    def __new__(cls, *args: object, **kwargs: object) -> Any:
        if cls is SglangGenModel:
            factory = compatibility_factory_for("sglang_native")
            if factory is not None:
                return factory(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        concurrency_limit: int | None = None,
        parent_limiter: anyio.CapacityLimiter | None = None,
        extra: dict[str, JSONValue] | None = None,
        transport: SglangTransport | None = None,
        **kwargs: object,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            max_retries=max_retries,
        )
        self._kwargs = dict(kwargs)
        self._extra = dict(extra) if extra is not None else None
        self._parent_limiter = parent_limiter
        self._limiter = (
            anyio.CapacityLimiter(concurrency_limit)
            if concurrency_limit is not None
            else None
        )
        endpoint = str(self._client.base_url).rstrip("/")
        shared_limiter = parent_limiter if parent_limiter is not None else self._limiter
        private_scope = uuid4().hex
        identity = ConnectionIdentity(
            endpoint=endpoint,
            connection_family="openai_sdk",
            credential_scope=(
                f"legacy-private:{private_scope}:explicit-credential"
                if api_key is not None
                else f"legacy-private:{private_scope}:environment-credential"
            ),
            retry_policy=f"openai-sdk:max-retries={max_retries}",
            quota_scope=f"legacy-private:{private_scope}",
        )
        self._deployment = Deployment(
            deployment_id=None,
            plan=None,
            engine=Engine("sglang"),
            engine_source="config",
            api_base=endpoint,
            endpoints={},
            topology=None,
            metrics_url=None,
            facts=ServingFacts(),
        )
        self._pool = ConnectionPool(self._client, identity, shared_limiter)
        legacy_transport = (
            transport if transport is not None else self._build_default_transport()
        )
        self._legacy_transport = legacy_transport
        self._transport = cast(Any, legacy_transport)
        self._lifecycle_owner = self

    def _build_default_transport(self) -> SglangTransport:
        return SglangTransport(self._client, self._model, self._api_base)

    @property
    def dialect_id(self) -> str:
        return "sglang_legacy"

    @property
    def runtime_plan(self) -> None:
        return None

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._legacy_transport.capabilities

    def _coerce_input(self, prompt: object) -> ModelInput:
        if isinstance(prompt, CompletionInput):
            return prompt
        if isinstance(prompt, str):
            return CompletionInput(prompt)
        raise TypeError("SglangGenModel prompt must be text or CompletionInput")

    async def arun(self, req: Request) -> Response:
        validate_request_invariants(req)
        async with self._pool.acquire(self._limiter):
            return await self._legacy_transport.arun(req)

    def with_dialect(self, dialect_id: str, runtime_plan: Any) -> Model:
        del dialect_id, runtime_plan
        raise RuntimeError(
            "sglang_legacy cannot rebind before the sglang_native PR-5 binder"
        )

    def _legacy_lifecycle_owner(self) -> "SglangGenModel":
        owner = self._lifecycle_owner
        if not isinstance(owner, SglangGenModel):
            raise RuntimeError("sglang_legacy binding has no lifecycle owner")
        return owner

    async def aclose(self) -> None:
        owner = self._legacy_lifecycle_owner()
        await owner._pool.aclose()

    async def __aenter__(self) -> Self:
        owner = self._legacy_lifecycle_owner()
        await owner._pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        await self.aclose()
