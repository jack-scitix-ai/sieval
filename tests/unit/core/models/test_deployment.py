"""Tests for realized deployments, route resolution, and connection lifetime.

AI-Generated Code - GPT-5.6 (OpenAI)
"""

import dataclasses
from typing import Any, cast

import anyio
import pytest

from sieval.core.models.deployment import (
    ConnectionIdentity,
    ConnectionPool,
    Deployment,
    DeploymentPlanProjection,
    DeploymentTopology,
    Engine,
    ResolvedRoute,
    RouteIntent,
    ServingFacts,
    deployment_fingerprint,
    resolve_route,
)


class _Connection:
    def __init__(self) -> None:
        self.close_calls = 0
        self.close_entered = anyio.Event()
        self.allow_close = anyio.Event()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_entered.set()
        await self.allow_close.wait()


class _SyncConnection:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FailingConnection:
    async def aclose(self) -> None:
        raise RuntimeError("close failed")


def _deployment(
    *,
    endpoints: dict[str, str] | None = None,
    api_base: str = "https://models.example/v1",
) -> Deployment:
    return Deployment(
        deployment_id="deployment-1",
        plan=DeploymentPlanProjection(
            fingerprint="sha256:plan",
            engine_id="vllm",
        ),
        engine=Engine("vllm"),
        engine_source="deployment",
        api_base=api_base,
        endpoints={} if endpoints is None else endpoints,
        topology=DeploymentTopology(
            is_disaggregated=False,
            roles=("full",),
            total_gpus=2,
        ),
        metrics_url="https://models.example/metrics",
        facts=ServingFacts(engine_version="0.9.0", tokenizer_available=True),
    )


def _identity(
    *,
    endpoint: str = "https://models.example/v1",
    family: str = "openai_sdk",
    credential_scope: str = "team-a",
) -> ConnectionIdentity:
    return ConnectionIdentity(
        endpoint=endpoint,
        connection_family=family,
        credential_scope=credential_scope,
        retry_policy="standard",
        quota_scope="deployment-1",
    )


class TestDeploymentValueTypes:
    @pytest.mark.parametrize(
        "value",
        [
            ServingFacts(),
            DeploymentPlanProjection("plan", "engine"),
            DeploymentTopology(False, ("full",), 1),
            RouteIntent(),
            ResolvedRoute("full", "https://x/v1", "http", "fingerprint"),
            Engine("engine"),
            _identity(),
        ],
    )
    def test_value_types_are_frozen(self, value: Any) -> None:
        field = dataclasses.fields(value)[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field.name, None)

    def test_deployment_copies_and_freezes_endpoint_mapping(self) -> None:
        source = {"full": "https://full.example/v1"}
        deployment = _deployment(endpoints=source)

        source["full"] = "https://mutated.example/v1"
        assert deployment.endpoints == {"full": "https://full.example/v1"}
        with pytest.raises(TypeError):
            cast(Any, deployment.endpoints)["full"] = "https://forbidden.example/v1"

    def test_fingerprint_is_stable_across_mapping_and_role_order(self) -> None:
        first = dataclasses.replace(
            _deployment(
                endpoints={
                    "prefill": "https://prefill.example/v1",
                    "decode": "https://decode.example/v1",
                }
            ),
            topology=DeploymentTopology(
                is_disaggregated=True,
                roles=("prefill", "decode"),
                total_gpus=2,
            ),
        )
        assert first.topology is not None
        second = dataclasses.replace(
            first,
            endpoints={
                "decode": "https://decode.example/v1",
                "prefill": "https://prefill.example/v1",
            },
            topology=dataclasses.replace(
                first.topology,
                roles=("decode", "prefill"),
            ),
        )

        assert first.fingerprint == deployment_fingerprint(second)
        assert first.fingerprint.startswith("sha256:")

    def test_plan_and_realized_engine_must_agree(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            dataclasses.replace(_deployment(), engine=Engine("sglang"))

    @pytest.mark.parametrize("engine_source", ["bogus", "", None, 7])
    def test_engine_source_must_use_the_closed_domain(
        self, engine_source: object
    ) -> None:
        with pytest.raises(ValueError, match="engine_source"):
            dataclasses.replace(_deployment(), engine_source=cast(Any, engine_source))

    def test_unknown_engine_identity_and_source_must_agree(self) -> None:
        with pytest.raises(ValueError, match="must agree"):
            dataclasses.replace(_deployment(), engine_source="unknown")

        with pytest.raises(ValueError, match="must agree"):
            dataclasses.replace(
                _deployment(),
                deployment_id=None,
                plan=None,
                engine=Engine("unknown"),
                engine_source="config",
            )

        external = dataclasses.replace(
            _deployment(),
            deployment_id=None,
            plan=None,
            engine=Engine("unknown"),
            engine_source="unknown",
        )
        assert external.engine.engine_id == "unknown"

    @pytest.mark.parametrize(
        ("deployment_id", "plan"),
        [
            (None, DeploymentPlanProjection("sha256:plan", "vllm")),
            ("deployment-1", None),
        ],
    )
    def test_managed_identity_and_plan_must_be_present_together(
        self,
        deployment_id: str | None,
        plan: DeploymentPlanProjection | None,
    ) -> None:
        with pytest.raises(ValueError, match="deployment_id and plan"):
            dataclasses.replace(_deployment(), deployment_id=deployment_id, plan=plan)

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (("", "engine"), "fingerprint"),
            (("plan", ""), "engine_id"),
            (("plan", "engine", cast(Any, ["full"])), "must be strings"),
            (("plan", "engine", ("full", "")), "must be strings"),
            (("plan", "engine", ("prefill", "decode")), "sorted and unique"),
            (("plan", "engine", ("full", "full")), "sorted and unique"),
        ],
    )
    def test_plan_projection_rejects_invalid_identity(
        self, args: tuple[object, ...], message: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            DeploymentPlanProjection(*cast(Any, args))

    @pytest.mark.parametrize(
        "name", ["connection_family", "credential_scope", "retry_policy", "quota_scope"]
    )
    def test_connection_identity_scopes_must_not_be_empty(self, name: str) -> None:
        with pytest.raises(ValueError, match=name):
            dataclasses.replace(_identity(), **{name: ""})

    def test_engine_and_endpoint_role_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="engine_id"):
            Engine("")
        with pytest.raises(ValueError, match="roles must not be empty"):
            _deployment(endpoints={"": "https://models.example/v1"})

    @pytest.mark.anyio
    async def test_default_engine_verifier_returns_or_probes_facts(self) -> None:
        deployment = _deployment()
        observed = ServingFacts(engine_version="probe-result")

        async def probe(received: Deployment) -> ServingFacts:
            assert received is deployment
            return observed

        assert await deployment.engine.verify_deployment(deployment) is deployment.facts
        assert await deployment.engine.verify_deployment(deployment, probe) is observed


class TestRouteResolution:
    def test_single_role_is_selected_and_fingerprint_is_deterministic(self) -> None:
        deployment = _deployment(endpoints={"full": "https://full.example/v1"})

        route = resolve_route(deployment, "openai_chat", "openai_sdk")
        repeated = resolve_route(deployment, "openai_chat", "openai_sdk")

        assert route == repeated
        assert route.service_role == "full"
        assert route.endpoint == "https://full.example/v1"
        assert route.fingerprint.startswith("sha256:")

    def test_explicit_role_selects_one_of_multiple_endpoints(self) -> None:
        deployment = _deployment(
            endpoints={
                "prefill": "https://prefill.example/v1",
                "decode": "https://decode.example/v1",
            }
        )

        route = resolve_route(
            deployment,
            "openai_completions",
            "openai_sdk",
            RouteIntent(service_role="decode"),
        )

        assert route.service_role == "decode"
        assert route.endpoint == "https://decode.example/v1"

    def test_multiple_endpoints_without_role_are_rejected(self) -> None:
        deployment = _deployment(
            endpoints={
                "prefill": "https://prefill.example/v1",
                "decode": "https://decode.example/v1",
            }
        )

        with pytest.raises(ValueError, match="ambiguous"):
            resolve_route(deployment, "openai_chat", "openai_sdk")

    def test_missing_explicit_role_is_rejected_without_api_base_fallback(self) -> None:
        deployment = _deployment(
            endpoints={"full": "https://full.example/v1"},
        )

        with pytest.raises(ValueError, match="no endpoint.*missing"):
            resolve_route(
                deployment,
                "openai_chat",
                "openai_sdk",
                RouteIntent(service_role="missing"),
            )

    def test_api_base_is_the_default_when_no_role_map_exists(self) -> None:
        route = resolve_route(_deployment(), "openai_chat", "openai_sdk")

        assert route.service_role == "default"
        assert route.endpoint == "https://models.example/v1"

    def test_explicit_role_uses_api_base_when_no_role_map_exists(self) -> None:
        route = resolve_route(
            _deployment(),
            "openai_chat",
            "openai_sdk",
            RouteIntent("decode"),
        )

        assert route.service_role == "decode"
        assert route.endpoint == "https://models.example/v1"

    def test_explicit_role_without_any_endpoint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no endpoint for service role"):
            resolve_route(
                _deployment(api_base=""),
                "openai_chat",
                "openai_sdk",
                RouteIntent("decode"),
            )

    @pytest.mark.parametrize(
        ("dialect_id", "connection_family", "message"),
        [("", "openai_sdk", "dialect_id"), ("openai_chat", "", "connection_family")],
    )
    def test_route_symbols_must_not_be_empty(
        self, dialect_id: str, connection_family: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            resolve_route(_deployment(), dialect_id, connection_family)

    def test_no_endpoint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no routable endpoint"):
            resolve_route(
                _deployment(api_base=""),
                "openai_chat",
                "openai_sdk",
            )


class TestConnectionPoolIdentity:
    @pytest.mark.anyio
    async def test_family_route_and_full_identity_are_verified(self) -> None:
        connection = _Connection()
        identity = _identity()
        pool = ConnectionPool(connection, identity)
        route = ResolvedRoute(
            service_role="full",
            endpoint=identity.endpoint,
            connection_family=identity.connection_family,
            fingerprint="route",
        )

        pool.verify_family("openai_sdk")
        pool.verify_route(route)
        pool.verify_identity(identity)
        assert pool.identity is identity
        with pytest.raises(ValueError, match="family mismatch"):
            pool.verify_family("http_json")
        with pytest.raises(ValueError, match="route mismatch"):
            pool.verify_route(dataclasses.replace(route, endpoint="https://other/v1"))
        with pytest.raises(ValueError, match="identity mismatch"):
            pool.verify_identity(
                dataclasses.replace(identity, credential_scope="another-team")
            )

        connection.allow_close.set()
        await pool.aclose()

    def test_endpoint_identity_rejects_embedded_secrets(self) -> None:
        with pytest.raises(ValueError, match="must not contain credentials"):
            _identity(endpoint="https://user:secret@models.example/v1")

    def test_endpoint_identity_rejects_relative_endpoint(self) -> None:
        with pytest.raises(ValueError, match="absolute URL"):
            _identity(endpoint="models/v1")

    @pytest.mark.parametrize(
        "endpoint",
        [
            "",
            "https://models.example/v1?api_key=secret",
            "https://models.example/v1#secret",
        ],
    )
    def test_endpoint_identity_rejects_empty_or_nonidentity_data(
        self, endpoint: str
    ) -> None:
        with pytest.raises(ValueError, match="empty|query data or fragments"):
            _identity(endpoint=endpoint)

    def test_pool_requires_a_close_method(self) -> None:
        with pytest.raises(TypeError, match="provide close"):
            ConnectionPool(object(), _identity())


class TestConnectionPoolLifetime:
    @pytest.mark.anyio
    async def test_acquire_enters_shared_then_local_limiters(self) -> None:
        connection = _Connection()
        shared = anyio.CapacityLimiter(2)
        local = anyio.CapacityLimiter(1)
        pool = ConnectionPool(connection, _identity(), shared)

        async with pool.acquire(local) as acquired:
            assert acquired is connection
            assert pool.connection is connection
            assert pool.shared_limiter is shared
            assert shared.borrowed_tokens == 1
            assert local.borrowed_tokens == 1
        assert shared.borrowed_tokens == 0
        assert local.borrowed_tokens == 0

        connection.allow_close.set()
        await pool.aclose()

    @pytest.mark.anyio
    async def test_close_rejects_new_work_and_drains_entered_request(self) -> None:
        connection = _Connection()
        pool = ConnectionPool(connection, _identity())
        request_entered = anyio.Event()
        release_request = anyio.Event()

        async def request() -> None:
            async with pool.acquire():
                request_entered.set()
                await release_request.wait()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(request)
            await request_entered.wait()
            task_group.start_soon(pool.aclose)
            while not pool.is_closing:
                await anyio.sleep(0)

            with pytest.raises(RuntimeError, match="admission rejected"):
                async with pool.acquire():
                    pytest.fail("closing pool admitted new work")
            assert connection.close_calls == 0

            release_request.set()
            await connection.close_entered.wait()
            connection.allow_close.set()

        assert pool.is_closed
        assert connection.close_calls == 1

    @pytest.mark.anyio
    async def test_concurrent_and_repeated_close_closes_connection_once(self) -> None:
        connection = _Connection()
        pool = ConnectionPool(connection, _identity())

        async with anyio.create_task_group() as task_group:
            for _ in range(5):
                task_group.start_soon(pool.aclose)
            await connection.close_entered.wait()
            assert connection.close_calls == 1
            connection.allow_close.set()

        await pool.aclose()
        assert connection.close_calls == 1

    @pytest.mark.anyio
    async def test_async_context_manager_closes_owned_connection(self) -> None:
        connection = _Connection()
        connection.allow_close.set()
        pool = ConnectionPool(connection, _identity())

        async with pool as entered:
            assert entered is pool
            assert not pool.is_closing

        assert pool.is_closed
        assert connection.close_calls == 1

    @pytest.mark.anyio
    async def test_synchronous_connection_closer_is_supported(self) -> None:
        connection = _SyncConnection()
        pool = ConnectionPool(connection, _identity())

        await pool.aclose()

        assert pool.is_closed
        assert connection.close_calls == 1

    @pytest.mark.anyio
    async def test_close_error_is_persisted_and_re_raised(self) -> None:
        pool = ConnectionPool(_FailingConnection(), _identity())

        with pytest.raises(RuntimeError, match="close failed"):
            await pool.aclose()
        assert pool.is_closed
        with pytest.raises(RuntimeError, match="close failed"):
            await pool.aclose()

    @pytest.mark.anyio
    async def test_context_entry_is_rejected_after_close_starts(self) -> None:
        connection = _SyncConnection()
        pool = ConnectionPool(connection, _identity())
        await pool.aclose()

        with pytest.raises(RuntimeError, match="context entry rejected"):
            async with pool:
                pytest.fail("closed pool entered its context")
