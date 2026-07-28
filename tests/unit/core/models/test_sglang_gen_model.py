"""Shell tests: backend selector wiring for SglangGenModel.

RFC #25 moved the native /generate wire logic (URL derivation, sampling-param
translation, triple parsing, ``_normalize_token_text``, the radix-cache guard)
into ``SglangTransport``; the former ``_agenerate_impl`` / ``_alogprobs_impl``
coverage moved with it to tests/unit/core/models/transports/test_sglang.py,
and the request-builder validation (n/stream types, alogprobs n=1) lives on
``Model`` in tests/unit/core/models/test_model.py. What remains here is the
selector contract: SglangGenModel pairs the shared client (and its api_base)
with that transport, which supplies the model's capabilities.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from sieval.core.models import Capability, SglangGenModel
from sieval.core.models.transports.sglang import SglangTransport


class TestDefaultTransport:
    def test_builds_sglang_transport(self):
        m = SglangGenModel(model="m", api_base="http://host:8000/v1", api_key="local")
        assert isinstance(m._transport, SglangTransport)
        assert Capability.SampledLogprobsWithTokenIds in m.capabilities

    def test_transport_bound_to_shared_client_model_and_api_base(self):
        m = SglangGenModel(model="m", api_base="http://host:8000/v1", api_key="local")
        assert m._transport._client is m._client
        assert m._transport._model == "m"
        assert m._transport._api_base == "http://host:8000/v1"
