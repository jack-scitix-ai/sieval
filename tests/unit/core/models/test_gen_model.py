"""Shell tests: backend selector wiring for GenModel.

RFC #25 moved the completions wire logic (streaming accumulation, echo split,
``_completion_top_logprobs`` sanitizing) into ``OpenAICompletionsTransport``;
the former ``_agenerate_impl`` / ``_alogprobs_impl`` coverage moved with it to
tests/unit/core/models/transports/test_openai_completions.py, and the
request-builder validation (n/stream types, alogprobs n=1) lives on ``Model``
in tests/unit/core/models/test_model.py. What remains here is the selector
contract: GenModel pairs the shared client with that transport, which supplies
the model's capabilities.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from sieval.core.models import Capability, GenModel
from sieval.core.models.transports.openai_completions import (
    OpenAICompletionsTransport,
)


class TestDefaultTransport:
    def test_builds_openai_completions_transport(self):
        m = GenModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAICompletionsTransport)
        assert Capability.InputScoring in m.capabilities

    def test_transport_bound_to_shared_client_and_model(self):
        m = GenModel(model="m", api_key="k")
        assert isinstance(m._transport, OpenAICompletionsTransport)
        assert m._transport._client is m._client
        assert m._transport._model == "m"
