"""SglangGenModel: native sglang ``/generate`` backend for text + logprobs.

sglang's OpenAI ``/v1/completions`` endpoint rejects ``echo=True`` together
with ``logprobs``, so PPL-style scoring (ARC/HellaSwag read the logprob of an
answer token appended to the prompt; CMMLU/MMLU-Base read the first output
token's top-k) cannot go through it. This backend speaks sglang's native
``/generate`` protocol for BOTH generation and logprob extraction, so a single
object talks one wire protocol end-to-end.

RFC #25 moved the wire logic (param translation, triple parsing, token-text
normalization, radix-cache guard) into
:class:`~sieval.core.models.transports.sglang.SglangTransport`; this class is
the backend selector that pairs the shared client/limiter pool with that
transport.

It extends ``Model[str]`` rather than ``GenModel`` deliberately: the only thing
``GenModel`` would contribute is its OpenAI-completions transport, which is a
different protocol than the ``/generate`` logprob path — incidental reuse, not
coupling. The genuinely shared infrastructure (OpenAI async client, limiters,
``with_args``/``meta``, the ``arun``/``agenerate``/``alogprobs`` surface) lives
in ``Model`` and is inherited directly.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

from .model import Model
from .transport import Transport


class SglangGenModel(Model[str]):
    """Model backend reading text and logprobs from sglang native ``/generate``."""

    def _build_default_transport(self) -> Transport:
        from .transports.sglang import SglangTransport

        return SglangTransport(
            client=self._client, model=self._model, api_base=self._api_base
        )
