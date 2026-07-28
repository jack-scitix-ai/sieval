"""
Unit tests for the ARC-Easy few-shot conditional-log-prob task (options).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import Request, Response, TopKEntry
from sieval.core.models.gen_model import GenModel
from sieval.core.models.transports import OpenAICompletionsTransport
from sieval.core.tasks import EvalMode, TaskContext
from sieval.core.tasks.meta import get_task_meta
from sieval.datasets.arc_easy import ARCEasyDataset, ARCEasyDatasetSample
from sieval.tasks.arc.arc_easy_kshot_clp import ARCEasyFewShotClpTask
from tests.conftest import HandlerTransport


class _TopLogprobsGenModel(GenModel):
    def __init__(self, top: dict[str, float]):
        self._top = top
        super().__init__(model="mock-gen", api_key="fake")

    def _build_default_transport(self) -> HandlerTransport:
        return HandlerTransport(
            self._stub_arun, OpenAICompletionsTransport.CAPABILITIES
        )

    async def _stub_arun(self, req: Request) -> Response:
        if not (req.return_logprobs or req.score_input):
            return Response(texts=("",))
        return Response(
            texts=("B",),
            top_logprobs=(
                tuple(TopKEntry(token=t, logprob=lp) for t, lp in self._top.items()),
            ),
        )


def _sample() -> ARCEasyDatasetSample:
    return {
        "question": "Which object is hottest?",
        "choices": ["ice", "fire", "snow"],
        "answer": 1,
    }


def _task(top: dict[str, float]) -> ARCEasyFewShotClpTask:
    dataset = ARCEasyDataset(
        _hf_dict=HFDatasetDict({"test": HFDataset.from_list([dict(_sample())])})
    )
    return ARCEasyFewShotClpTask(dataset, _TopLogprobsGenModel(top), n_shot=0)


@pytest.mark.anyio
async def test_preprocess_lists_options_with_letters():
    task = _task({})
    raw = _sample()

    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert pre["prompt"] == (
        "Question: Which object is hottest?\nA. ice\nB. fire\nC. snow\nAnswer:"
    )


@pytest.mark.anyio
async def test_argmax_over_option_letters():
    task = _task({" A": -3.0, " B": -0.1, " C": -2.0})  # gold index 1 -> "B"
    raw = _sample()
    ctx = TaskContext(sample_id=0, raw_sample=raw)
    pre = await task.preprocess(raw, ctx)
    inf = await task.infer(pre, ctx)
    post = await task.postprocess(inf, ctx)
    _finalize, feedback = await task.feedback(post, ctx)

    assert post["rollouts"][0]["prediction"] == 1
    assert feedback["rollouts"][0]["correct"] is True


def test_task_meta_points_to_arc_easy_dataset():
    meta = get_task_meta(ARCEasyFewShotClpTask)

    assert meta.name == "arc_easy_kshot_clp"
    assert meta.dataset == "arc_easy"
    assert meta.model_type == "gen"
    assert meta.n_shot == 25
    assert meta.eval_mode == EvalMode.CLP
