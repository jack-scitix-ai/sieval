"""Unit tests for the OpenBookQA k-shot completion-format generative task.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.gen_model import GenModel
from sieval.core.tasks import TaskContext
from sieval.datasets.openbookqa import OpenBookQADataset, OpenBookQADatasetSample
from sieval.tasks.openbookqa_kshot_base_gen import (
    STOP_SEQUENCES,
    OpenBookQAFewShotBaseGenTask,
)
from sieval.tasks.openbookqa_kshot_gen import build_fewshot_prefix, format_question


class _CapturingGenModel(GenModel):
    def __init__(self):
        super().__init__(model="mock-gen", api_key="fake")
        self.last_kwargs: dict[str, object] = {}
        self.last_prompt: object = None

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=["The answer is A."])


def _sample(stem: str, answer_key: str = "A") -> OpenBookQADatasetSample:
    return {
        "id": f"id-{stem}",
        "question_stem": stem,
        "choices": {"text": [f"{stem}-a", f"{stem}-b", f"{stem}-c", f"{stem}-d"]},
        "answerKey": answer_key,
    }


def _dataset(train: list[OpenBookQADatasetSample]) -> OpenBookQADataset:
    return OpenBookQADataset(
        _hf_dict=HFDatasetDict(
            {
                "train": HFDataset.from_list([dict(s) for s in train]),
                "test": HFDataset.from_list([dict(_sample("q-test"))]),
            }
        )
    )


def test_stop_sequences_pinned():
    # Pin the constant so accidental drift in the few-shot boundary is caught.
    assert STOP_SEQUENCES == ("Question:", "</s>", "<|im_end|>")


@pytest.mark.anyio
async def test_preprocess_returns_raw_string_not_chat_messages():
    # The whole point of the base variant: a completion string, no chat wrapper.
    dataset = _dataset([_sample("q-train", "B")])
    task = OpenBookQAFewShotBaseGenTask(dataset, _CapturingGenModel(), k=0)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert isinstance(pre, str)
    assert pre == format_question(raw)


@pytest.mark.anyio
async def test_prompt_construction_matches_chat_variant():
    # Base and chat variants must assemble byte-identical prompts; only the
    # delivery (string vs chat turn) differs.
    train = [_sample("q0", "A"), _sample("q1", "C"), _sample("q2", "D")]
    dataset = _dataset(train)
    task = OpenBookQAFewShotBaseGenTask(dataset, _CapturingGenModel(), k=2)
    await task.setup()

    raw = _sample("q-test")
    pre = await task.preprocess(raw, TaskContext(sample_id=0, raw_sample=raw))

    expected = build_fewshot_prefix(dataset, 2, "train") + format_question(raw)
    assert pre == expected
    assert "q2" not in pre  # third train row excluded at k=2


@pytest.mark.anyio
async def test_infer_forwards_stop_but_not_decoding_params():
    dataset = _dataset([_sample("q0")])
    model = _CapturingGenModel()
    task = OpenBookQAFewShotBaseGenTask(dataset, model, k=0)

    await task.infer("prompt", TaskContext(sample_id=0, raw_sample=_sample("q-test")))

    assert model.last_kwargs.get("stop") == list(STOP_SEQUENCES)
    for forbidden in ("temperature", "top_p", "max_tokens", "n"):
        assert forbidden not in model.last_kwargs


@pytest.mark.anyio
async def test_feedback_and_report_shape_matches_chat():
    dataset = _dataset([_sample("q0")])
    task = OpenBookQAFewShotBaseGenTask(dataset, _CapturingGenModel(), k=0)

    correct_raw = _sample("q-test", "A")
    wrong_raw = _sample("q-test", "B")
    post = await task.postprocess(
        ModelOutput(model=task.model.meta(), texts=["The answer is A."]),
        TaskContext(sample_id=0, raw_sample=correct_raw),
    )
    assert post == "A"

    _, fb_ok = await task.feedback(
        post, TaskContext(sample_id=0, raw_sample=correct_raw)
    )
    _, fb_bad = await task.feedback(
        post, TaskContext(sample_id=1, raw_sample=wrong_raw)
    )
    finals = [
        TaskContext(sample_id=0, raw_sample=correct_raw, feedback_result=fb_ok),
        TaskContext(sample_id=1, raw_sample=wrong_raw, feedback_result=fb_bad),
    ]
    report = await task.report(finals, [])

    assert report["score"] == 50.0
    assert report["accuracy"] == 50.0
    assert report["fails"] == 0
    assert isinstance(report["fails"], int)


def test_negative_k_rejected():
    dataset = _dataset([_sample("q0")])
    with pytest.raises(ValueError, match="k must be >= 0"):
        OpenBookQAFewShotBaseGenTask(dataset, _CapturingGenModel(), k=-1)
