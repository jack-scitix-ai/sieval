"""Unit tests for the HLE 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import numpy as np
import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.community.hle import (
    JUDGE_PROMPT,
    SYSTEM_PROMPT,
    aggregate_metrics,
    calib_err,
    parse_judge,
)
from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext
from sieval.datasets.hle import HLEDataset
from sieval.tasks.hle_0shot_gen import HLEZeroShotGenTask, JudgeFeedback


class _ScriptedChatModel(ChatModel):
    """ChatModel returning a fixed reply, recording the last agenerate kwargs."""

    def __init__(self, reply: str, model: str = "mock"):
        super().__init__(model=model, api_key="fake")
        self._reply = reply
        self.last_kwargs: dict[str, object] = {}

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = prompt
        self.last_kwargs = dict(kwargs)
        return ModelOutput(model=self.meta(), texts=[self._reply])

    async def _alogprobs_impl(
        self, prompt, *, max_tokens=1, logprobs=5, echo=True, temperature=0.0, **kwargs
    ) -> ModelOutput:
        _ = (prompt, max_tokens, logprobs, echo, temperature, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _row(image: str = "") -> dict:
    return {
        "id": "q1",
        "question": "What is 2 + 2?",
        "image": image,
        "answer": "4",
        "answer_type": "exactMatch",
        "author_name": "author",
        "rationale": "",
        "raw_subject": "Math",
        "category": "Math",
    }


def _dataset(rows: list[dict]) -> HLEDataset:
    return HLEDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list(rows)}))


def _task(
    grader_reply: str = "correct: yes\nconfidence: 90",
    *,
    rows: list[dict] | None = None,
    text_only: bool = True,
    n: int = 1,
):
    dataset = _dataset(rows if rows is not None else [_row()])
    model = _ScriptedChatModel(reply="Answer: 4", model="candidate")
    grader = _ScriptedChatModel(reply=grader_reply, model="judge-5.2")
    task = HLEZeroShotGenTask(dataset, model, grader=grader, n=n, text_only=text_only)
    return task, model, grader


# --- grader is mandatory; no deterministic fallback ---


def test_build_grader_requires_config():
    with pytest.raises(ValueError, match="requires an LLM judge"):
        HLEZeroShotGenTask._build_grader(None)


def test_build_grader_accepts_mapping_and_model():
    built = HLEZeroShotGenTask._build_grader({"model": "gpt-5.2", "api_key": "fake"})
    assert isinstance(built, ChatModel)
    existing = _ScriptedChatModel(reply="x")
    assert HLEZeroShotGenTask._build_grader(existing) is existing


# --- text-only selection drops image questions ---


def test_text_only_keeps_only_text_questions():
    task, _, _ = _task(rows=[_row(), _row(image="data:image/png;base64,AAAA")])
    assert task.dataset.test_set is not None
    assert len(task.dataset.test_set) == 1
    assert task.dataset.test_set[0]["image"] == ""


def test_full_set_keeps_image_questions():
    task, _, _ = _task(
        rows=[_row(), _row(image="data:image/png;base64,AAAA")], text_only=False
    )
    assert task.dataset.test_set is not None
    assert len(task.dataset.test_set) == 2


def test_text_only_all_multimodal_raises():
    with pytest.raises(ValueError, match="empty 'test' split"):
        _task(rows=[_row(image="data:image/png;base64,AAAA")])


# --- preprocess: HLE system prompt + user content blocks (mirrors format_message) ---


@pytest.mark.anyio
async def test_preprocess_text_only_message():
    task, _, _ = _task()
    messages = await task.preprocess(_row(), TaskContext(sample_id=0))
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
    ]


@pytest.mark.anyio
async def test_preprocess_attaches_image_block():
    task, _, _ = _task(text_only=False)
    raw = _row(image="data:image/png;base64,AAAA")
    messages = await task.preprocess(raw, TaskContext(sample_id=0))
    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "text", "text": "What is 2 + 2?"}
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }


# --- infer forwards ONLY n (no decode-param injection) ---


@pytest.mark.anyio
async def test_infer_forwards_only_n():
    task, model, _ = _task(n=1)
    await task.infer([{"role": "user", "content": "q"}], TaskContext(sample_id=0))
    # Decode params (temperature/top_p/max_tokens) must come from the model
    # layer, never from the task — infer passes n and nothing else.
    assert model.last_kwargs == {"n": 1}


# --- feedback: parse judge correct/confidence, record provenance ---


@pytest.mark.anyio
async def test_feedback_parses_correct_and_confidence():
    task, _, _ = _task(grader_reply="correct: yes\nconfidence: 90")
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    finalize, feedbacks = await task.feedback(["Answer: 4"], ctx)

    assert finalize is True
    fb: JudgeFeedback = feedbacks[0]
    assert fb["correct"] is True
    assert fb["confidence"] == 90
    assert fb["gold"] == "4"
    assert fb["predicted"] == "Answer: 4"
    assert fb["grader_model"] == "judge-5.2"


@pytest.mark.anyio
async def test_feedback_unparseable_reply_is_incorrect_default_confidence():
    task, _, _ = _task(grader_reply="the judge rambled without the fields")
    ctx = TaskContext(sample_id=0, raw_sample=_row())
    _, feedbacks = await task.feedback(["whatever"], ctx)
    assert feedbacks[0]["correct"] is False
    assert feedbacks[0]["confidence"] == 100


# --- report: accuracy over the full requested set (fails in denominator) ---


def _finals(grades: list[tuple[bool, int]]) -> list[TaskContext]:
    return [
        TaskContext(
            sample_id=i,
            feedback_result=[
                {
                    "correct": c,
                    "confidence": conf,
                    "gold": "",
                    "predicted": "",
                    "grader_model": "m",
                }
            ],
        )
        for i, (c, conf) in enumerate(grades)
    ]


@pytest.mark.anyio
async def test_report_accuracy_and_counts_fails_in_denominator():
    task, _, _ = _task()  # n=1
    finals = _finals([(True, 90), (False, 40)])
    fails = [TaskContext(sample_id=10)]
    report = await task.report(finals, fails)

    # n = (2 finals + 1 fail) * 1 = 3; 1 correct => 33.33%.
    # The old len(finals)=2 denominator would give 50.0, so this discriminates.
    assert report["n"] == 3
    assert report["n_graded"] == 2
    assert report["fails"] == 1
    # No infer_result on these contexts -> no truncation surfaced.
    assert report["truncated"] == 0
    assert report["accuracy"] == pytest.approx(33.33, abs=1e-2)
    assert report["score"] == report["accuracy"]


@pytest.mark.anyio
async def test_report_fails_weighted_by_n():
    task, _, _ = _task(n=2)
    # One finalized sample carrying its n=2 judged attempts (both correct).
    fb = {
        "correct": True,
        "confidence": 90,
        "gold": "",
        "predicted": "",
        "grader_model": "m",
    }
    finals = [TaskContext(sample_id=0, feedback_result=[fb, dict(fb)])]
    fails = [TaskContext(sample_id=5)]
    report = await task.report(finals, fails)
    # n = (1 final + 1 fail) * 2 = 4; 2 correct => 50.0%.
    # An unweighted (n=1) denominator would give 3 and 66.67%.
    assert report["n"] == 4
    assert report["n_graded"] == 2
    assert report["accuracy"] == pytest.approx(50.0)


@pytest.mark.anyio
async def test_report_counts_truncated_outputs():
    # A length-capped attempt (finish_reason "length") is surfaced as
    # `truncated` so the accuracy headline is self-documenting on this
    # collapse-prone benchmark. Counted per-attempt from infer_result.
    task, model, _ = _task()  # n=1
    meta = model.meta()
    fb = {
        "correct": False,
        "confidence": 100,
        "gold": "",
        "predicted": "",
        "grader_model": "m",
    }
    finals = [
        TaskContext(
            sample_id=0,
            infer_result=ModelOutput(model=meta, texts=[""], finish_reasons=["length"]),
            feedback_result=[dict(fb)],
        ),
        TaskContext(
            sample_id=1,
            infer_result=ModelOutput(
                model=meta, texts=["Answer: 4"], finish_reasons=["stop"]
            ),
            feedback_result=[{**fb, "correct": True}],
        ),
    ]
    report = await task.report(finals, [])
    assert report["truncated"] == 1
    assert report["n_graded"] == 2


@pytest.mark.anyio
async def test_report_empty_is_zero():
    task, _, _ = _task()
    report = await task.report([], [])
    assert report["n"] == 0
    assert report["accuracy"] == 0.0
    assert report["calibration_error"] == 0.0
    assert report["truncated"] == 0


# --- prompt fidelity: byte-for-byte pins on the vendored HLE prompts ---
# These lock the reproduction invariant so any drift from upstream
# (centerforaisafety/hle @ 26dca2e) fails loudly. `test_preprocess_*` above
# compare against the constants by reference and cannot catch such drift.


def test_system_prompt_pinned():
    assert SYSTEM_PROMPT == (
        "Your response should be in the following format:\n"
        "Explanation: {your explanation for your answer choice}\n"
        "Answer: {your chosen answer}\n"
        "Confidence: {your confidence score between 0% and 100% for your answer}"
    )


def test_judge_prompt_pinned():
    # Upstream ships a duplicated-word typo and pipe-escaped percent signs;
    # both are preserved verbatim.
    assert "i.e. if there if there is any inconsistency" in JUDGE_PROMPT
    assert r"confidence score between 0|\%| and 100|\%| from [response]" in JUDGE_PROMPT
    assert "extracted_final_answer:" in JUDGE_PROMPT
    for field in ("{question}", "{response}", "{correct_answer}"):
        assert field in JUDGE_PROMPT


# --- metric kernel: parse_judge, calib_err, aggregate_metrics ---


def test_parse_judge_last_field_wins():
    assert parse_judge("correct: yes\nconfidence: 85") == (True, 85)
    assert parse_judge("correct: no") == (False, 100)
    assert parse_judge("no recognizable fields") == (False, 100)
    # reasoning may mention "correct:"; the trailing field value wins.
    assert parse_judge("reasoning: correct: no\ncorrect: yes\nconfidence: 30") == (
        True,
        30,
    )
    # `\b` anchor: "incorrect: yes" must NOT be read as the `correct` field.
    # Without the anchor the substring "correct: yes" would match -> (True, 100);
    # with no real verdict field the parse must default to (False, 100).
    assert parse_judge("extracted_final_answer: 42 is incorrect: yes") == (False, 100)


def test_calib_err_matches_hand_computation():
    # beta=2 forces two bins over four samples so the first bin is scored
    # (upstream excludes the final bin via range(len(bins) - 1)).
    confidence = np.array([0.1, 0.2, 0.9, 0.95])
    correct = np.array([0, 0, 1, 1])
    # bin[0] conf mean 0.15, correct mean 0 -> diff 0.15;
    # cerr = sqrt(2/4 * 0.15**2) = 0.106066...
    assert calib_err(confidence, correct, p="2", beta=2) == pytest.approx(
        0.106066, abs=1e-5
    )


def test_aggregate_metrics_accuracy_ci_and_calibration_guard():
    # 1 correct of n=4 -> 25.0%; Wald half-width = 1.96*sqrt(25*75/4) = 42.44.
    m = aggregate_metrics([True, False], [100, 50], n=4)
    assert m["accuracy"] == pytest.approx(25.0)
    assert m["confidence_interval"] == pytest.approx(42.44, abs=1e-2)
    # Fewer than BETA judged records -> calibration guarded to 0.0.
    assert m["calibration_error"] == 0.0


def test_aggregate_metrics_zero_n():
    assert aggregate_metrics([], [], n=0) == {
        "accuracy": 0.0,
        "confidence_interval": 0.0,
        "calibration_error": 0.0,
    }
