"""Humanity's Last Exam (HLE) — 0-shot generative, LLM-judge graded.

Generative port of HLE (Center for AI Safety; Phan et al., 2025). The model
answers a closed-ended, multi-domain academic question under the HLE system
prompt (an ``Explanation / Answer / Confidence`` format), and a separate **LLM
judge** decides whether the free-form answer matches the gold. Headline metric
is accuracy; the judge also extracts the model's confidence, from which a
calibration error is computed (alongside a 95% Wald confidence interval).

Subset: the **text-only** subset is graded by default (``text_only=True``);
technical reports mark full-set (text + image) numbers with ``*``. Image
questions carry a base64 data URI in the ``image`` column; ``text_only`` drops
them, and with ``text_only=False`` the image is attached as an ``image_url``
content block (requires a vision-capable candidate + judge).

Deviations from upstream (``hle_eval`` @ 26dca2e; see ``sieval.community.hle``):

* The upstream o1-only ``system``→``user`` role swap is dropped; the system
  prompt is always sent as ``system`` (correct for general instruct models).
* The judge is reached through ``ChatModel`` (text), not upstream's
  ``beta.chat.completions.parse`` structured output; its ``correct``/``confidence``
  fields are parsed from the reply (see ``sieval.community.hle.parse_judge``).
* Calibration error is guarded below the bin size for slices/tests (docs there).

Decoding params are model-layer, set via ``models:`` / ``infer_args`` — never by
this task. Upstream HLE defaults to ``temperature=0`` and advises
``max_completion_tokens>=8192`` for reasoning models; specific reproductions
override these (e.g. a technical report may evaluate at ``temperature=1.0``,
``top_p=0.95`` with a large token budget).

Grader is a REAL LLM supplied via the ``grader`` task arg on its own
``api_base``/``api_key``. Correctness depends on the judge endpoint's model
version (not pinnable like a Hub revision) — pin the grader model for
reproducibility; each sample's ``correct``, ``confidence`` and grader model id
are persisted in the feedback record.

Target: report against technical-report HLE numbers (e.g. the GLM series
evaluates the text-only subset with a strong LLM judge, such as GPT-5.2); the
grading protocol (judge model, subset) is report-specific, so cite it alongside
any comparison.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from collections.abc import Mapping
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from openai.types.chat import ChatCompletionMessageParam

from sieval.community.hle import (
    JUDGE_PROMPT,
    SYSTEM_PROMPT,
    aggregate_metrics,
    parse_judge,
)
from sieval.core.datasets import Dataset
from sieval.core.models import ChatModel, Model, ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import HLEDatasetSample


class JudgeFeedback(TypedDict):
    correct: bool
    confidence: int
    gold: str
    predicted: str
    grader_model: str


@sieval_task(
    name="hle_0shot_gen",
    display_name="Humanity's Last Exam (0-shot, generative)",
    description="Multi-domain frontier academic QA graded by an LLM judge.",
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "reasoning", "academic"),
    model_type="chat",
    deps_group="hle",
    status="stable",
    reference_impl=ReferenceImpl(
        source="hle",
        url="https://github.com/centerforaisafety/hle/tree/26dca2e253b405105b4c3d8c2f5af06f86f90c66/hle_eval",
        notes=(
            "Generative port of Humanity's Last Exam (Center for AI Safety). "
            "SYSTEM_PROMPT and JUDGE_PROMPT are vendored byte-for-byte; the judge "
            "runs through sieval's ChatModel (text) rather than upstream's "
            "beta.chat.completions.parse structured output, and its correct/"
            "confidence fields are parsed from the reply. Metrics mirror upstream "
            "dump_metrics: accuracy, a 95% Wald confidence interval, and "
            "calibration error (calib_err, p=2, beta=100). Text-only subset by "
            "default (text_only=True); full set (text_only=False) is marked * in "
            "reports. Grader is a REAL LLM (upstream default o3-mini-2025-01-31) "
            "supplied via the `grader` task arg on its own api_base/api_key. "
            "REPRODUCIBILITY: scores depend on the judge endpoint's model version "
            "(not pinnable like a Hub revision) — pin the grader model; the "
            "per-sample correct/confidence and grader model id are persisted. "
            "VALIDATION: gpt-oss-20b scored 12.14 / 3.61 (reasoning=high / low, "
            "judge GPT-5.2, text-only, no tools) vs the gpt-oss model card "
            "(arXiv:2508.10925) 10.9 / 4.2 — within <3pp."
        ),
    ),
)
class HLEZeroShotGenTask(
    Task[
        HLEDatasetSample,
        list[ChatCompletionMessageParam],
        ModelOutput,
        list[str],
        list[JudgeFeedback],
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        grader: Mapping | Model | None = None,
        n: int = 1,
        text_only: bool = True,
    ):
        if text_only:
            dataset = self._select_text_only(dataset)
        super().__init__(dataset=dataset, model=model, name=name)
        self._n = n
        self._grader = self._build_grader(grader)

    @staticmethod
    def _select_text_only(
        dataset: Dataset[HLEDatasetSample],
    ) -> Dataset[HLEDatasetSample]:
        """Return a clone keeping only text-only questions (empty ``image``).

        ``image`` is a string column (``""`` when absent). The filter reads only
        that column (``input_columns=["image"]``) so HF never materializes the
        sibling ``image_preview`` / ``rationale_image`` ``Image`` features, which
        default to ``decode=True`` and would otherwise pull in Pillow. Raises if
        the filter empties the ``test`` split — a signal that the ``image``
        column is missing or every question is multi-modal.
        """
        source = dataset.dataset_dict
        filtered = HFDatasetDict(
            {
                split: ds.filter(lambda image: not image, input_columns=["image"])
                for split, ds in source.items()
            }
        )
        if "test" in filtered and len(filtered["test"]) == 0:
            raise ValueError(
                "HLE text-only selection produced an empty 'test' split; the "
                "source may lack the 'image' column or contain only multi-modal "
                "questions."
            )
        return type(dataset)(_hf_dict=filtered)

    @staticmethod
    def _build_grader(grader: Mapping | Model | None) -> Model:
        """Resolve the ``grader`` task arg into a Model.

        Accepts a pre-built Model (tests / advanced configs) or a model-config
        mapping (the YAML path, e.g. ``{model: gpt-5.2, api_base: ...}``).
        Grading is mandatory — there is no deterministic fallback — so ``None``
        raises.
        """
        if isinstance(grader, Model):
            return grader
        if isinstance(grader, Mapping):
            return ChatModel(**grader)
        raise ValueError(
            "HLE requires an LLM judge. Pass `grader:` in the task args — a "
            "model-config dict such as "
            "{model: gpt-5.2, api_base: ..., api_key: ..., reasoning_effort: medium}."
        )

    @override
    async def preprocess(self, raw, ctx):
        content: list[dict] = [{"type": "text", "text": raw["question"]}]
        if raw["image"]:  # "" when not multi-modal
            content.append({"type": "image_url", "image_url": {"url": raw["image"]}})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @override
    async def infer(self, pre, ctx):
        return await self.model.agenerate(pre, n=self._n)

    @override
    async def postprocess(self, inf, ctx):
        return list(inf.texts)

    @override
    async def feedback(self, post, ctx):
        raw = ctx.raw_sample
        question = raw["question"]
        gold = raw["answer"]
        grader_model = self._grader.meta()["model"]

        feedbacks: list[JudgeFeedback] = []
        for predicted in post:
            prompt = JUDGE_PROMPT.format(
                question=question,
                correct_answer=gold,
                response=predicted,
            )
            out = await self._grader.agenerate(prompt)
            reply = out.texts[0] if out.texts else ""
            correct, confidence = parse_judge(reply)
            feedbacks.append(
                {
                    "correct": correct,
                    "confidence": confidence,
                    "gold": gold,
                    "predicted": predicted,
                    "grader_model": grader_model,
                }
            )
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        correct = [fb["correct"] for f in finals for fb in (f.feedback_result or [])]
        confidence = [
            fb["confidence"] for f in finals for fb in (f.feedback_result or [])
        ]
        # Denominator spans the full requested set; pipeline failures (candidate
        # produced no gradeable answer) count as incorrect — matching upstream
        # (n = total questions) and the *_gen family, not just graded attempts.
        n = (len(finals) + len(fails)) * self._n
        # Length-capped attempts: a reasoning model can burn the whole token
        # budget and emit no answer, then get graded incorrect. Surface the
        # count so the headline is self-documenting on this collapse-prone
        # benchmark (upstream reports only accuracy).
        truncated = 0
        for f in finals:
            out = f.infer_result
            if out is None or out.finish_reasons is None:
                continue
            truncated += sum(reason == "length" for reason in out.finish_reasons)
        m = aggregate_metrics(correct, confidence, n)
        return {
            "score": m["accuracy"],
            "accuracy": m["accuracy"],
            "confidence_interval": m["confidence_interval"],
            "calibration_error": m["calibration_error"],
            "n": n,
            "n_graded": len(correct),
            "fails": len(fails),
            "truncated": truncated,
        }
