"""OpenBookQA k-shot completion-format generative task (instruct or base, gen path).

Same prompt template, few-shot assembly, and answer extraction as the chat
variant ``openbookqa_kshot_gen`` (all shared from that module and
``sieval.community.openbookqa``). The only difference: the assembled prompt is
sent to a completion (``GenModel``) endpoint as raw text, so no chat template is
applied. This mirrors how the Phi family reports OBQA "under completion format".

The prompt template and extractor originate from OpenCompass ``obqa_gen_9069e4``
(same as the chat variant). Deviations from that reference:
  - OpenCompass ``GenInferencer`` is templated and 0-shot (``ZeroRetriever``);
    here the prompt is fed as raw completion text and k>0 is supported (sieval
    extension, fixed first-k train rows).
  - The Phi "completion format" pipeline is an unpublished internal tool — this
    approximates the completion-format idea, it is not a port of it.
  - ``STOP_SEQUENCES`` is a sieval choice coupled to the few-shot block layout
    (each example begins with "Question:"), not an upstream constant.

Repro decoding: greedy ``temperature=0``, ``top_p=1``; ``max_gen_toks`` follows
the model/run config. Generation is bounded by ``STOP_SEQUENCES``.

AI-Generated Code - Opus 4.8 (Anthropic)
"""

from typing import override

from sieval.community.openbookqa import OBQA_OPTIONS, first_option_postprocess
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import OpenBookQADatasetSample
from sieval.tasks.openbookqa_kshot_gen import (
    DEFAULT_N_SHOT,
    Feedback,
    build_fewshot_prefix,
    format_question,
)

# Coupled to the few-shot block: examples are separated by "\n\n" and each
# begins with "Question:", so a completion model's run-on after the answer is
# cut at the next fabricated example. Chat end tokens guard instruct models.
STOP_SEQUENCES = ("Question:", "</s>", "<|im_end|>")


@sieval_task(
    name="openbookqa_kshot_base_gen",
    display_name="OpenBookQA (k-shot, completion generative)",
    description="OpenBookQA elementary-science MCQ, completion-format extraction.",
    eval_mode=EvalMode.GEN,
    n_shot=DEFAULT_N_SHOT,
    tags=("english", "science", "multiple-choice"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="opencompass",
        url="https://github.com/open-compass/opencompass/blob/5767b74899806c0c37efdc5529ffea01e7340e48/opencompass/configs/datasets/obqa/obqa_gen_9069e4.py",
        notes=(
            "Prompt template and first_option_postprocess vendored from "
            "OpenCompass; fed in completion format (no chat template) rather "
            "than via GenInferencer. k>0 is a sieval extension."
        ),
    ),
)
class OpenBookQAFewShotBaseGenTask(
    Task[
        OpenBookQADatasetSample,
        str,
        ModelOutput,
        str,
        Feedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = DEFAULT_N_SHOT,
        fewshot_split: str = "train",
        stop: tuple[str, ...] = STOP_SEQUENCES,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._fewshot_split = fewshot_split
        self._stop = stop
        self._fewshot_prefix: str | None = None

    @override
    async def setup(self) -> None:
        self._fewshot_prefix = build_fewshot_prefix(
            self.dataset, self._k, self._fewshot_split
        )

    @override
    async def preprocess(self, raw, ctx):
        prefix = self._fewshot_prefix if self._fewshot_prefix is not None else ""
        return prefix + format_question(raw)

    @override
    async def infer(self, pre, ctx):
        kwargs: dict[str, object] = {}
        if self._stop:
            kwargs["stop"] = list(self._stop)
        return await self.model.agenerate(pre, **kwargs)

    @override
    async def postprocess(self, inf, ctx):
        # n=1, only one choice
        return first_option_postprocess(inf.texts[0], OBQA_OPTIONS)

    @override
    async def feedback(self, post, ctx):
        answer = ctx.raw_sample["answerKey"]
        return True, {"correct": post == answer, "pred": post, "answer": answer}

    @override
    async def report(self, finals, fails):
        correct = sum(1 for ctx in finals if ctx.feedback_result["correct"])
        accuracy = 100 * correct / len(finals) if finals else 0.0
        # Mirror the chat variant's report shape exactly (sibling consistency).
        return {"score": accuracy, "fails": len(fails), "accuracy": accuracy}
