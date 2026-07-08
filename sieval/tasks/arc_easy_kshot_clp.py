"""
ARC-Easy few-shot base-model conditional-log-prob task (options format).

The "options" MCQ format (Wang et al. 2024, arXiv:2412.17758): the candidate
options are listed ``A/B/C/...`` in the prompt and the answer is the option
LETTER. Scoring reads the first output token's ``top_logprobs`` in ONE
inference and argmaxes over the option-letter log-probs — the ``clp`` protocol,
mirroring ``cmmlu_kshot_base_gen``. Scoring requires every option letter to be
present in the top-k and fails the sample otherwise, so partial coverage is
loud rather than a best-of-present guess (default ``logprobs=100``; SGLang
serves 100 by default, on vLLM start with ``--max-logprobs 100``).

This is the base-model options-format counterpart to the ``ppl`` separation task
(``arc_easy_kshot_ppl``, which scores full option text). DeepSeek switched from
separation to options after V1, so the DeepSeek-V3 number is the options target
here; the Qwen2.5 report uses separation (see the ppl sibling).

Comparison target: DeepSeek-V3 base ARC-Easy 25-shot EM = 98.4 (DeepSeek-V3
report, Table 3).

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

from typing import override

from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    sieval_task,
)
from sieval.datasets import ARCEasyDatasetSample

from ._arc import (
    DEFAULT_CLP_LOGPROBS,
    DEFAULT_FEWSHOT_SEED,
    ARCFeedback,
    arc_report,
    build_arc_clp_fewshot_prefix,
    choice_label,
    choice_text,
    clp_scores_from_top_logprobs,
    format_arc_clp_item,
    sample_arc_fewshot,
)

N_SHOT = 25


@sieval_task(
    name="arc_easy_kshot_clp",
    display_name="ARC-Easy (few-shot, conditional log-prob)",
    description="ARC-Easy few-shot options-format next-token letter accuracy.",
    eval_mode=EvalMode.CLP,
    n_shot=N_SHOT,
    tags=("english", "science", "multiple-choice", "base-model"),
    model_type="gen",
    reference_impl=ReferenceImpl(
        source="lm-evaluation-harness",
        url=(
            "https://github.com/EleutherAI/lm-evaluation-harness/blob/1dd931087362abba74e0375c8c631295559f48b2/lm_eval/tasks/arc/arc_easy.yaml"
        ),
        notes=(
            "Shares the ARC-Easy split/dataset/revision with "
            "lm-evaluation-harness. Uses the 'options' MCQ format (arXiv "
            "2412.17758): options listed A/B/C/... in the prompt, answer is the "
            "option letter, scored by one-call next-token top_logprobs argmax "
            "(the clp protocol; mirrors cmmlu_kshot_base_gen). Requires all "
            "option letters in the top-k and fails the sample otherwise "
            "(default logprobs=100; SGLang serves 100, on vLLM use "
            "--max-logprobs 100). Comparison target: DeepSeek-V3 base ARC-Easy "
            "25-shot EM = 98.4 (DeepSeek switched separation->options after V1; "
            "the ppl sibling reproduces the separation number)."
        ),
    ),
)
class ARCEasyFewShotClpTask(
    Task[
        ARCEasyDatasetSample,
        str,
        ModelOutput,
        int,
        ARCFeedback,
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        *,
        k: int = N_SHOT,
        logprobs: int = DEFAULT_CLP_LOGPROBS,
        fewshot_split: str = "train",
        fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if logprobs < 1:
            raise ValueError(f"logprobs must be >= 1, got {logprobs}")
        super().__init__(dataset=dataset, model=model, name=name)
        self._k = k
        self._logprobs = logprobs
        self._fewshot_split = fewshot_split
        self._fewshot_seed = fewshot_seed
        self._fewshot_prefix: str | None = None

    @override
    async def setup(self) -> None:
        # Built once here (setup runs before any preprocess) so the k-exemplar
        # prefix is not rejoined per sample.
        self._fewshot_prefix = self._build_fewshot_prefix()

    @override
    async def preprocess(self, raw, ctx):
        prefix = (
            self._fewshot_prefix
            if self._fewshot_prefix is not None
            else self._build_fewshot_prefix()
        )
        return prefix + format_arc_clp_item(raw["question"], raw["choices"])

    @override
    async def infer(self, pre, ctx):
        # One inference: the next-token distribution over the option letters.
        return await self.model.alogprobs(
            pre, max_tokens=1, logprobs=self._logprobs, echo=False
        )

    @override
    async def postprocess(self, inf, ctx):
        labels = [choice_label(i) for i in range(len(ctx.raw_sample["choices"]))]
        scores, all_present = clp_scores_from_top_logprobs(inf.top_logprobs, labels)
        if not all_present:
            missing = [label for label in labels if scores[label] == float("-inf")]
            raise RuntimeError(
                f"ARC-Easy top_logprobs missing option token(s) {missing}; "
                f"increase logprobs (got top-k of {self._logprobs}) or raise the "
                "server's max-logprobs so all option letters are returned."
            )
        best_label = max(scores.items(), key=lambda item: item[1])[0]
        return ord(best_label) - ord("A")

    @override
    async def feedback(self, post, ctx):
        answer = ctx.raw_sample["answer"]
        choices = ctx.raw_sample["choices"]
        return True, {
            "correct": post == answer,
            "answer": answer,
            "prediction": post,
            "answer_choice": choice_text(choices, answer),
            "prediction_choice": choice_text(choices, post),
        }

    @override
    async def report(self, finals, fails):
        return arc_report(finals, fails)

    def _build_fewshot_prefix(self) -> str:
        examples = sample_arc_fewshot(
            self.dataset, self._k, self._fewshot_split, self._fewshot_seed
        )
        return build_arc_clp_fewshot_prefix(examples)
