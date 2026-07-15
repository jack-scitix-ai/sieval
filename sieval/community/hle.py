# Sources (Humanity's Last Exam, Center for AI Safety), pinned to commit
# 26dca2e253b405105b4c3d8c2f5af06f86f90c66:
#   - SYSTEM_PROMPT (verbatim) and the user-message assembly this task mirrors:
#     hle_eval/run_model_predictions.py (SYSTEM_PROMPT, format_message)
#   - JUDGE_PROMPT (verbatim), the ExtractedAnswer field set, and the accuracy /
#     confidence-interval / calibration-error formulas: hle_eval/run_judge_results.py
#     (JUDGE_PROMPT, ExtractedAnswer, dump_metrics)
#   - calib_err (verbatim): hle_eval/run_judge_results.py, itself vendored from
#     https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
"""Humanity's Last Exam (HLE) prompt + grading assets.

HLE (Center for AI Safety) is a ~2,500-question, multi-domain, closed-ended
academic benchmark graded by an LLM judge. This module vendors the upstream
model system prompt, the judge prompt, and the metric kernel (accuracy, 95%
Wald confidence interval, and calibration error).

Deviations from upstream (``hle_eval/run_judge_results.py`` @ 26dca2e):

* Judge invocation. Upstream calls ``client.beta.chat.completions.parse`` with
  the ``ExtractedAnswer`` Pydantic ``response_format`` (server-enforced
  structured output). sieval reaches the judge through the generic ``ChatModel``
  (``chat.completions`` returning text), so ``parse_judge`` reads the judge's
  reply textually for the ``correct`` (yes/no) and ``confidence`` fields the
  prompt already specifies. Why: the judge runs on an arbitrary ``api_base`` and
  not every endpoint enforces a JSON schema — parsing the prompted field format
  keeps the judge endpoint-agnostic. Correctness follows upstream semantics
  (``"yes" in judge_response["correct"]``); an unparseable reply is scored
  incorrect with confidence 100 (upstream's "no confidence score" default).
* ``aggregate_metrics`` returns a dict instead of printing, and guards
  ``len(confidence) < beta``: the vendored ``calib_err`` indexes ``bins[-1]`` and
  raises ``IndexError`` below ``beta`` judged records — unreachable for the full
  HLE test set (~2.5k) but reachable via slices/tests, where it reports a
  calibration error of 0.0. Numerator (correct count) and calibration arrays are
  built only from judged attempts while the denominator ``n`` is supplied by the
  caller (finals + failures), mirroring upstream where unjudged/failed
  predictions stay in ``n`` but out of the ``correct`` array.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import math
import re

import numpy as np

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

# Target bin size for calibration error (upstream default).
BETA = 100

_CORRECT_RE = re.compile(r"correct\s*:\s*\**\s*(yes|no)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"confidence\s*:\s*\**\s*(\d+)", re.IGNORECASE)


def parse_judge(reply: str) -> tuple[bool, int]:
    """Extract ``(correct, confidence)`` from a judge reply.

    The judge is prompted (see ``JUDGE_PROMPT``) to emit ``correct: yes|no`` and
    ``confidence: <int>`` fields. Both appear after the free-form ``reasoning``
    field, so the last match of each pattern wins. Follows upstream semantics
    (``"yes" in judge_response["correct"]``); an unrecognizable reply is scored
    incorrect with confidence 100 (upstream's "no confidence score" default).
    """
    correct_matches = _CORRECT_RE.findall(reply)
    correct = bool(correct_matches) and correct_matches[-1].lower() == "yes"

    confidence_matches = _CONFIDENCE_RE.findall(reply)
    confidence = int(confidence_matches[-1]) if confidence_matches else 100
    return correct, confidence


# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p="2", beta=100):
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p == "infty" or p == "infinity" or p == "max":
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def aggregate_metrics(
    correct: list[bool], confidence: list[int], n: int
) -> dict[str, float]:
    """Aggregate judged ``correct``/``confidence`` into HLE metrics.

    Mirrors upstream ``dump_metrics``: ``correct``/``confidence`` are the judged
    attempts (numerator + calibration arrays), while ``n`` is the full requested
    count (denominator) so failed/ungraded attempts count as incorrect.

    Returns accuracy (%), the 95% Wald half-width (``confidence_interval``, in
    percentage points), and ``calibration_error`` (0..100). ``n == 0`` yields
    all-zero metrics; fewer than ``BETA`` judged attempts yields
    ``calibration_error`` 0.0 (see module docstring).
    """
    if n == 0:
        return {"accuracy": 0.0, "confidence_interval": 0.0, "calibration_error": 0.0}

    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_interval = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)

    if len(confidence) >= BETA:
        confidence_arr = np.array(confidence) / 100
        correct_arr = np.array(correct)
        calibration_error = 100 * round(
            float(calib_err(confidence_arr, correct_arr, p="2", beta=BETA)), 2
        )
    else:
        calibration_error = 0.0

    return {
        "accuracy": accuracy,
        "confidence_interval": confidence_interval,
        "calibration_error": calibration_error,
    }
