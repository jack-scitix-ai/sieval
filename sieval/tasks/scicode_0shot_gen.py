"""SciCode 0-shot task for instruct/chat models.

Each sample is one main problem decomposed into dependent sub-steps. Generation
is sequential *within* a problem: the prompt for step *i* embeds the model's own
code from steps ``1..i-1`` (upstream's default self-dependency setting, not gold
context). Different problems still run concurrently as separate samples.

Evaluation mirrors upstream ``test_generated_code.py`` — per step, concatenate
``required_dependencies`` + prior-step functions + current-step function + the
step's test cases, then execute — but the numeric targets that upstream reads
from ``test_data.h5`` inside the sandbox are read here on the eval side and
inlined into the program, keeping sieval's code-eval sandbox stateless. Three
scientist-authored steps (13.6, 62.1, 76.3) are not generated or tested; their
gold code is used only as context for later steps.

Metrics: sub-problem accuracy (passing steps / tested steps) and main-problem
accuracy (problems whose every tested step passes) — the headline resolve rate.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import os
import time
from typing import TypedDict, override

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.scicode import (
    build_test_program,
    encode_targets,
    extract_python_script,
    generate_prompt_with_steps,
    is_special_step,
    process_hdf5_to_tuple,
    special_step_code,
)
from sieval.core.models import ModelOutput
from sieval.core.tasks import (
    EvalMode,
    ReferenceImpl,
    Task,
    TaskStageOutput,
    sieval_task,
)
from sieval.core.utils.meta import build_stage_meta
from sieval.datasets import SciCodeDatasetSample


class StepCode(TypedDict):
    step_number: str
    tested: bool
    # dependencies + prior-step funcs + current-step func; None for special steps
    code_content: str | None
    # Raw model response for this step, kept for provenance/debugging. Empty for
    # special (gold) steps that are not generated.
    raw_response: str
    # True when extract_python_script found no code in raw_response — the step
    # (and any later step depending on it) will fail; surfaced in the report.
    empty_extraction: bool


class StepProgram(TypedDict):
    step_number: str
    program: str
    empty_extraction: bool


class StepFeedback(TypedDict):
    step_number: str
    correct: bool
    msg: str
    empty_extraction: bool


@sieval_task(
    name="scicode_0shot_gen",
    display_name="SciCode (0-shot, generative)",
    description="SciCode — research coding benchmark with dependent sub-steps; sub-problem and main-problem accuracy.",  # noqa: E501
    eval_mode=EvalMode.GEN,
    n_shot=0,
    tags=("english", "python", "code-exec"),
    model_type="chat",
    deps_group="scicode",
    reference_impl=ReferenceImpl(
        source="scicode",
        url="https://github.com/scicode-bench/SciCode/tree/69a8cfc829fe8788a426ce8b5de6292366dce7ef/eval/scripts",
        notes=(
            "Prompt assembly (gencode_json.py), code/h5 parsers, comparison "
            "helpers, and the 3 gold steps vendored into community/scicode. "
            "h5 targets are inlined into the sandbox program instead of read via "
            "process_hdf5_to_tuple in-sandbox."
        ),
    ),
)
class SciCodeZeroShotGenTask(
    Task[
        SciCodeDatasetSample,
        SciCodeDatasetSample,
        TaskStageOutput[list[StepCode]],
        list[StepProgram],
        list[StepFeedback],
        dict[str, float],
    ]
):
    def __init__(
        self,
        dataset,
        model,
        name: str | None = None,
        with_background: bool = False,
        h5_path: str | None = None,
        max_concurrency: int = 4,
        timeout: float = 300.0,
    ):
        super().__init__(dataset=dataset, model=model, name=name)
        self._with_background = with_background
        self._h5_path = h5_path
        self._timeout = timeout
        self._code_eval_api = os.getenv(
            "SIEVAL_CODE_EVAL_API", "http://localhost:11451/evaluations"
        )
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_concurrency)
        )

    @override
    async def setup(self):
        if self._h5_path is None:
            self._h5_path = getattr(self.dataset, "h5_path", None)
        if not self._h5_path or not os.path.exists(self._h5_path):
            raise FileNotFoundError(
                "SciCode numeric test data not found. Run "
                "`sieval dataset download scicode` to stage raw_ground.h5, or pass "
                f"h5_path=. Resolved path: {self._h5_path!r}"
            )

    @override
    async def preprocess(self, raw, ctx):
        # Prompt assembly needs the model's own prior-step code, so it happens in
        # infer; preprocess passes the problem through unchanged.
        return raw

    @override
    async def infer(self, pre, ctx):
        sub_steps = pre["sub_steps"]
        problem_id = str(pre["problem_id"])
        deps = pre["required_dependencies"]
        tot = len(sub_steps)

        previous_llm_code: list[str | None] = [None] * tot
        steps_out: list[StepCode] = []
        outputs: list[ModelOutput] = []

        for i in range(tot):
            step = sub_steps[i]
            step_number = step["step_number"]

            if is_special_step(problem_id, i):
                # Scientist-authored gold code: context only, never generated/tested.
                # Embed the whole gold file verbatim — do NOT extract a single
                # function by name. These files are self-contained, import-free
                # blocks that may define a class (13.6 -> Maxwell) or several
                # top-level objects (62.1 -> Block AND EnlargedBlock); pulling one
                # node by func_name (which resolves to `__init__` for a class
                # header) would drop the class and break every later step that
                # references it.
                previous_llm_code[i] = special_step_code(step_number)
                steps_out.append(
                    {
                        "step_number": step_number,
                        "tested": False,
                        "code_content": None,
                        "raw_response": "",
                        "empty_extraction": False,
                    }
                )
                continue

            prompt, previous_code = generate_prompt_with_steps(
                sub_steps, deps, i + 1, previous_llm_code, self._with_background
            )
            messages: list[ChatCompletionUserMessageParam] = [
                {"role": "user", "content": prompt}
            ]
            output = await self.model.agenerate(messages, n=1)
            outputs.append(output)
            raw_response = output.texts[0]
            extracted = extract_python_script(raw_response)
            # An extraction with no def/class means the model returned prose, was
            # truncated, or emitted an unfenced answer; the step will fail and so
            # will any later step calling its function. Log it and flag for the
            # report instead of silently scoring 0.
            empty = "def " not in extracted and "class " not in extracted
            if empty:
                logger.warning(
                    "SciCode empty code extraction: problem {} step {} "
                    "(finish={}, raw_len={}); dependent steps may cascade-fail.",
                    problem_id,
                    step_number,
                    (output.finish_reasons or ["?"])[0],
                    len(raw_response),
                )
            previous_llm_code[i] = extracted
            # Matches upstream save_response_with_steps: `{previous_code}\n{code}`.
            steps_out.append(
                {
                    "step_number": step_number,
                    "tested": True,
                    "code_content": f"{previous_code}\n{extracted}",
                    "raw_response": raw_response,
                    "empty_extraction": empty,
                }
            )

        # Box the structured per-step code as the stage value while recording
        # token usage from every model call via the stage meta.
        return TaskStageOutput(value=steps_out, meta=build_stage_meta(*outputs))

    @override
    async def postprocess(self, inf, ctx):
        steps_out: list[StepCode] = inf.value
        sub_steps = ctx.raw_sample["sub_steps"]
        by_number = {s["step_number"]: s for s in sub_steps}

        programs: list[StepProgram] = []
        for sc in steps_out:
            if not sc["tested"] or sc["code_content"] is None:
                continue
            step_number = sc["step_number"]
            test_cases = by_number[step_number]["test_cases"]
            targets = process_hdf5_to_tuple(step_number, len(test_cases), self._h5_path)
            program = build_test_program(
                sc["code_content"], encode_targets(targets), test_cases
            )
            programs.append(
                {
                    "step_number": step_number,
                    "program": program,
                    "empty_extraction": sc["empty_extraction"],
                }
            )
        return programs

    @override
    async def feedback(self, post, ctx):
        feedbacks: list[StepFeedback] = []
        for step in post:
            try:
                resp = await self._http_client.post(
                    self._code_eval_api,
                    json={
                        "uuid": f"{step['step_number']}-{time.perf_counter_ns()}",
                        "source": "scicode",
                        "code": step["program"],
                        "timeout": self._timeout,
                    },
                    timeout=self._timeout + 5,  # buffer for network latency
                )
                resp.raise_for_status()
                res = resp.json()
                feedbacks.append(
                    {
                        "step_number": step["step_number"],
                        "correct": res["status"],
                        "msg": res["msg"],
                        "empty_extraction": step["empty_extraction"],
                    }
                )
            except Exception as e:
                logger.warning(
                    "SciCode eval error for step {}: [{}] {}",
                    step["step_number"],
                    type(e).__name__,
                    e,
                )
                raise e
        return True, feedbacks

    @override
    async def report(self, finals, fails):
        total_problems = len(finals) + len(fails)
        if total_problems == 0:
            return {"score": 0.0, "fails": len(fails)}

        correct_steps = 0
        total_steps = 0
        correct_problems = 0
        empty_extractions = 0
        for f in finals:
            feedbacks = f.feedback_result
            n_correct = sum(1 for fb in feedbacks if fb["correct"])
            correct_steps += n_correct
            total_steps += len(feedbacks)
            empty_extractions += sum(
                1 for fb in feedbacks if fb.get("empty_extraction")
            )
            if feedbacks and n_correct == len(feedbacks):
                correct_problems += 1

        # Pipeline failures (fails) count as unsolved problems; their step counts
        # are unknown, so sub-problem accuracy is over evaluated steps only.
        main_accuracy = correct_problems * 100 / total_problems
        sub_accuracy = correct_steps * 100 / total_steps if total_steps else 0.0
        return {
            "score": main_accuracy,
            "main_problem_accuracy": main_accuracy,
            "sub_problem_accuracy": sub_accuracy,
            "correct_problems": correct_problems,
            "total_problems": total_problems,
            "correct_steps": correct_steps,
            "total_steps": total_steps,
            # Steps where the model produced no extractable code (truncation /
            # unfenced / prose). A non-zero count means some failures are
            # generation-side, not solution-correctness — investigate raw_response.
            "empty_extractions": empty_extractions,
            "fails": len(fails),
        }

    @override
    async def shutdown(self):
        await self._http_client.aclose()
