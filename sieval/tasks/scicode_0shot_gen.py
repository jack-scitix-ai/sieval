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

import asyncio
import os
import time
from typing import Literal, TypedDict, override

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam

from sieval.community.scicode import (
    build_test_program,
    encode_targets,
    extract_function_name,
    extract_python_script,
    generate_prompt_with_steps,
    get_function_from_code,
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
    status="experimental",
    reference_impl=ReferenceImpl(
        source="scicode",
        url="https://github.com/scicode-bench/SciCode/tree/69a8cfc829fe8788a426ce8b5de6292366dce7ef/eval/scripts",
        notes=(
            "Vendored from upstream eval/scripts into community/scicode: prompt "
            "templates, code/h5 parsers, comparison helpers, and the 3 "
            "non-generated gold steps (13.6/62.1/76.3, inlined in "
            "_gold_steps.py). special_step_mode selects gold injection and is "
            "SCORE-RELEVANT: 'verbatim' (default) injects the whole gold block; "
            "'extract' byte-matches upstream gencode_json.py, whose "
            "get_function_from_code matches the header's def before class and so "
            "silently drops the class wrapper for 13.6/62.1 (open upstream bugs "
            "#59/#49), leaving every dependent step to fail with NameError and "
            "making problems 13/62 structurally unsolvable. Use 'extract' when "
            "comparing against public leaderboard numbers (produced with that "
            "behavior). Calibration (Qwen2-72B, committed code, T=0 greedy, full "
            "test split, self-dependency): official expected values apply only "
            "to 'extract'. Observed extract main accuracy matches the official "
            "figures (1.5 without background, paper Table 2; 4.6 with background, "
            "Table 3). Observed main/sub pairs are extract no-background "
            "1.5/13.9, verbatim no-background 1.5/13.9, extract with-background "
            "4.6/24.0, and verbatim with-background 3.1/24.7; verbatim has no "
            "official expected value. Without background, the two modes have identical "
            "pass/fail outcomes because Qwen's affected downstream code is wrong "
            "even after the missing classes are restored. With background, "
            "verbatim rescues 13.14/62.2/62.4 (+3), as it does on a stronger "
            "model (gpt-5.6). An unrelated byte-identical program for stochastic "
            "step 14.2 flipped verdict between the two background runs: its "
            "unseeded Monte Carlo test creates an observed ±1-problem (about 1.5 "
            "point) main-accuracy jitter, while sglang generation differences in "
            "other non-special problems changed no verdicts in this calibration. "
            "Compatibility/parity adaptations and other deviations from upstream: "
            "(1) problems 2/28 import scipy.integrate.simps, removed in SciPy 1.14 "
            "(open upstream issue #2); a conditional legacy-compatible wrapper "
            "restores that API for only their 4 tested programs. Replaying the "
            "stored Qwen programs after this fix changes their failure causes but "
            "not their pass/fail outcomes, so the reported calibration is unchanged; "
            "(2) numeric h5 "
            "targets are read eval-side and inlined into the "
            "sandbox program (upstream reads test_data.h5 in-subprocess); "
            "(3) execution runs on a remote code-eval service over HTTP, not an "
            "in-process subprocess — the service caps sandbox memory (1 GB "
            "default) while upstream subprocesses are uncapped, so memory-heavy "
            "numeric steps could OOM here yet pass upstream; (4) pipeline "
            "failures count as unsolved problems in main-problem accuracy. "
            "Reproducing official numbers requires greedy decoding "
            "(temperature=0) in the model config; with_background defaults to "
            "False (the official headline mode). The code-eval service image "
            "must provide the sandbox scientific stack including sympy, which "
            "the vendored comparison shim injects — see the evaluator's "
            "requirements/scicode.txt."
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
        timeout: float = 1800.0,  # matches upstream test_generated_code.py
        special_step_mode: Literal["extract", "verbatim"] = "verbatim",
    ):
        if special_step_mode not in ("extract", "verbatim"):
            raise ValueError(
                "special_step_mode must be 'extract' or 'verbatim', "
                f"got {special_step_mode!r}"
            )
        super().__init__(dataset=dataset, model=model, name=name)
        self._with_background = with_background
        self._special_step_mode = special_step_mode
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
                gold = special_step_code(step_number)
                if self._special_step_mode == "extract":
                    # Upstream behavior (gencode_json.py): re-extract the node named
                    # by the header. For 13.6/62.1 the header's `def __init__` is
                    # matched before `class`, so this yields a bare method and drops
                    # the class wrapper — a known upstream bug (scicode-bench/SciCode
                    # #59, #49) that makes dependent steps fail with NameError. Kept
                    # as an opt-in for parity with the public leaderboard numbers.
                    gold = get_function_from_code(
                        gold, extract_function_name(step["function_header"])
                    )
                # else "verbatim" (default): inject the whole gold block (keeps the
                # class) — the fix proposed in #59/#49. Deviates from the public
                # (buggy) pipeline by design; see reference_impl.notes.
                previous_llm_code[i] = gold
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
            # Guard against an empty choices list (aborted/filtered response):
            # treat it like an empty extraction below instead of raising.
            raw_response = output.texts[0] if output.texts else ""
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

        pending: list[tuple[StepCode, str, list[str]]] = []
        for sc in steps_out:
            code = sc["code_content"]
            if not sc["tested"] or code is None:
                continue
            pending.append((sc, code, by_number[sc["step_number"]]["test_cases"]))

        def read_targets() -> dict[str, str]:
            # Numeric targets come from a large h5 file: blocking I/O that would
            # stall the event loop for every other in-flight sample. One worker-
            # thread hop per problem reads (and pickles) them all; h5py serializes
            # HDF5 access internally, so concurrent per-sample threads are safe.
            return {
                sc["step_number"]: encode_targets(
                    process_hdf5_to_tuple(sc["step_number"], len(cases), self._h5_path)
                )
                for sc, _code, cases in pending
            }

        targets_by_step = await asyncio.to_thread(read_targets)

        programs: list[StepProgram] = []
        for sc, code, cases in pending:
            step_number = sc["step_number"]
            programs.append(
                {
                    "step_number": step_number,
                    "program": build_test_program(
                        code, targets_by_step[step_number], cases
                    ),
                    "empty_extraction": sc["empty_extraction"],
                }
            )
        return programs

    @override
    async def feedback(self, post, ctx):
        # Steps are evaluated sequentially ON PURPOSE. Task-runner sample
        # concurrency determines how many problems are in flight; max_concurrency
        # only caps simultaneous HTTP connections to the code-eval service.
        # Fanning out per-step would multiply that load. Worst-case wall clock is
        # steps x timeout for a problem whose steps all run long — more likely
        # under "verbatim", where dependent steps genuinely execute (under
        # "extract" they die instantly on NameError).
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
                    # Must comfortably exceed the eval's OWN subprocess timeout
                    # (self._timeout) plus server-side profiling/serialization
                    # overhead. Too tight and a step that legitimately runs to
                    # the eval timeout trips a client ReadTimeout, failing the
                    # WHOLE problem instead of recording just that step as
                    # failed — the service already returns status=False (msg
                    # "subprocess timeout") on its own timeout.
                    # Do not time out while waiting for a pooled connection: the
                    # task runner may have more samples in flight than this
                    # client's max_connections limit. Connect/read/write retain
                    # the subprocess timeout plus transport-overhead buffer.
                    timeout=httpx.Timeout(self._timeout + 120, pool=None),
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
        timeouts = 0
        memory_errors = 0
        import_errors = 0
        for f in finals:
            feedbacks = f.feedback_result
            n_correct = sum(1 for fb in feedbacks if fb["correct"])
            correct_steps += n_correct
            total_steps += len(feedbacks)
            empty_extractions += sum(
                1 for fb in feedbacks if fb.get("empty_extraction")
            )
            messages = [str(fb.get("msg", "")).lower() for fb in feedbacks]
            timeouts += sum("timeout" in msg for msg in messages)
            memory_errors += sum("memoryerror" in msg for msg in messages)
            import_errors += sum("importerror" in msg for msg in messages)
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
            # Step-level execution failures remain incorrect answers, but surface
            # their causes so report.json does not hide systemic environment or
            # resource failures behind an otherwise healthy pipeline fails=0.
            "timeouts": timeouts,
            "memory_errors": memory_errors,
            "import_errors": import_errors,
            "fails": len(fails),
        }

    @override
    async def shutdown(self):
        await self._http_client.aclose()
