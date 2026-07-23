"""Unit tests for the SciCode 0-shot generative task.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict

from sieval.core.models import ModelOutput
from sieval.core.models.chat_model import ChatModel
from sieval.core.tasks import TaskContext, TaskStageOutput
from sieval.datasets.scicode import SciCodeDataset
from sieval.tasks.scicode_0shot_gen import SciCodeZeroShotGenTask

# h5py is an optional (scicode-group) dependency; skip the whole module without it.
h5py = pytest.importorskip("h5py")


class _ScriptedChatModel(ChatModel):
    """Returns queued replies in order, recording each prompt and kwargs."""

    def __init__(self, replies: list[str], model: str = "candidate"):
        super().__init__(model=model, api_key="fake")
        self._replies = list(replies)
        self.prompts: list = []
        self.last_kwargs: dict[str, object] = {}
        self.calls = 0

    async def _agenerate_impl(self, prompt, **kwargs) -> ModelOutput:
        self.prompts.append(prompt)
        self.last_kwargs = dict(kwargs)
        self.calls += 1
        return ModelOutput(model=self.meta(), texts=[self._replies.pop(0)])

    async def _alogprobs_impl(self, prompt, **kwargs) -> ModelOutput:
        _ = (prompt, kwargs)
        return ModelOutput(model=self.meta(), texts=[""])


def _code_reply(fn: str) -> str:
    return f"```python\ndef {fn}():\n    return 1\n```"


def _dataset() -> SciCodeDataset:
    row = {
        "problem_id": "1",
        "problem_name": "p",
        "required_dependencies": "import numpy as np",
        "sub_steps": [],
    }
    return SciCodeDataset(_hf_dict=HFDatasetDict({"test": HFDataset.from_list([row])}))


def _substep(number: str, header: str, tests: list[str]) -> dict:
    return {
        "step_number": number,
        "step_description_prompt": f"describe {number}",
        "step_background": f"background {number}",
        "ground_truth_code": "",
        "function_header": header,
        "return_line": "    return 1",
        "test_cases": tests,
    }


def _task(model, **kw) -> SciCodeZeroShotGenTask:
    kw.setdefault("h5_path", "unused")
    return SciCodeZeroShotGenTask(_dataset(), model, **kw)


# --- infer: sequential, single-n, prior code fed forward ---


@pytest.mark.anyio
async def test_infer_is_sequential_and_feeds_prior_code():
    sub_steps = [
        _substep("1.1", "def step_a():", ["assert step_a() == 1"]),
        _substep("1.2", "def step_b():", ["assert step_b() == 1"]),
    ]
    model = _ScriptedChatModel([_code_reply("step_a"), _code_reply("step_b")])
    task = _task(model)
    raw = {
        "problem_id": "1",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))

    assert isinstance(boxed, TaskStageOutput)
    assert model.calls == 2
    # Only the scheduling knob is forwarded to the model layer.
    assert model.last_kwargs == {"n": 1}
    # The second prompt must embed the first step's generated function.
    assert "def step_a" in model.prompts[1][0]["content"]

    steps = boxed.value
    assert [s["tested"] for s in steps] == [True, True]
    # code_content = dependencies + accumulated prior funcs + current func.
    assert steps[1]["code_content"].startswith("import numpy as np")
    assert "def step_a" in steps[1]["code_content"]
    assert "def step_b" in steps[1]["code_content"]


@pytest.mark.anyio
async def test_infer_uses_gold_context_for_special_step_without_calling_model():
    # Problem 76 step 3 (index 2) is scientist-authored: never generated/tested,
    # its gold code is fed as context to later steps.
    sub_steps = [
        _substep("76.1", "def step_a():", ["assert step_a() == 1"]),
        _substep("76.2", "def step_b():", ["assert step_b() == 1"]),
        _substep("76.3", "def generate_dna(N, PWM):", ["assert True"]),
        _substep("76.4", "def step_d():", ["assert step_d() == 1"]),
    ]
    model = _ScriptedChatModel(
        [_code_reply("step_a"), _code_reply("step_b"), _code_reply("step_d")]
    )
    task = _task(model)
    raw = {
        "problem_id": "76",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    steps = boxed.value

    # 3 generated steps -> 3 model calls; the special step is skipped.
    assert model.calls == 3
    assert steps[2]["tested"] is False
    assert steps[2]["code_content"] is None
    # The 4th step's prompt must contain the gold generate_dna function.
    assert "def generate_dna" in model.prompts[2][0]["content"]


@pytest.mark.anyio
async def test_infer_embeds_full_gold_class_for_class_special_step():
    # Problem 62 step 1 (index 0) is a gold step whose file defines TWO classes
    # (Block AND EnlargedBlock) under a `class ...:` header. The gold must be
    # embedded verbatim: extracting a single node by the header's function name
    # (which resolves to `__init__` for a class header) would drop the `class`
    # wrapper and the second class, breaking every later step that uses them.
    sub_steps = [
        _substep(
            "62.1", "class EnlargedBlock:\n    def __init__(self):", ["assert True"]
        ),
        _substep("62.2", "def uses_block():", ["assert uses_block() == 1"]),
    ]
    model = _ScriptedChatModel([_code_reply("uses_block")])
    task = _task(model)
    raw = {
        "problem_id": "62",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))

    # Only the one non-special step calls the model.
    assert model.calls == 1
    # The gold context injected into step 62.2's prompt must contain BOTH classes
    # with their `class` keyword intact (not just an unwrapped __init__).
    gold_ctx = model.prompts[0][0]["content"]
    assert "class Block" in gold_ctx
    assert "class EnlargedBlock" in gold_ctx


@pytest.mark.anyio
async def test_infer_flags_empty_extraction_when_model_returns_no_code():
    # A prose-only / truncated response yields no def/class: the step must be
    # flagged so the report can distinguish generation failure from wrong answers.
    sub_steps = [_substep("5.1", "def only_step():", ["assert only_step() == 1"])]
    model = _ScriptedChatModel(["I cannot help with that."])  # no code fence
    task = _task(model)
    raw = {
        "problem_id": "5",
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }

    boxed = await task.infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    step = boxed.value[0]
    assert step["empty_extraction"] is True
    assert step["raw_response"] == "I cannot help with that."
    # A normal fenced reply must NOT be flagged (discriminates the check).
    model2 = _ScriptedChatModel([_code_reply("only_step")])
    boxed2 = await _task(model2).infer(raw, TaskContext(sample_id=0, raw_sample=raw))
    assert boxed2.value[0]["empty_extraction"] is False


# --- postprocess: h5 targets inlined into a self-contained, runnable program ---


def _write_h5(path, step_number: str, value):
    with h5py.File(path, "w") as f:
        f.create_dataset(f"{step_number}/test1/var1", data=value)


@pytest.mark.anyio
async def test_postprocess_builds_program_that_executes(tmp_path):
    # The injected comparison shim (vendored cmp) imports sympy; it runs in the
    # code-eval sandbox at runtime, not on the eval side, so sympy is not a
    # declared eval-side dep. Exercise the full program only where it's present.
    pytest.importorskip("sympy")
    h5 = tmp_path / "raw_ground.h5"
    _write_h5(h5, "9.1", 42)
    model = _ScriptedChatModel([])
    task = _task(model, h5_path=str(h5))

    sub_step = _substep("9.1", "def f():", ["assert np.allclose(f(), target)"])
    raw = {"problem_id": "9", "sub_steps": [sub_step]}
    inf = TaskStageOutput(
        value=[
            {
                "step_number": "9.1",
                "tested": True,
                "code_content": "import numpy as np\n\ndef f():\n    return 42\n",
                "raw_response": "```python\ndef f():\n    return 42\n```",
                "empty_extraction": False,
            }
        ]
    )

    programs = await task.postprocess(inf, TaskContext(sample_id=0, raw_sample=raw))
    assert len(programs) == 1
    program = programs[0]["program"]

    # The vendored comparison import and the correct target must both resolve at
    # runtime; a matching solution passes without raising.
    exec(compile(program, "<program>", "exec"), {})

    # A wrong solution must fail the injected assertion.
    bad = program.replace("return 42", "return 0")
    with pytest.raises(AssertionError):
        exec(compile(bad, "<program>", "exec"), {})


# --- report: sub-problem and main-problem accuracy ---


def _final(sample_id, correct_flags, empty_flags=None):
    empty_flags = empty_flags or [False] * len(correct_flags)
    return TaskContext(
        sample_id=sample_id,
        feedback_result=[
            {"step_number": f"s{i}", "correct": c, "msg": "", "empty_extraction": e}
            for i, (c, e) in enumerate(zip(correct_flags, empty_flags, strict=True))
        ],
    )


@pytest.mark.anyio
async def test_report_sub_and_main_accuracy():
    model = _ScriptedChatModel([])
    task = _task(model)
    finals = [
        _final(0, [True, True]),  # fully solved
        _final(1, [True, False, True]),  # partial
    ]
    report = await task.report(finals, fails=[])

    # steps: 4 correct / 5 total = 80%; problems: 1 fully solved / 2 = 50%.
    assert report["correct_steps"] == 4
    assert report["total_steps"] == 5
    assert report["sub_problem_accuracy"] == pytest.approx(80.0)
    assert report["correct_problems"] == 1
    assert report["total_problems"] == 2
    assert report["main_problem_accuracy"] == pytest.approx(50.0)
    assert report["score"] == report["main_problem_accuracy"]


@pytest.mark.anyio
async def test_report_counts_empty_extractions():
    model = _ScriptedChatModel([])
    task = _task(model)
    # Sample 1 has one step whose code failed to extract (also counts as wrong).
    finals = [
        _final(0, [True, True]),
        _final(1, [False, True], empty_flags=[True, False]),
    ]
    report = await task.report(finals, fails=[])
    assert report["empty_extractions"] == 1
    # It still counts as an incorrect step, not a separate bucket.
    assert report["correct_steps"] == 3
    assert report["total_steps"] == 4


@pytest.mark.anyio
async def test_report_counts_pipeline_fails_as_unsolved_problems():
    model = _ScriptedChatModel([])
    task = _task(model)
    finals = [_final(0, [True, True])]
    fails = [TaskContext(sample_id=1)]
    report = await task.report(finals, fails)

    # The failed problem dilutes main-problem accuracy: 1 solved / 2 = 50%.
    assert report["fails"] == 1
    assert report["total_problems"] == 2
    assert report["main_problem_accuracy"] == pytest.approx(50.0)
    # Sub accuracy is over evaluated steps only (fail step-count unknown).
    assert report["total_steps"] == 2


@pytest.mark.anyio
async def test_report_empty_is_zero():
    model = _ScriptedChatModel([])
    task = _task(model)
    report = await task.report([], [])
    assert report["score"] == 0.0
