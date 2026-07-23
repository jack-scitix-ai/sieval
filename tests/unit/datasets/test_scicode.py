"""Unit tests for the SciCode dataset loader.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import json

from sieval.datasets.scicode import SciCodeDataset


def _problem(problem_id: str) -> dict:
    return {
        "problem_id": problem_id,
        "problem_name": f"problem {problem_id}",
        "required_dependencies": "import numpy as np",
        "sub_steps": [
            {
                "step_number": f"{problem_id}.1",
                "step_description_prompt": "do step 1",
                "step_background": "bg 1",
                "ground_truth_code": "def f():\n    return 1",
                "function_header": "def f():",
                "return_line": "    return 1",
                "test_cases": ["assert f() == 1"],
            }
        ],
    }


def _dev_problem(problem_id: str) -> dict:
    # The real problems_dev.jsonl carries extra columns absent from
    # problems_all.jsonl (general_solution at top level, ground_truth_code per
    # sub-step); the loader must tolerate the per-split schema difference.
    p = _problem(problem_id)
    p["general_solution"] = "def solve():\n    return 1"
    p["sub_steps"][0]["ground_truth_code"] = "def f():\n    return 1"
    return p


def _stage(tmp_path) -> str:
    root = tmp_path / "scicode"
    root.mkdir()
    (root / "problems_all.jsonl").write_text(
        "\n".join(json.dumps(_problem(str(i))) for i in (1, 2, 3)), encoding="utf-8"
    )
    (root / "problems_dev.jsonl").write_text(
        json.dumps(_dev_problem("1")), encoding="utf-8"
    )
    # load() never reads the h5; a placeholder is enough to resolve the path.
    (root / "raw_ground.h5").write_bytes(b"")
    return str(root)


def test_load_builds_test_and_dev_splits(tmp_path):
    ds = SciCodeDataset(_stage(tmp_path))
    assert set(ds.dataset_dict) == {"test", "dev"}
    assert ds.test_set is not None
    assert len(ds.test_set) == 3
    assert len(ds.dataset_dict["dev"]) == 1

    row = ds.test_set[0]
    assert row["problem_id"] == "1"
    assert row["required_dependencies"] == "import numpy as np"
    assert row["sub_steps"][0]["step_number"] == "1.1"
    assert row["sub_steps"][0]["test_cases"] == ["assert f() == 1"]


def test_h5_path_points_at_staged_file(tmp_path):
    root = _stage(tmp_path)
    ds = SciCodeDataset(root)
    assert ds.h5_path == f"{root}/raw_ground.h5"


def test_h5_path_survives_clone(tmp_path):
    # slice() returns a copy.copy clone; the captured h5 path must persist so the
    # task can still locate the numeric targets after sampling.
    ds = SciCodeDataset(_stage(tmp_path))
    sliced = ds.slice(2)
    assert sliced.test_set is not None
    assert len(sliced.test_set) == 2
    assert sliced.h5_path == ds.h5_path


def test_h5_path_none_when_built_from_dict(tmp_path):
    ds = SciCodeDataset(_stage(tmp_path))
    rebuilt = type(ds)(_hf_dict=ds.dataset_dict)
    assert rebuilt.h5_path is None
