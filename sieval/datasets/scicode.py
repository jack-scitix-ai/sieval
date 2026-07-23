"""SciCode dataset — scientific research coding benchmark (80 problems / 338 steps).

Problems come from the upstream GitHub jsonl (revision-pinned); the numeric
test-target file ``raw_ground.h5`` (byte-identical to the official
``test_data.h5``) is mirrored on HuggingFace as a plain HTTPS artifact, avoiding
the upstream Google-Drive dependency. The h5 path where the dataset was staged
is captured on :attr:`h5_path` for the task to read numeric targets from.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""

import os
from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset, ensure_dataset_dict

# github.com/scicode-bench/SciCode
SCICODE_GITHUB_REVISION = "69a8cfc829fe8788a426ce8b5de6292366dce7ef"
# huggingface.co/datasets/akshathmangudi/SciCode (raw/raw_ground.h5 == official
# test_data.h5; verified by matching sha256).
SCICODE_H5_MIRROR_REVISION = "526f6cbf273fc2b134444104fd1979a013e95d39"
H5_FILENAME = "raw_ground.h5"


class SciCodeSubStep(TypedDict):
    step_number: str
    step_description_prompt: str
    step_background: str
    ground_truth_code: str
    function_header: str
    return_line: str
    test_cases: list[str]


class SciCodeDatasetSample(TypedDict):
    problem_id: str
    problem_name: str
    required_dependencies: str
    sub_steps: list[SciCodeSubStep]


@sieval_dataset(
    name="scicode",
    display_name="SciCode",
    description="Research coding benchmark: 80 problems decomposed into 338 dependent sub-steps.",  # noqa: E501
    source=(
        f"url:https://raw.githubusercontent.com/scicode-bench/SciCode/{SCICODE_GITHUB_REVISION}/eval/data/problems_all.jsonl",
        f"url:https://raw.githubusercontent.com/scicode-bench/SciCode/{SCICODE_GITHUB_REVISION}/eval/data/problems_dev.jsonl",
        f"url:https://huggingface.co/datasets/akshathmangudi/SciCode/resolve/{SCICODE_H5_MIRROR_REVISION}/raw/{H5_FILENAME}",
    ),
    checksums={
        "problems_all.jsonl": "sha256:38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d",  # noqa: E501
        "problems_dev.jsonl": "sha256:193968aff23b7ed931c8f6d196b611e2f6694adb49e6cc085444a60ad2fc4f7b",  # noqa: E501
        "raw_ground.h5": "sha256:48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890",  # noqa: E501
    },
    categories=(Category(Level1Category.CODE, "CodeGeneration"),),
    tags=("english", "python", "code-exec"),
    license="apache-2.0",
)
class SciCodeDataset(Dataset[SciCodeDatasetSample]):
    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        # Capture the staged h5 location so the task can read numeric targets.
        # copy.copy-based clones (slice/shuffle/...) preserve this attribute.
        self._h5_path = os.path.join(name_or_path, H5_FILENAME)

        all_path = os.path.join(name_or_path, "problems_all.jsonl")
        dev_path = os.path.join(name_or_path, "problems_dev.jsonl")
        # Load each split independently: problems_dev carries extra optional
        # columns (general_solution, sub_steps[].ground_truth_code) absent from
        # problems_all, which trips HF's single-schema inference if the two are
        # generated together. Splits may keep distinct columns; the task reads
        # only the fields both share.
        test = ensure_dataset(
            load_dataset("json", data_files=all_path, split="train", **kwargs)
        )
        dev = ensure_dataset(
            load_dataset("json", data_files=dev_path, split="train", **kwargs)
        )
        return ensure_dataset_dict(HFDatasetDict({"test": test, "dev": dev}))

    @property
    def h5_path(self) -> str | None:
        """Filesystem path to the staged numeric test-target h5.

        ``None`` when the dataset was built from a pre-loaded dict (e.g. tests)
        rather than via :meth:`load`.
        """
        return getattr(self, "_h5_path", None)
