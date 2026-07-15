"""Humanity's Last Exam (HLE) dataset loader (Center for AI Safety).

HLE (Phan et al., 2025) is a multi-domain, closed-ended academic benchmark of
frontier-difficulty questions (mathematics, sciences, humanities), each with an
``exactMatch`` or ``multipleChoice`` gold answer suitable for automated
LLM-judge grading. The Hub repo exposes a single ``test`` split; this loader
mirrors it as-is.

The model-facing image is the ``image`` column — a plain string (a base64 data
URI, ``""`` when absent), preserved untouched. The repo also ships two auxiliary
``Image`` feature columns (``image_preview``, ``rationale_image``) that default
to ``decode=True``; no task consumes them, but materializing a row would decode
them and pull in Pillow. Decoding is disabled here (``Image(decode=False)``) so
the rows stay Pillow-free while keeping the raw bytes available; text-only vs.
full-set selection remains the task's concern.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from typing import TypedDict, override

from datasets import DatasetDict as HFDatasetDict
from datasets import Image, load_dataset

from sieval.core.datasets import (
    Category,
    Dataset,
    Level1Category,
    sieval_dataset,
)
from sieval.core.utils.hf import ensure_dataset_dict

# Pin the Hub revision for reproducibility (`main` at integration time).
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"


class HLEDatasetSample(TypedDict):
    id: str
    question: str
    image: str
    answer: str
    answer_type: str
    author_name: str
    rationale: str
    raw_subject: str
    category: str


@sieval_dataset(
    name="hle",
    display_name="Humanity's Last Exam",
    description="HLE — multi-domain, closed-ended frontier academic benchmark.",
    source=f"hf:cais/hle@{HLE_REVISION}",
    categories=(Category(Level1Category.KNOWLEDGE, "Multi-domain"),),
    tags=("english", "reasoning", "academic"),
    license="MIT",
)
class HLEDataset(Dataset[HLEDatasetSample]):
    # Auxiliary Image feature columns no task consumes; decoding them on row
    # access would require Pillow, so disable it (raw bytes are kept).
    _IMAGE_FEATURE_COLUMNS = ("image_preview", "rationale_image")

    @override
    def load(self, name_or_path: str, **kwargs) -> HFDatasetDict:
        dataset = ensure_dataset_dict(load_dataset(name_or_path, **kwargs))
        for split in dataset:
            for column in self._IMAGE_FEATURE_COLUMNS:
                dataset[split] = dataset[split].cast_column(column, Image(decode=False))
        return dataset
