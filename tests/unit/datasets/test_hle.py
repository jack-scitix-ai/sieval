"""Unit tests for the HLE dataset wrapper.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

from unittest.mock import patch

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import Features, Image, Value

import sieval.datasets.hle as hle_module
from sieval.core.datasets.meta import get_dataset_meta
from sieval.datasets.hle import HLE_REVISION, HLEDataset


def _row(image: str = "") -> dict:
    return {
        "id": "q1",
        "question": "What is 2 + 2?",
        "image": image,
        "answer": "4",
        "answer_type": "exactMatch",
        "author_name": "author",
        "rationale": "",
        "raw_subject": "Math",
        "category": "Math",
    }


def _hf_dict(rows: int = 1) -> HFDatasetDict:
    return HFDatasetDict({"test": HFDataset.from_list([_row()] * rows)})


def test_source_pins_hf_revision():
    meta_source = get_dataset_meta(HLEDataset).source
    assert meta_source == (f"hf:cais/hle@{HLE_REVISION}",)


def test_load_forwards_path_and_preserves_image_column():
    hf_dict = _hf_dict()
    dataset = HLEDataset(_hf_dict=hf_dict)
    with patch.object(hle_module, "load_dataset", return_value=hf_dict) as mock_load:
        loaded = dataset.load("cais/hle")

    # Minimal loader: pass name_or_path straight through, keep the "test" split.
    assert mock_load.call_args.args[0] == "cais/hle"
    assert "test" in loaded
    # The multimodal `image` column must be preserved (not welded to text-only).
    assert "image" in loaded["test"].column_names


def test_load_disables_auxiliary_image_decoding():
    # image_preview / rationale_image are HF Image features (decode=True upstream);
    # load() must disable decoding so a row fetch never requires Pillow.
    features = Features(
        {
            "id": Value("string"),
            "question": Value("string"),
            "image": Value("string"),
            "image_preview": Image(decode=True),
            "answer": Value("string"),
            "answer_type": Value("string"),
            "author_name": Value("string"),
            "rationale": Value("string"),
            "rationale_image": Image(decode=True),
            "raw_subject": Value("string"),
            "category": Value("string"),
        }
    )
    row = {**_row(), "image_preview": None, "rationale_image": None}
    hf = HFDatasetDict(
        {
            "test": HFDataset.from_dict(
                {k: [row[k]] for k in features}, features=features
            )
        }
    )
    dataset = HLEDataset(_hf_dict=hf)
    with patch.object(hle_module, "load_dataset", return_value=hf):
        loaded = dataset.load("cais/hle")

    assert loaded["test"].features["image_preview"].decode is False
    assert loaded["test"].features["rationale_image"].decode is False
