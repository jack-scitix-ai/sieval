"""Assemble a single self-contained test program for one SciCode sub-step.

Upstream (eval/scripts/test_generated_code.py) runs each step inside an
environment where ``scicode`` is installed and ``test_data.h5`` is on disk, then
appends ``targets = process_hdf5_to_tuple(step_id, n)`` followed by each test
case. sieval's code-eval sandbox is stateless, so instead we:

  * read the numeric targets from the h5 on the eval side and inline them as a
    pickled+zlib+base64 blob (mirrors how LiveCodeBench inlines its private
    tests), and
  * register an in-process ``scicode.compare.cmp`` module so the test bodies'
    ``from scicode.compare.cmp import cmp_tuple_or_list`` imports resolve.

The concatenated function code (dependencies + prior steps + current step) is
supplied verbatim by the caller. The sandbox must provide the scientific stack
the problems import (numpy / scipy / sympy / ...); h5py is NOT needed there.

AI-Generated Code - Claude Opus 4.8 (1M context) (Anthropic)
"""
import base64
import functools
import pickle
import zlib
from importlib import resources


@functools.lru_cache(maxsize=1)
def _cmp_source() -> str:
    return (
        resources.files("sieval.community.scicode")
        .joinpath("_cmp_upstream.py")
        .read_text(encoding="utf-8")
    )


# Registers the vendored comparison helpers under the module path the upstream
# test bodies import from, without requiring `scicode` to be installed.
_SHIM_FOOTER = """
import sys as _sys, types as _types

_scicode_cmp = _types.ModuleType("scicode.compare.cmp")
for _name in (
    "cmp_tuple_or_list",
    "are_dicts_close",
    "are_csc_matrix_close",
    "process_symbol_in_dict",
):
    setattr(_scicode_cmp, _name, globals()[_name])
_scicode = _types.ModuleType("scicode")
_scicode_compare = _types.ModuleType("scicode.compare")
_scicode_compare.cmp = _scicode_cmp
_scicode.compare = _scicode_compare
_sys.modules.setdefault("scicode", _scicode)
_sys.modules.setdefault("scicode.compare", _scicode_compare)
_sys.modules["scicode.compare.cmp"] = _scicode_cmp
"""


def encode_targets(targets: list) -> str:
    """Serialize h5 targets to a base64 string safe to embed in source."""
    return base64.b64encode(zlib.compress(pickle.dumps(targets))).decode("ascii")


def build_test_program(code_content: str, targets_b64: str, test_cases: list) -> str:
    """Return a runnable program: solution code + shim + targets + test cases.

    *code_content* is ``dependencies + prior-step funcs + current-step func``.
    *targets_b64* is :func:`encode_targets` output. *test_cases* are the raw
    upstream test-body strings; each references ``target`` (the i-th target).
    """
    parts = [
        code_content,
        "",
        "# --- sieval scicode test harness (injected) ---",
        "import base64 as _b64, zlib as _zlib, pickle as _pkl",
        _cmp_source(),
        _SHIM_FOOTER,
        f'targets = _pkl.loads(_zlib.decompress(_b64.b64decode("{targets_b64}")))',
        "",
    ]
    for idx, case in enumerate(test_cases):
        parts.append(f"target = targets[{idx}]\n")
        parts.append(case)
        parts.append("")
    return "\n".join(parts)
