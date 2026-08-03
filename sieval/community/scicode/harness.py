"""Assemble a single self-contained test program for one SciCode sub-step.

Upstream (eval/scripts/test_generated_code.py) runs each step inside an
environment where ``scicode`` is installed and ``test_data.h5`` is on disk, then
appends ``targets = process_hdf5_to_tuple(step_id, n)`` followed by each test
case. sieval's code-eval sandbox is stateless, so instead we:

  * read the numeric targets from the h5 on the eval side and inline them as a
    pickled+zlib+base64 blob (mirrors how LiveCodeBench inlines its private
    tests), and
  * register an in-process ``scicode.compare.cmp`` module so the test bodies'
    ``from scicode.compare.cmp import cmp_tuple_or_list`` imports resolve, and
  * restore the removed ``scipy.integrate.simps`` API only for the four programs
    whose upstream dependencies import it (SciCode issue #2).

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


# Problems 2/28 import scipy.integrate.simps, which SciPy removed in 1.14. The
# official SciCode results predate that removal. Install a legacy-compatible
# wrapper before those four programs execute, without downgrading SciPy for the
# other 284 tested steps. Defining it inside the installer keeps its numpy/scipy
# dependencies in a closure rather than leaking them into solution globals.
_SCIPY_SIMPS_IMPORT = "from scipy.integrate import simps"
_SCIPY_SIMPS_COMPAT = r'''
def _install_scicode_simps_compat():
    import numpy as _np
    import scipy.integrate as _integrate

    if hasattr(_integrate, "simps"):
        return

    def _slice_axis(array, axis, start, stop):
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(start, stop)
        return array[tuple(slices)]

    def simps(y, x=None, dx=1.0, axis=-1, even="avg"):
        """Compatibility implementation of the pre-SciPy-1.14 simps API."""
        y = _np.asarray(y)
        if y.ndim == 0:
            return _integrate.simpson(y, x=x, dx=dx, axis=axis)
        if axis < -y.ndim or axis >= y.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array dimension {y.ndim}")
        axis %= y.ndim
        sample_count = y.shape[axis]

        if even not in ("avg", "first", "last"):
            raise ValueError("Parameter 'even' must be 'avg', 'last', or 'first'.")
        if sample_count % 2 == 1 or sample_count <= 2:
            return _integrate.simpson(y, x=x, dx=dx, axis=axis)

        x_array = None if x is None else _np.asarray(x)
        if x_array is not None:
            if x_array.ndim == 1:
                x_axis = 0
            elif x_array.ndim == y.ndim:
                x_axis = axis
            else:
                raise ValueError("If given, shape of x must be 1-D or the same as y.")
            if x_array.shape[x_axis] != sample_count:
                raise ValueError(
                    "If given, length of x along axis must be the same as y."
                )
            x_first = _slice_axis(x_array, x_axis, None, -1)
            x_last = _slice_axis(x_array, x_axis, 1, None)
            first_dx = _np.take(x_array, 1, axis=x_axis) - _np.take(
                x_array, 0, axis=x_axis
            )
            last_dx = _np.take(x_array, -1, axis=x_axis) - _np.take(
                x_array, -2, axis=x_axis
            )
        else:
            x_first = x_last = None
            first_dx = last_dx = dx

        y_first = _slice_axis(y, axis, None, -1)
        y_last = _slice_axis(y, axis, 1, None)
        first = _integrate.simpson(y_first, x=x_first, dx=dx, axis=axis)
        first += 0.5 * last_dx * (
            _np.take(y, -2, axis=axis) + _np.take(y, -1, axis=axis)
        )
        last = 0.5 * first_dx * (
            _np.take(y, 0, axis=axis) + _np.take(y, 1, axis=axis)
        )
        last += _integrate.simpson(y_last, x=x_last, dx=dx, axis=axis)

        if even == "first":
            return first
        if even == "last":
            return last
        return 0.5 * (first + last)

    _integrate.simps = simps

_install_scicode_simps_compat()
del _install_scicode_simps_compat
'''


# Execute the vendored comparison module in its own namespace. Inlining the
# source at program top level used to leak np/scipy/sympy into solution globals,
# allowing an undeclared import to pass here but fail in the upstream harness.
@functools.lru_cache(maxsize=1)
def _cmp_shim() -> str:
    source = _cmp_source()
    return f"""
import sys as _sys, types as _types

_scicode_cmp = _types.ModuleType("scicode.compare.cmp")
exec(compile({source!r}, "<sieval scicode.compare.cmp>", "exec"), _scicode_cmp.__dict__)
_scicode = _types.ModuleType("scicode")
_scicode_compare = _types.ModuleType("scicode.compare")
_scicode_compare.cmp = _scicode_cmp
_scicode.compare = _scicode_compare
_sys.modules["scicode"] = _scicode
_sys.modules["scicode.compare"] = _scicode_compare
_sys.modules["scicode.compare.cmp"] = _scicode_cmp
del _scicode, _scicode_compare, _scicode_cmp, _sys, _types
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
    parts = []
    if _SCIPY_SIMPS_IMPORT in code_content:
        parts.extend([_SCIPY_SIMPS_COMPAT, ""])
    parts.extend(
        [
            code_content,
            "",
            "# --- sieval scicode test harness (injected) ---",
            "import base64 as _b64, zlib as _zlib, pickle as _pkl",
            _cmp_shim(),
            f'targets = _pkl.loads(_zlib.decompress(_b64.b64decode("{targets_b64}")))',
            "",
        ]
    )
    for idx, case in enumerate(test_cases):
        parts.append(f"target = targets[{idx}]\n")
        parts.append(case)
        parts.append("")
    return "\n".join(parts)
