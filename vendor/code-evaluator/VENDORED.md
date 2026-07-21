# Vendored: code-evaluator

Self-contained, in-tree copy of the multi-language code-execution service that
sieval talks to over HTTP (`SIEVAL_CODE_EVAL_API`, default
`http://localhost:11451/evaluations`). Previously referenced as a git submodule
at `submodules/code-evaluator`; converted to a vendored copy so a plain
`git clone` of sieval is self-contained (no `git submodule update` required).

- **Upstream**: https://github.com/scitix/code-evaluator
- **Vendored at commit**: `e4802268f2b491c7ea3d7ed7704dd8582bc079be`

## Local modifications on top of the vendored commit

- **float-tol** (`app/exec_py_test.py`): clearer output-comparison messages +
  opt-in float tolerance via `CODE_EVAL_FLOAT_TOL` (default off = exact `==`,
  matching official LiveCodeBench). Ported from the upstream fork branch
  `fix/checker-messages-float-tol` (`cfc47d8`).

## Deployment note

`requirements/scicode.txt` pins `scipy==1.16.3`, which requires **Python ≥ 3.11**
(and `numpy==1.26.4` requires **< 3.13**), so the SciCode execution environment
must run on Python 3.11 or 3.12 — unlike the other Dockerfiles here, which use
`python:3.10-slim`.
