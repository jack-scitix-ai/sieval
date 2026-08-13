"""Task-test collection tweaks for mutation testing.

The ``test_import_does_not_pull_<dep>`` convention spawns a FRESH python
interpreter (``subprocess.run([sys.executable, "-c", "import sieval..."])``)
to prove an optional dependency is lazy-imported. Under mutation testing the
subprocess inherits the environment pointing at mutmut's instrumented
``mutants/`` tree, whose trampolines require mutmut's in-process runtime
config — unavailable in a bare interpreter — so the import crashes for
reasons unrelated to what the test verifies. These tests also exercise no
``sieval/core`` mutants, so skipping them under mutation testing loses no
kill power. ``MUTANT_UNDER_TEST`` is set by mutmut for every pytest run it
drives and is absent otherwise.

AI-Generated Code - Claude Fable 5 (Anthropic)
"""

import os

import pytest


def pytest_collection_modifyitems(items):
    if not os.environ.get("MUTANT_UNDER_TEST"):
        return
    skip = pytest.mark.skip(
        reason="fresh-interpreter lazy-import check cannot run against "
        "mutmut's instrumented tree"
    )
    for item in items:
        if item.name.startswith("test_import_does_not_pull_"):
            item.add_marker(skip)
