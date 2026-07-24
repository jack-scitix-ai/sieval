# Copyright 2024 The SciCode authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored SciCode evaluation assets (github.com/scicode-bench/SciCode @ 69a8cfc).

Local adaptation of the upstream prompt templates, code/HDF5 parsers, comparison
helpers, and the three scientist-authored gold steps, plus a sandbox-program
builder that inlines h5 targets so sieval's stateless code-eval service can run
the tests.
"""

from sieval.community.scicode.harness import (
    build_test_program,
    encode_targets,
)
from sieval.community.scicode.parse import (
    extract_python_script,
    process_hdf5_to_tuple,
)
from sieval.community.scicode.prompts import (
    generate_prompt_with_steps,
    is_special_step,
    special_step_code,
)

__all__ = [
    "build_test_program",
    "encode_targets",
    "extract_python_script",
    "process_hdf5_to_tuple",
    "generate_prompt_with_steps",
    "is_special_step",
    "special_step_code",
]
