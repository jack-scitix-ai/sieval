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

# Prompt templates and step-assembly logic adapted from SciCode @ 69a8cfc:
#   - templates: eval/data/{multistep_template.txt,background_comment_template.txt}
#   - assembly:  eval/scripts/gencode_json.py (Gencode.process_problem_steps /
#     process_problem_code / generate_prompt_with_steps)
# The upstream Gencode class is decoupled here from file IO and model calls:
# these pure functions take the already-generated prior-step code as an argument
# so the sieval task can drive the sequential loop in-memory. The upstream
# template naming is preserved: "with background" selects the multistep template
# (scientist background injected); "without background" selects the
# background-comment template (the model must produce its own background).

from importlib import resources

# eval/data/multistep_template.txt (used when with_background=True)
MULTISTEP_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```.
"""

# eval/data/background_comment_template.txt (used when with_background=False)
BACKGROUND_COMMENT_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""

# Steps that upstream does NOT generate; their scientist-authored gold code
# (eval/data/{prob}.{step}.txt) is used as context for later steps and the
# steps themselves are never tested. Keyed by (problem_id, zero-based step idx).
SPECIAL_STEPS = {("13", 5), ("62", 0), ("76", 2)}


def is_special_step(problem_id: str, step_idx: int) -> bool:
    """True if (problem_id, zero-based step_idx) is a non-generated gold step."""
    return (str(problem_id), step_idx) in SPECIAL_STEPS


def special_step_code(step_number: str) -> str:
    """Return the scientist-authored gold code for a special step.

    *step_number* is the ``sub_steps[i]["step_number"]`` value (e.g. ``"13.6"``),
    which also names the vendored ``data/{step_number}.txt`` file.
    """
    return (
        resources.files("sieval.community.scicode")
        .joinpath("data", f"{step_number}.txt")
        .read_text(encoding="utf-8")
    )


def prompt_template(with_background: bool) -> str:
    return MULTISTEP_TEMPLATE if with_background else BACKGROUND_COMMENT_TEMPLATE


def _process_problem_code(sub_steps: list, num_steps: int) -> str:
    header_docstring = sub_steps[num_steps - 1]["function_header"]
    return_str = sub_steps[num_steps - 1]["return_line"]
    return f"{header_docstring}\n\n{return_str}"


def process_problem_steps(
    sub_steps: list,
    num_steps: int,
    previous_llm_code: list,
    with_background: bool,
):
    """Assemble prior-steps text, the next-step block, and the prior-code string.

    *previous_llm_code* holds the generated (import-stripped) code for each prior
    step, indexed by zero-based step. Mirrors Gencode.process_problem_steps,
    including the upstream operator precedence of the with_background ternary.
    """
    output_lines = []
    previous_code = []
    for i in range(num_steps - 1):
        output_lines.append(
            sub_steps[i]["step_description_prompt"] + "\n" + sub_steps[i]["step_background"]
            if with_background
            else sub_steps[i]["step_description_prompt"]
        )
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    next_step = [
        sub_steps[num_steps - 1]["step_description_prompt"]
        + "\n"
        + sub_steps[num_steps - 1]["step_background"]
        if with_background
        else sub_steps[num_steps - 1]["step_description_prompt"],
        _process_problem_code(sub_steps, num_steps),
    ]
    output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return output_str, next_step_str, previous_code_str


def generate_prompt_with_steps(
    sub_steps: list,
    required_dependencies: str,
    num_steps: int,
    previous_llm_code: list,
    with_background: bool,
):
    """Return ``(prompt, previous_code)`` for the *num_steps*-th step (1-based).

    ``previous_code`` is ``f"{dependencies}\\n{previous_code_str}\\n"`` — the
    import block plus accumulated prior functions, ready to prepend to the
    newly generated step code before testing.
    """
    problem_steps_str, next_step_str, previous_code_str = process_problem_steps(
        sub_steps, num_steps, previous_llm_code, with_background
    )
    assert next_step_str
    prompt = prompt_template(with_background).format(
        problem_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=required_dependencies,
    )
    return prompt, f"{required_dependencies}\n{previous_code_str}\n"
