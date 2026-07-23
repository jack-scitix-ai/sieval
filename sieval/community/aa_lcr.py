# prompt template + equality-checker prompt: AA-LCR dataset card (Apache-2.0)
# https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR
#   revision bdae010bbce259820c0e34c1d7cce210d966fb75, README "Prompt Template"
#   and "Scoring Approach" sections — reproduced verbatim.
"""Artificial Analysis Long Context Reasoning (AA-LCR) prompt + scoring assets.

AA-LCR (Artificial Analysis) is a 100-question long-context reasoning benchmark:
each question is answered against a set of real-world documents (~100k tokens per
set) loaded into the same prompt, and answers must be *reasoned* across multiple
sources rather than directly retrieved.

The upstream repo ships no evaluation code — only the dataset card. Both the
input ``PROMPT_TEMPLATE`` (with per-document ``BEGIN DOCUMENT``/``END DOCUMENT``
wrapping via ``DOCUMENT_TEMPLATE``) and the ``GRADER_TEMPLATE`` below are the
card's own snippets reproduced verbatim. Deviations from upstream:

* ``build_prompt`` mirrors the card's two-step assembly (join documents, then
  fill the outer template); documents are joined in the given order — the card
  requires ``data_source_filenames`` order, which the loader preserves.
* Grading is **binary** ``CORRECT``/``INCORRECT`` (no ``NOT_ATTEMPTED`` tier),
  so ``parse_grade`` maps an unrecognized reply to ``INCORRECT`` (conservative),
  and ``aggregate_metrics`` reports plain accuracy — unlike the SimpleQA-style
  A/B/C + F1 aggregation in ``sieval.community.simpleqa_verified``.

The official equality checker is Qwen3 235B A22B 2507 Non-reasoning; the checker
model is a runtime choice supplied by the task, not pinned here.

AI-Generated Code - Claude Opus 4.8 (Anthropic)
"""

import re

# Per-document wrapper; ``index`` is 0-based, printed 1-based per the card.
DOCUMENT_TEMPLATE = "BEGIN DOCUMENT {number}:\n{document}\nEND DOCUMENT {number}"

# Outer prompt framing the joined documents and the question.
PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

# LLM equality-checker prompt (binary CORRECT/INCORRECT).
GRADER_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT.
"""


def build_prompt(documents: list[str], question: str) -> str:
    """Assemble the AA-LCR user prompt from ordered *documents* + *question*.

    Documents are wrapped and joined in the order given — the caller is
    responsible for supplying them in ``data_source_filenames`` order, which the
    dataset loader guarantees.
    """
    documents_text = "\n\n".join(
        DOCUMENT_TEMPLATE.format(number=i + 1, document=doc)
        for i, doc in enumerate(documents)
    )
    return PROMPT_TEMPLATE.format(documents_text=documents_text, question=question)


def parse_grade(grading_response: str) -> str:
    """Map an equality-checker reply to ``CORRECT`` / ``INCORRECT``.

    The grader is instructed to reply with only ``CORRECT`` or ``INCORRECT``, so
    the leading verdict wins. ``INCORRECT`` takes precedence over ``CORRECT``:
    it contains ``CORRECT`` as a substring, and a reply that mentions both
    (e.g. "not CORRECT, so INCORRECT") is a negative verdict. Anything without a
    recognizable verdict — empty or malformed — is ``INCORRECT``; AA-LCR has no
    not-attempted tier.
    """
    text = grading_response.strip().upper()
    if text.startswith("INCORRECT"):
        return "INCORRECT"
    if text.startswith("CORRECT"):
        return "CORRECT"
    if re.search(r"\bINCORRECT\b", text):
        return "INCORRECT"
    if re.search(r"\bCORRECT\b", text):
        return "CORRECT"
    return "INCORRECT"


def aggregate_metrics(grades: list[str]) -> dict[str, float]:
    """Aggregate per-sample grades into AA-LCR accuracy (in ``[0, 1]``).

    Returns the correct/incorrect rates and the headline ``accuracy`` (the
    correct rate). Empty input yields all-zero metrics.
    """
    total = len(grades)
    if total == 0:
        return {"accuracy": 0.0, "is_correct": 0.0, "is_incorrect": 0.0}

    is_correct = sum(g == "CORRECT" for g in grades) / total
    return {
        "accuracy": is_correct,
        "is_correct": is_correct,
        "is_incorrect": 1.0 - is_correct,
    }
