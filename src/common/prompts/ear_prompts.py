"""Prompt templates for the EAR (Equilibrium Aggregation Reasoning) method.

Three critique strategies, one shared judge prompt, one shared revision prompt.
All are registered in the PROMPTS registry.
"""

from src.common.prompts.gsm8k_cot import PromptTemplate
from src.registry import PROMPTS

# ---------------------------------------------------------------------------
# Critique prompts
# ---------------------------------------------------------------------------

# answer_level: external critic sees only the question and the candidate's final answer.
_ANSWER_LEVEL_CRITIQUE = """\
You are evaluating an answer to a math problem.

Question: {question}

Candidate's answer: {answer}

Identify one specific flaw or error in this answer. \
If you believe the answer is correct, state that clearly. \
Be concise and direct."""


# reasoning_level: external critic sees the question and the full reasoning chain.
_REASONING_LEVEL_CRITIQUE = """\
You are evaluating a solution to a math problem.

Question: {question}

Candidate's reasoning:
{reasoning}

Candidate's final answer: {answer}

Identify one specific logical error or flaw in the reasoning above. \
Point to the exact step that is wrong. \
If you believe the reasoning is correct, state that clearly. \
Be concise."""


# panel: cross-candidate critique. The critiquer sees all candidates' solutions
# and is asked to critique a specific target candidate.
_PANEL_CRITIQUE = """\
You are participating in a panel reviewing multiple solutions to a math problem.

Question: {question}

All candidate solutions:
{panel_context}

Your task: critique Candidate {target_id}'s solution specifically.

Candidate {target_id}'s reasoning:
{target_reasoning}

Candidate {target_id}'s answer: {target_answer}

Identify one specific flaw in Candidate {target_id}'s reasoning or answer. \
Use the other candidates' solutions as reference if helpful. \
Be precise and direct."""


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE = """\
You are judging whether a critique of a math solution is valid.

Question: {question}

Candidate's reasoning:
{reasoning}

Candidate's answer: {answer}

Critique: {critique}

Does this critique correctly identify a genuine error in the candidate's reasoning or answer?

Respond with exactly "VALID" if the critique is correct and points to a real flaw, \
or "INVALID" if the critique is wrong, irrelevant, or the candidate's answer is actually correct. \
Then briefly explain your judgment in one sentence."""


# ---------------------------------------------------------------------------
# Revision prompt
# ---------------------------------------------------------------------------

_REVISION = """\
You are revising your solution to a math problem based on feedback.

Question: {question}

Your original reasoning:
{reasoning}

Your original answer: {answer}

Critique of your solution: {critique}

If this critique identifies a genuine error, revise your solution. \
If the critique is incorrect, explain why and maintain your original answer.

Show your reasoning step by step. \
End your response with "The answer is [number]"."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@PROMPTS.register("ear_answer_level_critique")
def _get_answer_level_critique(**kwargs) -> PromptTemplate:
    return PromptTemplate(_ANSWER_LEVEL_CRITIQUE)


@PROMPTS.register("ear_reasoning_level_critique")
def _get_reasoning_level_critique(**kwargs) -> PromptTemplate:
    return PromptTemplate(_REASONING_LEVEL_CRITIQUE)


@PROMPTS.register("ear_panel_critique")
def _get_panel_critique(**kwargs) -> PromptTemplate:
    return PromptTemplate(_PANEL_CRITIQUE)


@PROMPTS.register("ear_judge")
def _get_judge_prompt(**kwargs) -> PromptTemplate:
    return PromptTemplate(_JUDGE)


@PROMPTS.register("ear_revision")
def _get_revision_prompt(**kwargs) -> PromptTemplate:
    return PromptTemplate(_REVISION)
