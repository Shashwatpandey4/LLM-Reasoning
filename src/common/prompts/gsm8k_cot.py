from src.registry import PROMPTS


class PromptTemplate:
    """Basic interface for prompt templates."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


GSM8K_ZERO_SHOT_COT_TEMPLATE = """Solve the following math word problem.
Reason step-by-step to arrive at the solution.
At the very end of your response, write your final numerical answer in the
format: "The answer is [number]".

Question: {question}

Response: Let's think step by step.
"""


MCQ_ZERO_SHOT_COT_TEMPLATE = """Solve the following multiple-choice question.
Reason step-by-step before selecting the best answer.
At the very end of your response, write your final answer in the format:
"The answer is [choice letter]".

Question: {question}

Choices:
{choices}

Response: Let's think step by step.
"""


def build_gsm8k_prompt(question: str) -> str:
    return GSM8K_ZERO_SHOT_COT_TEMPLATE.format(question=question)


def _format_choices(choices) -> str:
    """
    Supports either:
    - a list like ["A. ...", "B. ..."]
    - a list like ["...", "..."] -> auto-label as A/B/C...
    - a dict like {"A": "...", "B": "..."}
    """
    if isinstance(choices, dict):
        return "\n".join(f"{key}. {value}" for key, value in choices.items())

    if isinstance(choices, list):
        formatted = []
        for idx, choice in enumerate(choices):
            if isinstance(choice, str):
                stripped = choice.strip()
                # If already labeled, keep as is
                if len(stripped) >= 2 and stripped[1] in [".", ")", ":"]:
                    formatted.append(stripped)
                else:
                    label = chr(ord("A") + idx)
                    formatted.append(f"{label}. {stripped}")
            else:
                label = chr(ord("A") + idx)
                formatted.append(f"{label}. {choice}")
        return "\n".join(formatted)

    raise ValueError("Unsupported choices format. Expected list or dict.")


def build_mcq_prompt(question: str, choices) -> str:
    return MCQ_ZERO_SHOT_COT_TEMPLATE.format(
        question=question,
        choices=_format_choices(choices),
    )


class AdaptiveCoTPromptTemplate:
    """
    Uses the GSM8K prompt for standard QA and the MCQ prompt when `choices` are present.
    """

    def format(self, **kwargs) -> str:
        question = kwargs["question"]
        choices = kwargs.get("choices")
        if choices is not None:
            return build_mcq_prompt(question, choices)
        return build_gsm8k_prompt(question)


@PROMPTS.register("gsm8k_cot")
def get_gsm8k_cot_prompt(**kwargs):
    return AdaptiveCoTPromptTemplate()
