from src.registry import PROMPTS


class PromptTemplate:
    """Basic interface for prompt templates."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


# A zero-shot CoT prompt for GSM8K.
# We add "Let's think step by step" to encourage reasoning,
# and ask for a specific format for the final answer so its easy to parse.

GSM8K_ZERO_SHOT_COT_TEMPLATE = """Solve the following math word problem.
Reason step-by-step to arrive at the solution.
At the very end of your response, write your final numerical answer in the
format: "The answer is [number]".

Question: {question}

Response: Let's think step by step.
"""


@PROMPTS.register("gsm8k_cot")
def get_gsm8k_cot_prompt(**kwargs) -> PromptTemplate:
    return PromptTemplate(GSM8K_ZERO_SHOT_COT_TEMPLATE)
