from src.registry import PROMPTS


def build_mcq_prompt(question: str, choices: dict) -> str:
    choices_str = "\n".join(f"({k}) {v}" for k, v in choices.items())
    return (
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Reason step by step, then write your final answer as: "
        f'"The answer is (X)" where X is A, B, C, or D.'
    )


class MCQCoTPromptTemplate:
    def format(self, **kwargs) -> str:
        return build_mcq_prompt(kwargs["question"], kwargs["choices"])


@PROMPTS.register("mcq_cot")
def get_mcq_cot_prompt(**kwargs) -> MCQCoTPromptTemplate:
    return MCQCoTPromptTemplate()
