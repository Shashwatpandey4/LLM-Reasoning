from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseReasoningMethod(ABC):
    """
    Abstract base class for a Reasoning Method.
    It takes a model, a prompt template, and a parser to execute a reasoning strategy.
    """
    def __init__(self, model: Any, prompt_template: Any, parser: Any):
        self.model = model
        self.prompt_template = prompt_template
        self.parser = parser

    @abstractmethod
    def run(self, instance: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the reasoning method on a single instance.
        Returns a dictionary containing the extracted answer and any intermediate reasoning.
        """
        pass
