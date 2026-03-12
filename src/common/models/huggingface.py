from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

from src.registry import MODELS


class HuggingFaceModelWrapper:
    """Wrapper for Hugging Face models using the transformers pipeline."""

    def __init__(self, model_name: str, max_new_tokens: int = 512, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        print(f"Model {model_name} loaded successfully on {device}.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generates text from a prompt, handling chat templates if needed."""
        # For instruct/chat models, we often need to wrap in messages
        messages = [{"role": "user", "content": prompt}]

        # Use simple string pipeline execution
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
            do_sample=(self.temperature if temperature is None else temperature) > 0.0,
            return_full_text=False,
        )

        return outputs[0]["generated_text"]

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the given text using the model's tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


@MODELS.register("gemma-3-1b-it")
def load_gemma_3_1b_it(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("google/gemma-3-1b-it", **kwargs)
