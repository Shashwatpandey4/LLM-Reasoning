import re
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

from src.registry import MODELS

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    If the block is incomplete (no closing tag), the model ran out of tokens
    mid-thought and produced no usable answer — return empty string so the
    caller records a parse failure rather than extracting garbage.
    """
    if "<think>" not in text:
        return text
    if "</think>" not in text:
        return ""
    return _THINK_RE.sub("", text).strip()


class HuggingFaceModelWrapper:
    """Wrapper for Hugging Face models using the transformers pipeline.

    Args:
        model_name: HuggingFace model identifier.
        max_new_tokens: Default token budget per generation call.
        temperature: Default sampling temperature.
        strip_thinking: If True, remove <think>...</think> blocks from outputs
            before returning. Required for DeepSeek-R1 distill models whose
            outputs contain extended chain-of-thought wrapped in these tags.
        system_prompt: Optional system prompt prepended to every message.
            Used to enable thinking mode on Qwen3 models.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        strip_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.strip_thinking = strip_thinking
        self.system_prompt = system_prompt

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        print(f"Model {model_name} loaded on {device}.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt using the chat template."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
            do_sample=(self.temperature if temperature is None else temperature) > 0.0,
            return_full_text=False,
        )

        text = outputs[0]["generated_text"]
        if self.strip_thinking:
            text = _strip_thinking(text)
        return text

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's own tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Class 1 — Compact IT (existing)
# ---------------------------------------------------------------------------


@MODELS.register("gemma-3-1b-it")
def load_gemma_3_1b_it(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("google/gemma-3-1b-it", **kwargs)


@MODELS.register("qwen3-0.6b")
def load_qwen3_0_6b(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("Qwen/Qwen3-0.6B", system_prompt="/no_think", **kwargs)


@MODELS.register("qwen3-1.7b")
def load_qwen3_1_7b(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("Qwen/Qwen3-1.7B", system_prompt="/no_think", **kwargs)


@MODELS.register("qwen3-4b")
def load_qwen3_4b(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("Qwen/Qwen3-4B", system_prompt="/no_think", **kwargs)


@MODELS.register("qwen3-8b")
def load_qwen3_8b(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("Qwen/Qwen3-8B", system_prompt="/no_think", **kwargs)


@MODELS.register("qwen3-14b")
def load_qwen3_14b(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("Qwen/Qwen3-14B", system_prompt="/no_think", **kwargs)


@MODELS.register("phi-3.5-mini")
def load_phi_3_5_mini(**kwargs) -> HuggingFaceModelWrapper:
    """Phi-3.5-mini-Instruct (3.8B) — Microsoft. Class 1 representative."""
    return HuggingFaceModelWrapper("microsoft/Phi-3.5-mini-instruct", **kwargs)


# ---------------------------------------------------------------------------
# Class 2 — Mid-size IT
# ---------------------------------------------------------------------------


@MODELS.register("llama-3.1-8b")
def load_llama_3_1_8b(**kwargs) -> HuggingFaceModelWrapper:
    """Llama-3.1-8B-Instruct (8B) — Meta. Class 2 representative."""
    return HuggingFaceModelWrapper("meta-llama/Llama-3.1-8B-Instruct", **kwargs)


# ---------------------------------------------------------------------------
# Class 3 — Large IT
# ---------------------------------------------------------------------------


@MODELS.register("gemma-3-27b-it")
def load_gemma_3_27b_it(**kwargs) -> HuggingFaceModelWrapper:
    """Gemma-3-27B-IT (27B) — Google. Class 3 representative."""
    return HuggingFaceModelWrapper("google/gemma-3-27b-it", **kwargs)


# ---------------------------------------------------------------------------
# Class 4 — Thinking / RL-Reasoning
# ---------------------------------------------------------------------------


@MODELS.register("deepseek-r1-distill-7b")
def load_deepseek_r1_distill_7b(**kwargs) -> HuggingFaceModelWrapper:
    """DeepSeek-R1-Distill-Qwen-7B (7B) — DeepSeek. Class 4 representative.

    Outputs extended chain-of-thought inside <think>...</think> tags before
    the final answer. strip_thinking=True removes these blocks so downstream
    parsers see only the answer text.
    """
    return HuggingFaceModelWrapper(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        strip_thinking=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Class 5 — MoE
# ---------------------------------------------------------------------------


@MODELS.register("mixtral-8x7b")
def load_mixtral_8x7b(**kwargs) -> HuggingFaceModelWrapper:
    """Mixtral-8x7B-Instruct-v0.1 (~13B active / 47B total) — Mistral.
    Class 5 representative. Requires device_map='auto' across multiple GPUs.
    """
    return HuggingFaceModelWrapper("mistralai/Mixtral-8x7B-Instruct-v0.1", **kwargs)


