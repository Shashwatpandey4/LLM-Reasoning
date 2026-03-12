import re
from typing import Optional, Tuple
from src.registry import PARSERS

class GSM8KParser:
    """Parses standard GSM8K outputs."""
    
    def parse(self, text: str) -> Tuple[Optional[str], int]:
        """
        Attempts to extract the final answer from the model's text.
        Based on our prompt, we expect: 'The answer is [number]'
        If that fails, falls back to finding the last number in the text.
        
        Returns:
            Tuple[Optional[str], int]: The extracted answer string and the character index
                                       where the answer string begins in the raw text.
        """
        
        # 1. Try prompt-specific extraction 'The answer is X'
        # We want to know where 'The answer is' starts
        match = re.search(r"The answer is\s*\**([0-9.,]+)", text, re.IGNORECASE)
        if match:
            ans = match.group(1).replace(",", "")
            # Return just the digits (and period)
            clean_ans = re.sub(r"[^\d.]", "", ans)
            if clean_ans:
                # Remove trailing periods
                clean_ans = clean_ans.rstrip('.')
                return clean_ans, match.start()
        
        # 2. Fallback: Find the last continuous string of digits
        numbers = list(re.finditer(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text.replace(",", "")))
        if numbers:
            # Often models say "... is 42."
            # finditer gets the match objects so we can find the start index
            last_match = numbers[-1]
            return last_match.group(), last_match.start()
            
        return None, len(text)

@PARSERS.register("gsm8k_parser")
def get_gsm8k_parser(**kwargs) -> GSM8KParser:
    return GSM8KParser()
