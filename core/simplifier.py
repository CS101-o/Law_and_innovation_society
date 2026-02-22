"""
Stage 2: Response Simplification Layer
Converts technical legal responses into elderly-friendly, plain-language explanations.
Uses mlx-community/Qwen2.5-7B-Instruct-4bit for efficient local inference.
"""

import os
from typing import Optional

try:
    from mlx_lm import load, generate
except ImportError:
    raise ImportError(
        "mlx-lm is not installed. Please run: pip install mlx-lm"
    )


class ResponseSimplifier:
    """
    Takes technical legal responses and simplifies them into
    everyday language suitable for elderly clients.
    """

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"):
        """
        Initialize the simplifier with the base instruct model.

        Args:
            model_id: HuggingFace model ID (defaults to 4-bit quantized Qwen2.5)
        """
        print(f"📖 Loading simplifier model: {model_id}...")
        self.model, self.tokenizer = load(model_id)
        print("✅ Simplifier ready for elderly-friendly responses.")

    def simplify(
        self,
        technical_response: str,
        original_question: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Convert a technical legal response into simple, plain-language explanation.

        Args:
            technical_response: The IRAC response from the fine-tuned model
            original_question: The user's original question (for context)
            max_tokens: Maximum tokens to generate (default 512)

        Returns:
            Simplified response in elderly-friendly language
        """
        # System prompt that enforces simplicity and warmth
        system_prompt = """You are a kind, patient legal assistant helping elderly clients understand the law.
Your job is to rewrite complex legal explanations in simple everyday language.

Rules:
- Use short, simple sentences (one idea per sentence)
- No legal jargon (avoid words like 'estoppel', 'promissory', 'consideration', 'breach')
- Warm, friendly tone - like talking to a friend
- Explain what this means for the person in practical terms
- End with: "Would you like me to explain anything else?"

Be concise but clear. Help them understand without overwhelming them with details."""

        # Format messages using the tokenizer's chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Original question: {original_question}

Complex legal explanation:
{technical_response}

Please rewrite this in simple language for an elderly person.""",
            },
        ]

        # Apply chat template to format messages properly
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate the simplified response
        simplified = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        return simplified.strip()


# Cached singleton to avoid reloading model
_simplifier_instance: Optional[ResponseSimplifier] = None


def get_simplifier() -> ResponseSimplifier:
    """Get or create the global simplifier instance."""
    global _simplifier_instance
    if _simplifier_instance is None:
        _simplifier_instance = ResponseSimplifier()
    return _simplifier_instance
