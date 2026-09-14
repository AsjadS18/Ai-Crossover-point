"""Model pair loading and prompt formatting.

The tokenizer always comes from the TARGET path.  The target defines correct
output, so its tokenizer defines the token ids everything else is measured in.
tests/test_tokenizers.py proves the draft agrees.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from src import compat

TARGET_PATH = "models/target-7b"
DRAFT_PATH = "models/draft-0.5b"


def load_pair(
    target_path: str = TARGET_PATH,
    draft_path: str = DRAFT_PATH,
    device: str = "cuda",
) -> tuple[Any, Any, Any]:
    """Load the target/draft pair.

    Returns (tokenizer, target_model, draft_model).  Target is 4-bit NF4
    (~5 GB VRAM), draft is plain float16 (~1.2 GB).  Both are in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target = compat.load_model(target_path, quantize=True, device=device)
    draft = compat.load_model(draft_path, quantize=False, device=device)
    return tokenizer, target, draft


def chat_ids(tokenizer: Any, prompt: str, device: str = "cuda") -> torch.Tensor:
    """Format `prompt` as a single user turn and tokenize it.

    Returns an int64 tensor of shape [1, T] on `device`, ending with the
    generation prompt so the model's next token is the start of its reply.
    """
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(TARGET_PATH)
    ids = chat_ids(tok, "Write a function that reverses a list.", device="cpu")
    print("shape:", tuple(ids.shape), "dtype:", ids.dtype)
    print("decoded:")
    print(tok.decode(ids[0]))
