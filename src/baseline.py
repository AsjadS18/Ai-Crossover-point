"""Standard greedy decoding, hand-written.

Deliberately NOT model.generate().  The speculative decoder manages its own
cache, and the two paths must be identical apart from the speculation itself,
or the speed comparison measures the wrong thing.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from src import compat


def forward_new(
    model: Any, cache: Any, seq: torch.Tensor, cached_len: int
) -> torch.Tensor:
    """Run one forward pass over only the tokens the cache has not seen.

    Feeds seq[:, cached_len:] and appends to `cache` in place.  Returns logits
    of shape [1, seq.shape[1] - cached_len, vocab].  The caller is responsible
    for advancing its own cached_len to seq.shape[1] afterwards.
    """
    return model(
        input_ids=seq[:, cached_len:],
        past_key_values=cache,
        use_cache=True,
    ).logits


@torch.inference_mode()
def baseline_generate(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_id: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedy-decode up to `max_new_tokens` tokens from `input_ids`.

    Returns (sequence, stats) where sequence is [1, prompt + generated] and
    stats holds:
        tokens     int   generated tokens, excluding the prompt
        seconds    float wall clock for the generation loop
        tok_per_s  float tokens / seconds
    """
    cache = compat.make_cache(model)
    seq = input_ids
    cached_len = 0
    generated = 0

    torch.cuda.synchronize()
    started = time.perf_counter()

    for _ in range(max_new_tokens):
        logits = forward_new(model, cache, seq, cached_len)
        cached_len = seq.shape[1]

        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
        generated += 1

        if eos_id is not None and next_tok.item() == eos_id:
            break

    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    stats = {
        "tokens": generated,
        "seconds": seconds,
        "tok_per_s": generated / seconds if seconds > 0 else 0.0,
    }
    return seq, stats


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from src.models import DRAFT_PATH, chat_ids

    tok = AutoTokenizer.from_pretrained(DRAFT_PATH)
    model = compat.load_model(DRAFT_PATH, quantize=False)
    ids = chat_ids(tok, "Write a Python function that reverses a list.")

    seq, stats = baseline_generate(model, ids, max_new_tokens=64,
                                  eos_id=tok.eos_token_id)
    print(tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True))
    print("stats:", stats)
