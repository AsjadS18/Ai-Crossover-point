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


class GraphedGreedyDecoder:
    """Greedy decoding whose decode step is a replayed CUDA graph.

    Same algorithm as baseline_generate, but the per-token forward is captured
    once and replayed, which removes the ~940 eager kernel launches that dominate
    this platform.  Measured 2.6x on the 4-bit 7B and 7.3x on the 0.5B, with
    token-identical output.

    The prefill stays eager because its length varies.  One decoder instance
    holds one static cache and one graph and is reused across prompts: reset()
    zeroes the cache in place, so the addresses the graph baked in stay valid.

    Capture happens once, on the first generate() call, and is NOT included in
    any reported timing -- it is one-off setup, amortised across a run.
    """

    def __init__(
        self, model: Any, max_cache_len: int = 512, device: str = "cuda"
    ) -> None:
        self.model = model
        self.device = device
        self.max_cache_len = max_cache_len
        self.cache = compat.make_static_cache(model, max_cache_len)
        self.graph: compat.GraphedForward | None = None

    def _ensure_graph(self, input_ids: torch.Tensor) -> None:
        """Capture the decode graph on a throwaway prefill.

        Capture must not share a run with a real generation: its warmup passes
        write into the cache, and a generation that starts from that residue
        produces different output than one starting from a clean reset. Doing it
        on a throwaway prefill keeps every generate() call identical, first
        included.
        """
        if self.graph is not None:
            return
        with torch.inference_mode():
            self.cache.reset()
            self.model(input_ids=input_ids, past_key_values=self.cache,
                       use_cache=True)
            self.graph = compat.GraphedForward(
                self.model, self.cache, width=1, device=self.device)
            self.cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_id: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Greedy-decode from `input_ids`.  Same contract as baseline_generate.

        Returns (sequence, stats) with stats holding tokens, seconds, tok_per_s
        plus prefill_seconds, so prefill and decode can be reported separately.
        """
        prompt_len = input_ids.shape[1]
        if prompt_len + max_new_tokens > self.max_cache_len:
            raise ValueError(
                f"prompt {prompt_len} + {max_new_tokens} new tokens exceeds "
                f"max_cache_len {self.max_cache_len}; a static cache cannot grow"
            )

        # Capture before any timing, on a throwaway prefill, so that every run
        # including the first starts from an identical clean cache.
        self._ensure_graph(input_ids)
        self.cache.reset()

        torch.cuda.synchronize()
        started = time.perf_counter()
        prefill = self.model(input_ids=input_ids, past_key_values=self.cache,
                             use_cache=True)
        # The prefill's own last logits give the FIRST generated token, exactly
        # as baseline_generate's first loop iteration does. Re-feeding the last
        # prompt token here instead would double-feed it and change the output.
        first_id = int(prefill.logits[0, -1, :].argmax().item())
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started

        seq = torch.cat(
            [input_ids, torch.tensor([[first_id]], device=self.device)], dim=1)
        generated = 1
        tok_buf = torch.tensor([[first_id]], device=self.device)

        torch.cuda.synchronize()
        decode_started = time.perf_counter()

        if not (eos_id is not None and first_id == eos_id):
            for _ in range(max_new_tokens - 1):
                logits = self.graph.replay(tok_buf)
                next_id = int(logits[0, -1, :].argmax().item())
                seq = torch.cat(
                    [seq, torch.tensor([[next_id]], device=self.device)], dim=1)
                generated += 1
                if eos_id is not None and next_id == eos_id:
                    break
                tok_buf.fill_(next_id)

        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        seconds = prefill_seconds + decode_seconds

        stats = {
            "tokens": generated,
            "seconds": seconds,
            "tok_per_s": generated / seconds if seconds > 0 else 0.0,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "graphed": True,
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
