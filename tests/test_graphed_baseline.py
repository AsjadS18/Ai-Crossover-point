"""Graphed greedy decoding must match the eager static path, and be faster.

Compared against STATIC-eager, not dynamic-eager: a StaticCache and a
DynamicCache drift apart over a long decode (see test_compat.py), so comparing
across both cache type and execution mode would conflate two effects.

The risk this file exists to catch: one captured graph is reused across prompts
of different lengths. If anything length-dependent were baked in at capture, the
second prompt's output would be silently wrong.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compat
from src.baseline import GraphedGreedyDecoder, baseline_generate
from src.models import DRAFT_PATH, chat_ids

MAX_CACHE = 512
STEPS = 64

PROMPTS = [
    "Write a Python function that reverses a list in place.",
    "Explain what a hash map is and when you would use one, in detail, with "
    "examples of collision handling and load factors.",
    "hi",
    "Return a JSON object describing a book: title, author, year, tags.",
]

_loaded: tuple | None = None


def _load() -> tuple:
    global _loaded
    if _loaded is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(DRAFT_PATH)
        model = compat.load_model(DRAFT_PATH, quantize=False)
        _loaded = (tok, model)
    return _loaded


@pytest.fixture(scope="module")
def pair():
    return _load()


def _static_eager_tokens(model, ids, steps: int) -> list[int]:
    """Greedy ids using a StaticCache, eagerly.  The reference for this file.

    Mirrors baseline_generate: the prefill's own last logits give token 1, then
    each generated token is fed once.  The last prompt token is never re-fed.
    """
    cache = compat.make_static_cache(model, MAX_CACHE)
    with torch.inference_mode():
        prefill = model(input_ids=ids, past_key_values=cache, use_cache=True)
        nxt = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = [int(nxt.item())]
        for _ in range(steps - 1):
            logits = model(input_ids=nxt, past_key_values=cache,
                           use_cache=True).logits
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(int(nxt.item()))
    return out


def test_graphed_matches_static_eager(pair):
    """Graphed output must equal static-eager output, exactly."""
    tok, model = pair
    ids = chat_ids(tok, PROMPTS[0])
    ref = _static_eager_tokens(model, ids, STEPS)

    dec = GraphedGreedyDecoder(model, max_cache_len=MAX_CACHE)
    seq, stats = dec.generate(ids, max_new_tokens=STEPS, eos_id=None)
    got = seq[0, ids.shape[1]:].tolist()

    match = got == ref
    first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), -1)
    print(f"[{'PASS' if match else 'FAIL'}] graphed vs static-eager: "
          f"{len(got)} tokens, identical={match}"
          + ("" if match else f", diverges at {first}"))
    assert match


def test_graph_reused_across_prompt_lengths(pair):
    """ONE captured graph, four prompts of different lengths, all correct.

    If capture baked in the first prompt's length, prompts 2-4 would be wrong.
    """
    tok, model = pair
    dec = GraphedGreedyDecoder(model, max_cache_len=MAX_CACHE)

    for i, prompt in enumerate(PROMPTS):
        ids = chat_ids(tok, prompt)
        ref = _static_eager_tokens(model, ids, 32)
        seq, _ = dec.generate(ids, max_new_tokens=32, eos_id=None)
        got = seq[0, ids.shape[1]:].tolist()
        match = got == ref
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), -1)
        print(f"[{'PASS' if match else 'FAIL'}] prompt {i} "
              f"({ids.shape[1]} tokens): identical={match}"
              + ("" if match else f", diverges at {first}"))
        assert match, f"prompt {i} diverged; the graph is length-dependent"


def test_graphed_is_faster(pair):
    """The whole point: fewer kernel launches must show up as throughput."""
    tok, model = pair
    ids = chat_ids(tok, PROMPTS[0])

    _, eager_stats = baseline_generate(model, ids, max_new_tokens=STEPS,
                                      eos_id=None)
    dec = GraphedGreedyDecoder(model, max_cache_len=MAX_CACHE)
    dec.generate(ids, max_new_tokens=8, eos_id=None)      # absorb capture
    _, graph_stats = dec.generate(ids, max_new_tokens=STEPS, eos_id=None)

    ratio = graph_stats["tok_per_s"] / eager_stats["tok_per_s"]
    print(f"[INFO] eager  {eager_stats['tok_per_s']:.2f} tok/s")
    print(f"[INFO] graphed {graph_stats['tok_per_s']:.2f} tok/s  "
          f"({ratio:.2f}x)  prefill {graph_stats['prefill_seconds'] * 1000:.1f} ms")
    assert ratio > 1.5, f"graphed decode only {ratio:.2f}x eager"


def test_rejects_overflowing_cache(pair):
    """A static cache cannot grow, so overflow must fail loudly, not corrupt."""
    tok, model = pair
    ids = chat_ids(tok, PROMPTS[0])
    dec = GraphedGreedyDecoder(model, max_cache_len=64)
    with pytest.raises(ValueError):
        dec.generate(ids, max_new_tokens=256, eos_id=None)
    print("[PASS] overflowing the static cache raises instead of corrupting")


def test_eos_stops_generation(pair):
    """EOS must terminate the graphed loop too."""
    tok, model = pair
    ids = chat_ids(tok, "Reply with exactly one word: the capital of Japan.")
    dec = GraphedGreedyDecoder(model, max_cache_len=MAX_CACHE)
    seq, stats = dec.generate(ids, max_new_tokens=200, eos_id=tok.eos_token_id)
    last = int(seq[0, -1].item())
    print(f"[INFO] stopped after {stats['tokens']} tokens, last={last}, "
          f"eos={tok.eos_token_id}")
    assert stats["tokens"] < 200
    assert last == tok.eos_token_id


if __name__ == "__main__":
    tok, model = _load()
    ids = chat_ids(tok, PROMPTS[0])
    ref = _static_eager_tokens(model, ids, STEPS)
    dec = GraphedGreedyDecoder(model, max_cache_len=MAX_CACHE)
    seq, stats = dec.generate(ids, max_new_tokens=STEPS, eos_id=None)
    got = seq[0, ids.shape[1]:].tolist()
    print("identical:", got == ref)
    print("stats:", stats)
    print(tok.decode(got))
