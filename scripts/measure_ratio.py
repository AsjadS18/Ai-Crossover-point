"""Measure the draft/target cost ratio r, plus target baseline throughput.

r = draft_ms / target_ms for a single-token forward pass with a warm cache.
That is the regime speculation runs in: every drafted token costs one
single-token draft forward, and verification costs one target forward.

Writes results/cost_ratio.json.  Gate: r should land between 0.06 and 0.20.
If r > 0.3 the wrong draft model is loaded.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compat
from src.baseline import baseline_generate, forward_new
from src.models import DRAFT_PATH, TARGET_PATH, chat_ids, load_pair

WARMUP = 10
TIMED = 50
PRIME_PROMPT = "Write a Python function that reverses a list in place."
BASELINE_PROMPT = "Explain what a hash map is and when you would use one."
BASELINE_TOKENS = 128
OUT_PATH = "results/cost_ratio.json"


@torch.inference_mode()
def time_single_token_forward(model: Any, ids: torch.Tensor) -> float:
    """Mean wall clock of one single-token forward pass, in milliseconds.

    The cache is primed with `ids` first, then one token is fed repeatedly.
    The cache is cropped back after each timed pass so every measurement runs
    at the same cache length.
    """
    cache = compat.make_cache(model)
    forward_new(model, cache, ids, 0)
    primed_len = compat.cache_length(cache)

    one = ids[:, -1:]
    seq = torch.cat([ids, one], dim=1)

    for _ in range(WARMUP):
        forward_new(model, cache, seq, primed_len)
        compat.crop_cache(cache, primed_len)

    # The crop is deliberately outside the timed region: it is bookkeeping for
    # this measurement, not part of a forward pass.
    total = 0.0
    for _ in range(TIMED):
        torch.cuda.synchronize()
        started = time.perf_counter()
        forward_new(model, cache, seq, primed_len)
        torch.cuda.synchronize()
        total += time.perf_counter() - started
        compat.crop_cache(cache, primed_len)

    return (total / TIMED) * 1000.0


def main() -> int:
    tokenizer, target, draft = load_pair()
    prime_ids = chat_ids(tokenizer, PRIME_PROMPT)

    print(f"primed cache length: {prime_ids.shape[1]} tokens")
    print(f"timing {TIMED} single-token forwards after {WARMUP} warmups\n")

    draft_ms = time_single_token_forward(draft, prime_ids)
    target_ms = time_single_token_forward(target, prime_ids)
    r = draft_ms / target_ms

    print(f"draft_ms:  {draft_ms:.3f}")
    print(f"target_ms: {target_ms:.3f}")
    print(f"r:         {r:.4f}")

    base_ids = chat_ids(tokenizer, BASELINE_PROMPT)
    _, base_stats = baseline_generate(
        target, base_ids, max_new_tokens=BASELINE_TOKENS,
        eos_id=tokenizer.eos_token_id,
    )
    print(f"\nbaseline target throughput: {base_stats['tok_per_s']:.2f} tok/s "
          f"({base_stats['tokens']} tokens in {base_stats['seconds']:.2f}s)")

    record: dict[str, Any] = {
        "draft_ms": draft_ms,
        "target_ms": target_ms,
        "r": r,
        "warmup_runs": WARMUP,
        "timed_runs": TIMED,
        "primed_tokens": prime_ids.shape[1],
        "baseline_tok_per_s": base_stats["tok_per_s"],
        "baseline_tokens": base_stats["tokens"],
        "baseline_seconds": base_stats["seconds"],
        "baseline_max_new_tokens": BASELINE_TOKENS,
        "target_path": TARGET_PATH,
        "draft_path": DRAFT_PATH,
        "compat": compat.describe(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.flush()
    print(f"\nwrote {OUT_PATH}")

    if not 0.06 <= r <= 0.20:
        print(f"WARNING: r = {r:.4f} is outside the expected 0.06 - 0.20 band")
        if r > 0.3:
            print("r > 0.3 -- check that models/draft-0.5b is really the 0.5B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
