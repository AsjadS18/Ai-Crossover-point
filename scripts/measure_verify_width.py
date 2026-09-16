"""Why is gamma=3 a dip in every domain?

Graphed speedups are not monotone in gamma: every one of the six domains dips
at gamma=3 and recovers by gamma=5 (math 1.239 -> 1.178 -> 1.362, structured
1.180 -> 1.072 -> 1.197, code 1.155 -> 1.071 -> 1.189). Six for six, on flat
baselines and identical prompts, so it is not contamination.

A speculative round verifies gamma+1 tokens in ONE forward, so gamma=3 means a
width-4 forward. If width 4 lands on a worse GEMM tile than widths 3 and 6, the
verify cost is not linear in gamma, the cost model gamma*r+1 cannot predict the
optimum, and the dip is a hardware artifact rather than anything about
speculation.

This times the captured verify graph directly, per width, and reports cost per
token so the non-linearity is visible.

    python scripts/measure_verify_width.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src import compat
from src.models import TARGET_PATH, chat_ids, load_pair

MAX_CACHE = 256
WIDTHS = tuple(range(1, 14))
WARMUP = 10
TIMED = 50
OUT_PATH = "results/verify_width.json"
PROMPT = "Write a Python function that reverses a list in place."


@torch.inference_mode()
def time_width(model: Any, ids: torch.Tensor, width: int) -> dict[str, float]:
    """Mean ms for one graphed forward of `width` tokens, and per-token cost."""
    cache = compat.make_static_cache(model, MAX_CACHE)
    model(input_ids=ids, past_key_values=cache, use_cache=True)
    prefill_len = compat.cache_length(cache)

    graph = compat.GraphedForward(model, cache, width=width)
    tokens = torch.zeros(1, width, dtype=torch.long, device="cuda")

    for _ in range(WARMUP):
        graph.replay(tokens, start_position=prefill_len)
        compat.crop_cache(cache, prefill_len)

    total = 0.0
    for _ in range(TIMED):
        torch.cuda.synchronize()
        started = time.perf_counter()
        graph.replay(tokens, start_position=prefill_len)
        torch.cuda.synchronize()
        total += time.perf_counter() - started
        compat.crop_cache(cache, prefill_len)

    ms = (total / TIMED) * 1000.0
    return {"width": width, "ms": ms, "ms_per_token": ms / width}


def main() -> int:
    tokenizer, target, draft = load_pair()
    ids = chat_ids(tokenizer, PROMPT)

    print(f"graphed verify forward, widths {WIDTHS[0]}-{WIDTHS[-1]}, "
          f"{TIMED} timed runs each\n")
    print(f"{'width':>6} {'gamma':>6} {'ms':>9} {'ms/token':>10} {'vs width 1':>11}")

    rows: list[dict[str, float]] = []
    for width in WIDTHS:
        row = time_width(target, ids, width)
        rows.append(row)
        base = rows[0]["ms"]
        print(f"{width:6d} {width - 1:6d} {row['ms']:9.3f} "
              f"{row['ms_per_token']:10.3f} {row['ms'] / base:11.3f}",
              flush=True)

    # A dip means width w costs MORE per token than both its neighbours.
    print("\nnon-monotonic points (ms/token worse than both neighbours):")
    flagged = []
    for i in range(1, len(rows) - 1):
        prev, cur, nxt = rows[i - 1], rows[i], rows[i + 1]
        if cur["ms_per_token"] > prev["ms_per_token"] and \
           cur["ms_per_token"] > nxt["ms_per_token"]:
            flagged.append(cur["width"])
            print(f"  width {cur['width']} (gamma={cur['width'] - 1}): "
                  f"{cur['ms_per_token']:.3f} ms/token vs "
                  f"{prev['ms_per_token']:.3f} and {nxt['ms_per_token']:.3f}")
    if not flagged:
        print("  none -- cost per token is monotone, so the gamma=3 dip needs "
              "another explanation")

    record = {
        "widths": rows,
        "non_monotonic_widths": flagged,
        "gamma3_is_width4_dip": 4 in flagged,
        "timed_runs": TIMED,
        "max_cache_len": MAX_CACHE,
        "target_path": TARGET_PATH,
        "compat": compat.describe(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.flush()
    print(f"\nwrote {OUT_PATH}")
    print("VERDICT:", "width 4 IS a per-token cost dip -- explains gamma=3"
          if 4 in flagged else
          "width 4 is NOT a cost dip -- the gamma=3 dip is something else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
