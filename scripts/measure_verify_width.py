"""Why is gamma=3 a dip in every domain?

Graphed speedups are not monotone in gamma: every one of the six domains dips
at gamma=3 and recovers by gamma=5 (math 1.239 -> 1.178 -> 1.362, structured
1.180 -> 1.072 -> 1.197, code 1.155 -> 1.071 -> 1.189). Six for six, on flat
baselines and identical prompts, so it is not contamination.

A speculative round verifies gamma+1 tokens in ONE forward, so gamma=3 means a
width-4 forward. This times the captured verify graph directly at every width.

ANSWER (measured): the original hypothesis -- a per-token GEMM tiling dip at
width 4 -- was WRONG; cost per token is monotone. The cause is a STEP in
absolute cost: ~7.6 ms per extra token up to width 3, a +11 ms jump into width
4, then nearly flat (~3 ms across widths 4-13). gamma=3 pays the whole step
while verifying only four tokens, which makes it the pessimal gamma, and the
linear cost model gamma*r+1 cannot represent it.

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

    # The explanation turned out NOT to be a per-token dip (cost per token is
    # monotone). It is a STEP in absolute cost: the largest single jump between
    # consecutive widths, after which cost is nearly flat. A gamma whose verify
    # width sits just past the step pays for the step without amortising it.
    deltas = [(rows[i]["width"], rows[i]["ms"] - rows[i - 1]["ms"])
              for i in range(1, len(rows))]
    step_width, step_ms = max(deltas, key=lambda d: d[1])
    after = [r["ms"] for r in rows if r["width"] >= step_width]
    plateau_span = max(after) - min(after)
    before = [d for w, d in deltas if w < step_width]
    per_token_before = sum(before) / len(before) if before else 0.0
    widths_after = [r["width"] for r in rows if r["width"] >= step_width]

    print(f"\nlargest jump: +{step_ms:.1f} ms into width {step_width} "
          f"(gamma={step_width - 1})")
    print(f"before it:    ~{per_token_before:.1f} ms per extra token")
    print(f"after it:     widths {widths_after[0]}-{widths_after[-1]} span only "
          f"{plateau_span:.1f} ms in total")

    record = {
        "widths": rows,
        "step_width": step_width,
        "step_ms": step_ms,
        "plateau_span_ms": plateau_span,
        "ms_per_extra_token_before_step": per_token_before,
        "pessimal_gamma": step_width - 1,
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
    is_step = plateau_span < step_ms
    print("VERDICT:",
          f"verify cost is a STEP at width {step_width}: gamma={step_width - 1} "
          f"pays the full step while verifying only {step_width} tokens, so it is "
          f"the pessimal gamma. Use gamma <= {step_width - 2} or a much larger "
          "gamma that amortises the step."
          if is_step else
          "no clear step -- cost grows smoothly with width")
    return 0


if __name__ == "__main__":
    sys.exit(main())
