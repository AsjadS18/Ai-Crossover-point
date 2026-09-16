"""Record a real generation trace for /replay.

One prompt that naturally mixes prose and code, so the recorded token stream
shows the project's thesis in a few seconds: explanation text accepts
patchily, while the code block runs long stretches of accepted draft tokens.

The output file uses the SAME event schema as /stream, which is what lets the
UI replay it with no model loaded -- and means a demo cannot fail on a
misbehaving GPU, as long as you say it is a recording.

    python scripts/mixed_trace.py
    python scripts/mixed_trace.py --mode spec --gamma 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compat
from src.adaptive import AdaptiveGamma
from src.baseline import GraphedGreedyDecoder
from src.models import DRAFT_PATH, TARGET_PATH, chat_ids, load_pair
from src.specdec import GraphedSpecDecoder

OUT_PATH = "results/mixed_trace.json"
PROMPT = (
    "In two sentences, explain what a hash map is and why lookups are fast. "
    "Then write a short Python class HashMap with put and get methods using "
    "separate chaining."
)
MAX_NEW_TOKENS = 256
CACHE_LEN = 512
R_GRAPHED = 0.248


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("adaptive", "spec"),
                        default="adaptive")
    parser.add_argument("--gamma", type=int, default=2)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    tokenizer, target, draft = load_pair()
    ids = chat_ids(tokenizer, args.prompt)

    # Graphed baseline first, for the identity check and the speedup figure.
    base = GraphedGreedyDecoder(target, max_cache_len=CACHE_LEN)
    base.generate(ids, max_new_tokens=8, eos_id=None)            # capture
    b_seq, b_stats = base.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                                   eos_id=tokenizer.eos_token_id)
    b_text = tokenizer.decode(b_seq[0, ids.shape[1]:], skip_special_tokens=True)

    controller = None
    if args.mode == "adaptive":
        controller = AdaptiveGamma(r=R_GRAPHED, gmax=8, cooldown=2)
        spec = GraphedSpecDecoder(target, draft, gamma=1,
                                  max_cache_len=CACHE_LEN,
                                  capture_gammas=tuple(range(1, 9)))
    else:
        spec = GraphedSpecDecoder(target, draft, gamma=args.gamma,
                                  max_cache_len=CACHE_LEN)

    trace: list[dict[str, Any]] = []
    s_seq, s_stats = spec.generate(
        ids, max_new_tokens=MAX_NEW_TOKENS, eos_id=tokenizer.eos_token_id,
        adaptive=controller, trace=trace, tokenizer=tokenizer)
    s_text = tokenizer.decode(s_seq[0, ids.shape[1]:], skip_special_tokens=True)

    # /replay paces itself on each round's "seconds". Per-round timing is not
    # instrumented, so spread the measured total evenly -- the replay is
    # realistic in aggregate, not round-by-round.
    per_round = s_stats["seconds"] / max(1, len(trace))
    for record in trace:
        record["seconds"] = per_round

    counts: dict[str, int] = {}
    for record in trace:
        for tok in record["tokens"]:
            counts[tok["origin"]] = counts.get(tok["origin"], 0) + 1

    payload = {
        "mode": args.mode,
        "gamma": args.gamma if args.mode == "spec" else "adaptive",
        "regime": "graphed (recorded)",
        "prompt": args.prompt,
        "prompt_tokens": ids.shape[1],
        "text": s_text,
        "identical_to_graphed_baseline": s_text == b_text,
        "baseline_tok_per_s": b_stats["tok_per_s"],
        "speedup": s_stats["tok_per_s"] / b_stats["tok_per_s"],
        "origin_counts": counts,
        "stats": {k: v for k, v in s_stats.items()},
        "trace": trace,
        "target_path": TARGET_PATH,
        "draft_path": DRAFT_PATH,
        "compat": compat.describe(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if controller is not None:
        payload["controller_gamma_path"] = [row[2] for row in controller.history]

    os.makedirs("results", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()

    print(f"rounds {len(trace)}, tokens {s_stats['tokens']}, origins {counts}")
    print(f"spec {s_stats['tok_per_s']:.1f} tok/s vs graphed baseline "
          f"{b_stats['tok_per_s']:.1f} tok/s -> {payload['speedup']:.3f}x")
    print(f"identical to graphed baseline: {payload['identical_to_graphed_baseline']}")
    if controller is not None:
        print(f"controller gamma path: {payload['controller_gamma_path']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
