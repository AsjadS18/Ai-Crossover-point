"""Does the adaptive controller beat a fixed gamma?

The sweep says it should: the graphed regime has FOUR different optimal gammas
across six domains (8 for math and structured, 5 for code, 1 for reasoning and
prose, 0 for translation). Any single fixed gamma is therefore wrong for most of
the set, and a controller that reads acceptance and adjusts should win on the
mixed workload without being told which domain it is in.

The honest comparison is against the best fixed gamma chosen WITH HINDSIGHT over
the whole run, not against a badly chosen one. If the controller cannot beat
that, it is not worth its complexity and this script should say so.

Every run is also checked for byte-identity against its own graphed baseline:
the controller changes gamma mid-generation, so losslessness has to hold across
gamma switches, not just within one.

    python scripts/validate_controller.py
    python scripts/validate_controller.py --limit 60 --gammas 1,2,8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src import compat
from src.adaptive import AdaptiveGamma
from src.baseline import GraphedGreedyDecoder
from src.models import DRAFT_PATH, TARGET_PATH, chat_ids, load_pair
from src.specdec import GraphedSpecDecoder

EVAL_PATH = "data/eval.jsonl"
OUT_PATH = "results/controller_comparison.json"
FIXED_GAMMAS = (1, 2, 3, 5, 8)
MAX_NEW_TOKENS = 128
R_GRAPHED = 0.248
BURN_IN = 150.0
DOMAINS = ("structured", "code", "reasoning", "prose", "math", "translation")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_verify_cost(path: str = "results/verify_width.json") -> dict[int, float] | None:
    """Measured verify cost per width, normalised to one single-token forward.

    Produced by scripts/measure_verify_width.py. Returns None if absent, in
    which case the controller falls back to the linear model.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    rows = table.get("widths", [])
    single = next((r["ms"] for r in rows if r["width"] == 1), None)
    if not single:
        return None
    return {int(r["width"]): r["ms"] / single for r in rows}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gammas", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--burn-in", type=float, default=BURN_IN)
    parser.add_argument("--gmax", type=int, default=8)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    fixed = (tuple(int(g) for g in args.gammas.split(","))
             if args.gammas else FIXED_GAMMAS)
    prompts = read_jsonl(EVAL_PATH)
    if args.limit:
        prompts = prompts[: args.limit]

    tokenizer, target, draft = load_pair()
    longest = max(chat_ids(tokenizer, r["prompt"]).shape[1] for r in prompts)
    cache_len = longest + args.max_new_tokens + max(fixed + (args.gmax,)) + 2
    print(f"{len(prompts)} prompts, cache_len {cache_len}, "
          f"fixed gammas {fixed}, controller gmax {args.gmax}", flush=True)

    base = GraphedGreedyDecoder(target, max_cache_len=cache_len)

    # Thermal steady state before any timing, as in run_sweep.
    warm = chat_ids(tokenizer, "warm the card to steady state")
    base.generate(warm, max_new_tokens=8, eos_id=None)
    if args.burn_in > 0:
        import time as _time

        started = _time.perf_counter()
        runs = 0
        while _time.perf_counter() - started < args.burn_in:
            base.generate(warm, max_new_tokens=128, eos_id=None)
            runs += 1
        print(f"burn-in: {runs} runs, {_time.perf_counter() - started:.0f}s",
              flush=True)

    # Baselines, for paired per-prompt speedups.
    base_text: dict[str, str] = {}
    base_tps: dict[str, float] = {}
    for i, rec in enumerate(prompts, start=1):
        ids = chat_ids(tokenizer, rec["prompt"])
        seq, stats = base.generate(ids, max_new_tokens=args.max_new_tokens,
                                   eos_id=tokenizer.eos_token_id)
        base_text[rec["id"]] = tokenizer.decode(
            seq[0, ids.shape[1]:], skip_special_tokens=True)
        base_tps[rec["id"]] = stats["tok_per_s"]
        if i % 50 == 0 or i == len(prompts):
            print(f"  baselines {i}/{len(prompts)}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []

    def record(label: str, rows: list[dict[str, Any]]) -> None:
        results[label] = {
            "mean_speedup": mean([r["speedup"] for r in rows]),
            "mean_tok_per_s": mean([r["tok_per_s"] for r in rows]),
            "identical": sum(1 for r in rows if r["identical"]),
            "runs": len(rows),
            "by_domain": {
                d: mean([r["speedup"] for r in rows if r["domain"] == d])
                for d in DOMAINS
                if any(r["domain"] == d for r in rows)
            },
        }

    # Fixed gammas.
    for gamma in fixed:
        spec = GraphedSpecDecoder(target, draft, gamma=gamma,
                                  max_cache_len=cache_len)
        rows = []
        for rec in prompts:
            ids = chat_ids(tokenizer, rec["prompt"])
            seq, stats = spec.generate(ids, max_new_tokens=args.max_new_tokens,
                                       eos_id=tokenizer.eos_token_id)
            text = tokenizer.decode(seq[0, ids.shape[1]:],
                                    skip_special_tokens=True)
            same = text == base_text[rec["id"]]
            if not same:
                mismatches.append({"config": f"fixed_{gamma}", "id": rec["id"]})
            rows.append({
                "id": rec["id"], "domain": rec["domain"],
                "tok_per_s": stats["tok_per_s"],
                "speedup": stats["tok_per_s"] / base_tps[rec["id"]],
                "identical": same,
            })
        record(f"fixed_gamma_{gamma}", rows)
        print(f"fixed gamma={gamma}: "
              f"{results[f'fixed_gamma_{gamma}']['mean_speedup']:.4f}x mean, "
              f"{results[f'fixed_gamma_{gamma}']['identical']}/{len(rows)} "
              f"identical", flush=True)
        del spec
        torch.cuda.empty_cache()

    # Two controllers: the textbook linear cost model, and the same controller
    # given the MEASURED verify curve. measure_verify_width.py showed the linear
    # model is wrong here (verify cost is a step function), and the pilot showed
    # a linear controller parks in gamma 2-4, the worst region. Running both
    # separates "the controller idea does not work" from "the cost model was
    # wrong", which are very different conclusions.
    verify_table = load_verify_cost()
    if verify_table:
        print(f"measured verify cost loaded for widths "
              f"{min(verify_table)}-{max(verify_table)}", flush=True)
    else:
        print("no results/verify_width.json; measured controller SKIPPED",
              flush=True)

    controllers: list[tuple[str, AdaptiveGamma]] = [
        ("adaptive_linear_cost",
         AdaptiveGamma(r=R_GRAPHED, gmax=args.gmax, cooldown=2)),
    ]
    if verify_table:
        controllers.append((
            "adaptive_measured_cost",
            AdaptiveGamma(r=R_GRAPHED, gmax=args.gmax, cooldown=2,
                          verify_cost=verify_table),
        ))

    for label, controller in controllers:
        spec = GraphedSpecDecoder(target, draft, gamma=controller.gamma,
                                  max_cache_len=cache_len,
                                  capture_gammas=tuple(range(1, args.gmax + 1)))
        rows = []
        gamma_used: list[int] = []
        for rec in prompts:
            ids = chat_ids(tokenizer, rec["prompt"])
            before = len(controller.history)
            seq, stats = spec.generate(ids, max_new_tokens=args.max_new_tokens,
                                       eos_id=tokenizer.eos_token_id,
                                       adaptive=controller)
            text = tokenizer.decode(seq[0, ids.shape[1]:],
                                    skip_special_tokens=True)
            same = text == base_text[rec["id"]]
            if not same:
                mismatches.append({"config": label, "id": rec["id"]})
            gamma_used += [row[2] for row in controller.history[before:]]
            rows.append({
                "id": rec["id"], "domain": rec["domain"],
                "tok_per_s": stats["tok_per_s"],
                "speedup": stats["tok_per_s"] / base_tps[rec["id"]],
                "identical": same,
            })
        record(label, rows)

        counts: dict[int, int] = {}
        for g in gamma_used:
            counts[g] = counts.get(g, 0) + 1
        results[label]["gamma_histogram"] = dict(sorted(counts.items()))
        results[label]["final_alpha"] = controller.alpha
        results[label]["rounds"] = len(gamma_used)
        print(f"{label}: {results[label]['mean_speedup']:.4f}x mean, "
              f"gamma histogram {results[label]['gamma_histogram']}", flush=True)
        del spec
        torch.cuda.empty_cache()

    # Verdict.
    best_fixed = max(
        ((k, v["mean_speedup"]) for k, v in results.items()
         if k.startswith("fixed_")),
        key=lambda kv: kv[1],
    )
    best_adaptive = max(
        ((k, v["mean_speedup"]) for k, v in results.items()
         if k.startswith("adaptive")),
        key=lambda kv: kv[1],
    )
    adaptive_mean = best_adaptive[1]
    verdict = (
        f"{best_adaptive[0]} WINS ({adaptive_mean:.4f}x vs best fixed "
        f"{best_fixed[0]} {best_fixed[1]:.4f}x)"
        if adaptive_mean > best_fixed[1] else
        f"every controller LOSES to {best_fixed[0]} at {best_fixed[1]:.4f}x "
        f"(best controller {best_adaptive[0]} {adaptive_mean:.4f}x)"
    )

    print("\n=== mean speedup vs graphed baseline ===")
    for label in sorted(results, key=lambda k: -results[k]["mean_speedup"]):
        r = results[label]
        print(f"  {label:18s} {r['mean_speedup']:.4f}x  "
              f"{r['mean_tok_per_s']:6.2f} tok/s  "
              f"{r['identical']}/{r['runs']} identical")
    print(f"\nbest fixed: {best_fixed[0]} at {best_fixed[1]:.4f}x")
    print(f"adaptive:   {adaptive_mean:.4f}x")
    print(f"VERDICT: {verdict}")

    payload = {
        "verdict": verdict,
        "adaptive_mean_speedup": adaptive_mean,
        "best_fixed": {"config": best_fixed[0], "mean_speedup": best_fixed[1]},
        "best_adaptive": {"config": best_adaptive[0],
                          "mean_speedup": adaptive_mean},
        "configs": results,
        "losslessness_mismatches": mismatches,
        "prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "r_used_by_controller": R_GRAPHED,
        "controller": {"gmax": args.gmax, "cooldown": 2, "beta": 0.85},
        "note": (
            "The controller uses the LINEAR cost model gamma*r+1, which "
            "measure_verify_width.py showed is wrong: verify cost is a step "
            "function, flat from width 4 onward. Expect the controller to "
            "under-use large gamma."
        ),
        "target_path": TARGET_PATH,
        "draft_path": DRAFT_PATH,
        "compat": compat.describe(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
