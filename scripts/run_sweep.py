"""The experiment: acceptance and speedup for every prompt at every gamma.

For each prompt in data/eval.jsonl this runs the baseline once, then the
speculative decoder at each gamma, and records one JSONL row per run with
speedup measured against that same prompt's own baseline.

Results append to results/sweep.jsonl, flushed per record, so a crash costs one
row.  Re-running skips work already present, so the sweep can be stopped and
resumed freely.

    python scripts/run_sweep.py                 full sweep
    python scripts/run_sweep.py --limit 6       pilot on the first 6 prompts
    python scripts/run_sweep.py --gammas 1,3,5  narrower gamma set
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compat
from src.baseline import baseline_generate, forward_new
from src.models import DRAFT_PATH, TARGET_PATH, chat_ids, load_pair
from src.specdec import spec_generate

EVAL_PATH = "data/eval.jsonl"
OUT_PATH = "results/sweep.jsonl"
GAMMAS = (1, 2, 3, 5, 8, 12)
MAX_NEW_TOKENS = 128


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done_keys(path: str) -> set[tuple[str, str, int]]:
    """Existing (id, mode, gamma) rows, so a resumed run skips them."""
    keys = set()
    for rec in read_jsonl(path):
        if rec.get("record") == "metadata":
            continue
        keys.add((rec.get("id"), rec.get("mode"), rec.get("gamma", -1)))
    return keys


def append(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def diagnose_divergence(
    target: Any, tokenizer: Any, base_seq: Any, spec_seq: Any, prompt_len: int
) -> dict[str, Any]:
    """Explain one losslessness mismatch at the token level.

    Returns the first divergent token index, both candidate tokens, and the
    target's own top-2 logit gap at that position on the single-token path.
    A gap of 0.0 means the target had an exact fp16 tie, so greedy decoding is
    undefined there rather than the decoder being wrong.
    """
    import torch

    base_toks = base_seq[0, prompt_len:].tolist()
    spec_toks = spec_seq[0, prompt_len:].tolist()
    first = next(
        (i for i, (a, b) in enumerate(zip(base_toks, spec_toks)) if a != b),
        min(len(base_toks), len(spec_toks)),
    )

    info: dict[str, Any] = {
        "first_diff_token_index": first,
        "baseline_token": base_toks[first] if first < len(base_toks) else None,
        "spec_token": spec_toks[first] if first < len(spec_toks) else None,
        "baseline_token_text": (
            tokenizer.decode([base_toks[first]]) if first < len(base_toks) else None
        ),
        "spec_token_text": (
            tokenizer.decode([spec_toks[first]]) if first < len(spec_toks) else None
        ),
        "baseline_tokens": len(base_toks),
        "spec_tokens": len(spec_toks),
    }

    if first == 0 or first >= len(base_toks):
        return info

    prefix = base_seq[:, : prompt_len + first]
    with torch.inference_mode():
        cache = compat.make_cache(target)
        forward_new(target, cache, prefix[:, :-1], 0)
        logits = forward_new(target, cache, prefix, prefix.shape[1] - 1)
        top = logits[0, -1, :].float().topk(2)
    info["top2_gap"] = (top.values[0] - top.values[1]).item()
    info["top2_tokens"] = top.indices.tolist()
    info["exact_fp16_tie"] = info["top2_gap"] == 0.0
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N prompts, for a pilot run")
    parser.add_argument("--gammas", default=None,
                        help="comma-separated gamma values")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    gammas = (
        tuple(int(g) for g in args.gammas.split(","))
        if args.gammas else GAMMAS
    )
    prompts = read_jsonl(EVAL_PATH)
    if args.limit:
        prompts = prompts[: args.limit]

    os.makedirs("results", exist_ok=True)
    already = done_keys(args.out)
    prior_baseline_tps = {
        r["id"]: r["tok_per_s"]
        for r in read_jsonl(args.out)
        if r.get("mode") == "baseline"
    }
    if already:
        print(f"resuming: {len(already)} rows already in {args.out}")

    tokenizer, target, draft = load_pair()

    if not already:
        append(args.out, {
            "record": "metadata",
            "gammas": list(gammas),
            "max_new_tokens": args.max_new_tokens,
            "eval_path": EVAL_PATH,
            "prompts": len(prompts),
            "target_path": TARGET_PATH,
            "draft_path": DRAFT_PATH,
            "compat": compat.describe(),
            "utc": datetime.now(timezone.utc).isoformat(),
        })

    # The first generation in a process runs cold: in the pilot, prompt 1's
    # baseline came in at 17.1 tok/s against ~21.1 for every prompt after it,
    # which inflated that prompt's speedups by about 20%. Burn one throwaway
    # generation through both paths before any timing is recorded.
    warm_ids = chat_ids(tokenizer, "Write a short Python function.")
    baseline_generate(target, warm_ids, max_new_tokens=16,
                      eos_id=tokenizer.eos_token_id)
    spec_generate(target, draft, warm_ids, max_new_tokens=16, gamma=4,
                  eos_id=tokenizer.eos_token_id)
    print("warmup done", flush=True)

    total = len(prompts) * (1 + len(gammas))
    done = 0
    mismatches = 0
    ties = 0
    started = time.perf_counter()

    for rec in prompts:
        pid, domain = rec["id"], rec["domain"]
        ids = chat_ids(tokenizer, rec["prompt"])
        prompt_len = ids.shape[1]

        # Baseline first: every speedup on this prompt is relative to it.
        base_text = None
        base_seq_full = None
        base_stats: dict[str, Any] | None = None

        # On a resumed run the baseline row may already exist while its TEXT is
        # not in memory. Without regenerating it, identical_to_baseline would
        # default to True and the losslessness check would silently stop
        # checking for that prompt. Regenerate the text, but do not re-record
        # the row: the recorded timing stays the one from the original run.
        if (pid, "baseline", 0) in already and any(
            (pid, "spec", g) not in already for g in gammas
        ):
            seq, _ = baseline_generate(
                target, ids, max_new_tokens=args.max_new_tokens,
                eos_id=tokenizer.eos_token_id)
            base_seq_full = seq
            base_text = tokenizer.decode(
                seq[0, prompt_len:], skip_special_tokens=True)
            if pid in prior_baseline_tps:
                base_stats = {"tok_per_s": prior_baseline_tps[pid]}

        if (pid, "baseline", 0) not in already:
            seq, base_stats = baseline_generate(
                target, ids, max_new_tokens=args.max_new_tokens,
                eos_id=tokenizer.eos_token_id)
            base_seq_full = seq
            base_text = tokenizer.decode(
                seq[0, prompt_len:], skip_special_tokens=True)
            append(args.out, {
                "record": "run", "id": pid, "domain": domain,
                "mode": "baseline", "gamma": 0,
                "tokens": base_stats["tokens"],
                "seconds": base_stats["seconds"],
                "tok_per_s": base_stats["tok_per_s"],
                "speedup": 1.0,
            })
        done += 1

        for gamma in gammas:
            if (pid, "spec", gamma) in already:
                done += 1
                continue
            seq, stats = spec_generate(
                target, draft, ids, max_new_tokens=args.max_new_tokens,
                gamma=gamma, eos_id=tokenizer.eos_token_id)
            spec_text = tokenizer.decode(
                seq[0, prompt_len:], skip_special_tokens=True)

            # Losslessness is re-checked here, not assumed. A sweep that
            # silently diverged would produce meaningless speedups.
            identical = (base_text is None) or (spec_text == base_text)
            divergence: dict[str, Any] | None = None
            if not identical:
                mismatches += 1
                divergence = diagnose_divergence(
                    target, tokenizer, base_seq_full, seq, prompt_len)
                if divergence.get("exact_fp16_tie"):
                    ties += 1
                print(f"    MISMATCH {pid} gamma={gamma}: "
                      f"token {divergence['first_diff_token_index']} "
                      f"{divergence['baseline_token_text']!r} vs "
                      f"{divergence['spec_token_text']!r}, "
                      f"top2_gap={divergence.get('top2_gap')}, "
                      f"exact_tie={divergence.get('exact_fp16_tie')}", flush=True)

            append(args.out, {
                "record": "run", "id": pid, "domain": domain,
                "mode": "spec", "gamma": gamma,
                "tokens": stats["tokens"],
                "seconds": stats["seconds"],
                "tok_per_s": stats["tok_per_s"],
                "rounds": stats["rounds"],
                "proposed": stats["proposed"],
                "accepted": stats["accepted"],
                "acceptance_rate": stats["acceptance_rate"],
                "tokens_per_round": stats["tokens_per_round"],
                "speedup": (
                    stats["tok_per_s"] / base_stats["tok_per_s"]
                    if base_stats and base_stats["tok_per_s"] > 0 else None
                ),
                "identical_to_baseline": identical,
                "divergence": divergence,
            })
            done += 1

        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"{done}/{total} runs  {pid} ({domain})  "
              f"{mismatches} mismatches ({ties} fp16 ties)  "
              f"eta {remaining / 60:.0f} min", flush=True)

    print(f"\nwrote {args.out}")
    print(f"RESULT: {done} runs, {mismatches} losslessness mismatches, "
          f"{ties} of them exact fp16 ties")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
