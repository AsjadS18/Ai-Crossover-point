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
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch

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


def gpu_state() -> str:
    """Temperature and SM clock, for the log.  Empty string if unavailable."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out
    except Exception:
        return ""


def burn_in(decoder: Any, ids: Any, eos_id: int | None, seconds: float) -> None:
    """Run throwaway generations until the GPU reaches thermal steady state.

    The card really does move: 44 C / 1920 MHz / 53 W cold, settling to
    72 C / 1837 MHz / 170 W (the power cap) after ~150 s. Starting every pass
    from the same operating point removes that as a variable, which matters
    because this sweep is gamma-outer: baselines are timed in pass 1 while their
    speculative runs happen later.

    HONESTY NOTE: burn-in was added because a baseline pass appeared to decay
    40.11 -> 35.12 tok/s, which I attributed to this warm-up. That was WRONG.
    With burn-in the ratio moved only 0.88 -> 0.90, and a within-domain check
    showed throughput is flat (ratios 0.997-1.005) while full-length runs held
    40.77 -> 40.74 across the pass. The apparent decay was domain composition,
    not temperature -- see drift_report. Burn-in is kept because controlling the
    operating point is still correct practice, not because it fixed that.
    """
    if seconds <= 0:
        return
    print(f"burn-in: {seconds:.0f}s to reach steady state "
          f"(before: {gpu_state()})", flush=True)
    started = time.perf_counter()
    runs = 0
    while time.perf_counter() - started < seconds:
        decoder.generate(ids, max_new_tokens=128, eos_id=eos_id)
        runs += 1
    print(f"burn-in done: {runs} throwaway generations, "
          f"{time.perf_counter() - started:.0f}s (after: {gpu_state()})",
          flush=True)


def drift_report(
    label: str,
    values: list[float],
    token_counts: list[int] | None = None,
    full_length: int | None = None,
    window: int = 15,
) -> str:
    """Compare throughput early and late in a pass, on comparable runs only.

    A pass whose throughput decays as it runs was measuring resource exhaustion,
    not the decoder: the contaminated sweep decayed 47.9 -> 4.4 tok/s WITHIN one
    gamma pass and every speedup in it was invalid.

    BUT a naive first15-vs-last15 comparison is confounded, because prompts run
    in domain order. tok_per_s = generated / (prefill + decode), so a prompt that
    stops early at EOS amortises its fixed prefill over fewer tokens and scores
    lower for reasons unrelated to time. translation averages 23.7 generated
    tokens against 89-128 elsewhere and therefore ~36 tok/s against ~40, which
    made a perfectly healthy pass look like a 10% decline and cost a needless
    restart.

    So the comparison is restricted to runs that generated the FULL token
    budget: identical work per run, leaving position as the only variable.
    """
    if token_counts is not None and full_length is not None:
        values = [v for v, n in zip(values, token_counts) if n >= full_length]
        label = f"{label} [full-length runs only]"

    if len(values) < 2 * window:
        return f"{label}: {len(values)} comparable runs, too few to assess drift"
    first = statistics.mean(values[:window])
    last = statistics.mean(values[-window:])
    ratio = last / first if first else 0.0
    verdict = "OK" if ratio >= 0.9 else "DEGRADED -- timings are NOT valid"
    return (f"{label}: first{window} {first:.2f} tok/s, last{window} "
            f"{last:.2f} tok/s, ratio {ratio:.2f}  {verdict}")


def run_graphed_sweep(args: Any) -> int:
    """Sweep in the graphed regime: graphed baseline vs graphed speculation.

    Losslessness is compared WITHIN the regime. Comparing graphed spec to the
    eager DynamicCache baseline would report false mismatches caused by
    StaticCache/DynamicCache numerical drift, which is characterised in
    tests/test_compat.py.

    Structured gamma-outer so only ONE GraphedSpecDecoder is alive at a time:
    each holds two static caches plus captured graph memory pools, and six at
    once risks running the 12 GB card out of room.
    """
    from src.baseline import GraphedGreedyDecoder
    from src.specdec import GraphedSpecDecoder

    gammas = (
        tuple(int(g) for g in args.gammas.split(","))
        if args.gammas else GAMMAS
    )
    prompts = read_jsonl(EVAL_PATH)
    if args.limit:
        prompts = prompts[: args.limit]

    os.makedirs("results", exist_ok=True)
    already = done_keys(args.out)
    if already:
        print(f"resuming: {len(already)} rows already in {args.out}")

    tokenizer, target, draft = load_pair()

    if not already:
        append(args.out, {
            "record": "metadata",
            "regime": "graphed",
            "gammas": list(gammas),
            "max_new_tokens": args.max_new_tokens,
            "max_cache_len": args.max_cache_len,
            "eval_path": EVAL_PATH,
            "prompts": len(prompts),
            "target_path": TARGET_PATH,
            "draft_path": DRAFT_PATH,
            "compat": compat.describe(),
            "utc": datetime.now(timezone.utc).isoformat(),
        })

    # A static cache costs the same attention work at every position, so an
    # oversized one is pure waste: 512 slots doubled the cost of a run that
    # never needs more than prompt + max_new_tokens + gamma headroom.
    longest = max(chat_ids(tokenizer, r["prompt"]).shape[1] for r in prompts)
    needed = longest + args.max_new_tokens + max(gammas) + 2
    cache_len = min(args.max_cache_len, max(128, needed))
    print(f"longest prompt {longest} tokens, cache_len {cache_len} "
          f"(requested {args.max_cache_len})", flush=True)

    base = GraphedGreedyDecoder(target, max_cache_len=cache_len)
    base.generate(chat_ids(tokenizer, "warm up the capture"),
                  max_new_tokens=8, eos_id=None)
    print("baseline graph captured", flush=True)

    # Reach thermal steady state BEFORE any timing, so the baseline pass is not
    # measured on a cold card while the spec passes run hot.
    burn_in(base, chat_ids(tokenizer, "warm the card to steady state"),
            None, args.burn_in)

    # Pass 1: every baseline, kept in memory for the losslessness comparison.
    base_text: dict[str, str] = {}
    base_tps: dict[str, float] = {}
    base_tokens: dict[str, int] = {}
    for i, rec in enumerate(prompts, start=1):
        pid = rec["id"]
        ids = chat_ids(tokenizer, rec["prompt"])
        seq, stats = base.generate(
            ids, max_new_tokens=args.max_new_tokens,
            eos_id=tokenizer.eos_token_id)
        base_text[pid] = tokenizer.decode(
            seq[0, ids.shape[1]:], skip_special_tokens=True)
        base_tps[pid] = stats["tok_per_s"]
        base_tokens[pid] = stats["tokens"]
        if (pid, "baseline", 0) not in already:
            append(args.out, {
                "record": "run", "id": pid, "domain": rec["domain"],
                "mode": "baseline", "gamma": 0,
                "tokens": stats["tokens"], "seconds": stats["seconds"],
                "tok_per_s": stats["tok_per_s"], "speedup": 1.0,
            })
        if i % 25 == 0 or i == len(prompts):
            print(f"baselines {i}/{len(prompts)}", flush=True)
    print(drift_report("baseline",
                       [base_tps[r["id"]] for r in prompts],
                       [base_tokens[r["id"]] for r in prompts],
                       args.max_new_tokens),
          flush=True)

    # Pass 2: one gamma at a time.
    mismatches = 0
    for gamma in gammas:
        spec = GraphedSpecDecoder(target, draft, gamma=gamma,
                                  max_cache_len=cache_len)
        pass_tps: list[float] = []
        pass_tokens: list[int] = []
        print(f"gamma={gamma} graphs captured", flush=True)
        for i, rec in enumerate(prompts, start=1):
            pid = rec["id"]
            if (pid, "spec", gamma) in already:
                continue
            ids = chat_ids(tokenizer, rec["prompt"])
            seq, stats = spec.generate(
                ids, max_new_tokens=args.max_new_tokens,
                eos_id=tokenizer.eos_token_id)
            text = tokenizer.decode(seq[0, ids.shape[1]:],
                                    skip_special_tokens=True)
            identical = text == base_text[pid]
            if not identical:
                mismatches += 1
            append(args.out, {
                "record": "run", "id": pid, "domain": rec["domain"],
                "mode": "spec", "gamma": gamma,
                "tokens": stats["tokens"], "seconds": stats["seconds"],
                "tok_per_s": stats["tok_per_s"], "rounds": stats["rounds"],
                "proposed": stats["proposed"], "accepted": stats["accepted"],
                "acceptance_rate": stats["acceptance_rate"],
                "tokens_per_round": stats["tokens_per_round"],
                "speedup": (
                    stats["tok_per_s"] / base_tps[pid] if base_tps[pid] else None
                ),
                "identical_to_baseline": identical,
            })
            pass_tps.append(stats["tok_per_s"])
            pass_tokens.append(stats["tokens"])
            if i % 25 == 0 or i == len(prompts):
                print(f"gamma={gamma} {i}/{len(prompts)}  "
                      f"{mismatches} mismatches", flush=True)
        print(drift_report(f"gamma={gamma}", pass_tps, pass_tokens,
                           args.max_new_tokens), flush=True)
        del spec
        torch.cuda.empty_cache()

    print(f"\nwrote {args.out}")
    print(f"RESULT: graphed sweep complete, {mismatches} losslessness mismatches")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N prompts, for a pilot run")
    parser.add_argument("--gammas", default=None,
                        help="comma-separated gamma values")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--graphed", action="store_true",
                        help="run in the CUDA-graph regime (graphed baseline "
                             "vs graphed speculation)")
    parser.add_argument("--max-cache-len", type=int, default=512,
                        help="static cache size, graphed mode only")
    parser.add_argument("--burn-in", type=float, default=150.0,
                        help="seconds of throwaway generation before timing, to "
                             "reach thermal steady state (graphed mode only)")
    args = parser.parse_args()

    if args.graphed:
        if args.out == OUT_PATH:
            args.out = "results/sweep_graphed.jsonl"
        return run_graphed_sweep(args)

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
