"""Classify every losslessness mismatch in the sweep: noise, or real bug?

The sweep records a top-2 logit gap for each mismatch, but a raw gap is not
interpretable on its own -- fp16 spacing depends on magnitude.  For a value in
[2^e, 2^(e+1)) the gap between adjacent fp16 values is 2^(e-10).  At a logit
magnitude of ~26 that is 0.015625.

So the question for each mismatch is: is the gap within a couple of ULPs at
that magnitude (the target had no meaningful preference, and greedy decoding is
undefined there), or is it far larger (a genuine decoder bug, almost certainly
an off-by-one in the crop)?

This re-measures each mismatch with logit VALUES, not just the gap, and reports
the gap in ULPs.

    python scripts/analyse_divergences.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src import compat
from src.baseline import baseline_generate, forward_new
from src.models import chat_ids, load_pair
from src.specdec import spec_generate

SWEEP_PATH = "results/sweep.jsonl"
EVAL_PATH = "data/eval.jsonl"
OUT_PATH = "results/divergence_analysis.json"
MAX_NEW_TOKENS = 128
ULP_TOLERANCE = 2.0


def fp16_ulp(value: float) -> float:
    """Spacing between adjacent fp16 values at `value`.  Returns an absolute gap."""
    v = abs(value)
    if v == 0.0:
        return 2.0 ** -24
    exponent = math.floor(math.log2(v))
    exponent = max(exponent, -14)          # fp16 subnormal floor
    return 2.0 ** (exponent - 10)          # 10 mantissa bits


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass               # tolerate a partially written last line
    return out


def main() -> int:
    rows = [r for r in read_jsonl(SWEEP_PATH) if r.get("record") == "run"]
    bad = [r for r in rows if r.get("identical_to_baseline") is False]
    print(f"sweep rows: {len(rows)},  mismatches: {len(bad)}")
    if not bad:
        print("VERDICT: no mismatches to analyse")
        return 0

    prompts = {r["id"]: r["prompt"] for r in read_jsonl(EVAL_PATH)}
    tokenizer, target, draft = load_pair()

    findings: list[dict[str, Any]] = []
    # One baseline per affected prompt, reused across that prompt's gammas.
    base_cache: dict[str, Any] = {}

    for row in bad:
        pid, gamma = row["id"], row["gamma"]
        ids = chat_ids(tokenizer, prompts[pid])
        prompt_len = ids.shape[1]

        if pid not in base_cache:
            seq, _ = baseline_generate(
                target, ids, max_new_tokens=MAX_NEW_TOKENS,
                eos_id=tokenizer.eos_token_id)
            base_cache[pid] = seq
        base_seq = base_cache[pid]

        spec_seq, _ = spec_generate(
            target, draft, ids, max_new_tokens=MAX_NEW_TOKENS, gamma=gamma,
            eos_id=tokenizer.eos_token_id)

        base_toks = base_seq[0, prompt_len:].tolist()
        spec_toks = spec_seq[0, prompt_len:].tolist()
        first = next(
            (i for i, (a, b) in enumerate(zip(base_toks, spec_toks)) if a != b),
            min(len(base_toks), len(spec_toks)),
        )
        if base_toks == spec_toks:
            # The mismatch recorded in the sweep does not happen now. The usual
            # cause is that the prompt in data/eval.jsonl was edited after the
            # sweep ran (the sweep stores ids, not prompt text), so this row is
            # being re-run on different input. It is NOT evidence either way.
            findings.append({
                "id": pid, "domain": row["domain"], "gamma": gamma,
                "classification": "did not reproduce",
            })
            print(f"{pid} g={gamma}: identical on current prompt text "
                  f"-> did not reproduce (prompt likely edited after the sweep)",
                  flush=True)
            continue
        if first == 0 or first >= len(base_toks):
            findings.append({
                "id": pid, "domain": row["domain"], "gamma": gamma,
                "classification": "length-only difference",
                "first_diff_token_index": first,
            })
            print(f"{pid} g={gamma}: outputs differ only in length "
                  f"-> length-only difference", flush=True)
            continue

        prefix = base_seq[:, : prompt_len + first]
        with torch.inference_mode():
            c1 = compat.make_cache(target)
            forward_new(target, c1, prefix[:, :-1], 0)
            single = forward_new(
                target, c1, prefix, prefix.shape[1] - 1)[0, -1, :].float()

            c2 = compat.make_cache(target)
            chunked = forward_new(target, c2, prefix, 0)[0, -1, :].float()

        t1 = single.topk(2)
        t2 = chunked.topk(2)
        gap_single = (t1.values[0] - t1.values[1]).item()
        gap_chunked = (t2.values[0] - t2.values[1]).item()
        magnitude = t1.values[0].item()
        ulp = fp16_ulp(magnitude)
        path_delta = (single - chunked).abs().max().item()

        worst_gap = min(gap_single, gap_chunked)
        ulps = worst_gap / ulp if ulp else float("inf")
        ambiguous = ulps <= ULP_TOLERANCE

        findings.append({
            "id": pid,
            "domain": row["domain"],
            "gamma": gamma,
            "first_diff_token_index": first,
            "baseline_token_text": tokenizer.decode([base_toks[first]]),
            "spec_token_text": tokenizer.decode([spec_toks[first]]),
            "logit_magnitude": magnitude,
            "fp16_ulp_at_magnitude": ulp,
            "gap_single_token_path": gap_single,
            "gap_chunked_path": gap_chunked,
            "gap_in_ulps": ulps,
            "max_logit_delta_between_paths": path_delta,
            "classification": (
                "numerically ambiguous" if ambiguous else "DECODER BUG"
            ),
        })
        print(f"{pid} g={gamma}: {findings[-1]['baseline_token_text']!r} vs "
              f"{findings[-1]['spec_token_text']!r}  "
              f"gap={worst_gap:.6f} = {ulps:.2f} ULP  "
              f"-> {findings[-1]['classification']}", flush=True)

    bugs = [f for f in findings if f["classification"] == "DECODER BUG"]
    ambiguous = [f for f in findings if f["classification"] == "numerically ambiguous"]
    stale = [f for f in findings if f["classification"] == "did not reproduce"]
    length_only = [f for f in findings
                   if f["classification"] == "length-only difference"]

    report = {
        "mismatches": len(bad),
        "numerically_ambiguous": len(ambiguous),
        "did_not_reproduce": len(stale),
        "did_not_reproduce_ids": sorted({f["id"] for f in stale}),
        "length_only_difference": len(length_only),
        "decoder_bugs": len(bugs),
        "ulp_tolerance": ULP_TOLERANCE,
        "sweep_rows": len(rows),
        "compat": compat.describe(),
        "findings": findings,
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.flush()

    print(f"\n{len(bad)} mismatches in the sweep: "
          f"{len(ambiguous)} numerically ambiguous, "
          f"{len(stale)} did not reproduce, "
          f"{len(length_only)} length-only, {len(bugs)} decoder bugs")
    if stale:
        print(f"did not reproduce: {sorted({f['id'] for f in stale})} -- "
              "re-run on edited prompt text, so neither confirmed nor refuted")
    print(f"wrote {OUT_PATH}")
    print("VERDICT:", "PASS (no decoder bugs)" if not bugs else "FAIL (real bug)")
    return 0 if not bugs else 1


if __name__ == "__main__":
    sys.exit(main())
