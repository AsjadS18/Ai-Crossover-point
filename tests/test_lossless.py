"""THE critical test: speculative output must be byte-identical to baseline.

This is the project's headline claim, and it is proved by comparison, not by
argument.  Gate is 100%.  Not 99.8%.  A single mismatch is a real bug, almost
always an off-by-one in the cache crop.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import compat
from src.baseline import baseline_generate
from src.models import DRAFT_PATH, TARGET_PATH, chat_ids, load_pair
from src.specdec import spec_generate

EVAL_PATH = "data/eval.jsonl"
GAMMAS = (1, 3, 5, 8)
MAX_NEW_TOKENS = 96
FAILURES_PATH = "results/lossless_failures.json"
REPORT_PATH = "results/lossless_report.json"


def load_eval(path: str = EVAL_PATH) -> list[dict[str, Any]]:
    """Read the prompt set.  Returns one dict per line, order preserved."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def first_difference(a: str, b: str) -> int:
    """Index of the first differing character, or the length of the shorter."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def run_sweep(verbose: bool = True) -> dict[str, Any]:
    """Compare baseline against every gamma for every prompt.

    Returns a report dict and writes results/lossless_report.json, plus
    results/lossless_failures.json if anything mismatched.
    """
    prompts = load_eval()
    tokenizer, target, draft = load_pair()

    checked = 0
    failures: list[dict[str, Any]] = []
    per_gamma: dict[int, dict[str, float]] = {
        g: {"checked": 0, "identical": 0, "acceptance_sum": 0.0} for g in GAMMAS
    }
    total = len(prompts) * len(GAMMAS)

    for record in prompts:
        ids = chat_ids(tokenizer, record["prompt"])
        base_seq, _ = baseline_generate(
            target, ids, max_new_tokens=MAX_NEW_TOKENS,
            eos_id=tokenizer.eos_token_id)
        base_text = tokenizer.decode(
            base_seq[0, ids.shape[1]:], skip_special_tokens=True)

        for gamma in GAMMAS:
            spec_seq, stats = spec_generate(
                target, draft, ids, max_new_tokens=MAX_NEW_TOKENS,
                gamma=gamma, eos_id=tokenizer.eos_token_id)
            spec_text = tokenizer.decode(
                spec_seq[0, ids.shape[1]:], skip_special_tokens=True)

            checked += 1
            per_gamma[gamma]["checked"] += 1
            per_gamma[gamma]["acceptance_sum"] += stats["acceptance_rate"]

            if spec_text == base_text:
                per_gamma[gamma]["identical"] += 1
            else:
                failures.append({
                    "id": record.get("id"),
                    "domain": record.get("domain"),
                    "gamma": gamma,
                    "first_diff_index": first_difference(base_text, spec_text),
                    "baseline": base_text,
                    "speculative": spec_text,
                })

            if verbose:
                print(f"  {checked}/{total} checked, {len(failures)} mismatches "
                      f"({record.get('id')} gamma={gamma} "
                      f"acc={stats['acceptance_rate']:.3f})", flush=True)

    identical = checked - len(failures)
    print(f"RESULT: {identical}/{checked} identical")

    os.makedirs("results", exist_ok=True)
    if failures:
        with open(FAILURES_PATH, "w", encoding="utf-8") as fh:
            json.dump(failures, fh, indent=2)
            fh.flush()
        print(f"wrote {FAILURES_PATH}")

    report = {
        "checked": checked,
        "identical": identical,
        "mismatches": len(failures),
        "prompts": len(prompts),
        "gammas": list(GAMMAS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "eval_path": EVAL_PATH,
        "per_gamma": {
            str(g): {
                "checked": v["checked"],
                "identical": v["identical"],
                "mean_acceptance": (
                    v["acceptance_sum"] / v["checked"] if v["checked"] else 0.0
                ),
            }
            for g, v in per_gamma.items()
        },
        "target_path": TARGET_PATH,
        "draft_path": DRAFT_PATH,
        "compat": compat.describe(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.flush()
    print(f"wrote {REPORT_PATH}")
    return report


@pytest.mark.skipif(
    not os.path.exists(EVAL_PATH), reason=f"{EVAL_PATH} does not exist yet"
)
def test_speculative_output_is_identical_to_baseline():
    report = run_sweep()
    assert report["mismatches"] == 0, (
        f"{report['mismatches']} of {report['checked']} outputs differed; "
        f"see {FAILURES_PATH}"
    )


if __name__ == "__main__":
    rep = run_sweep()
    print("VERDICT:", "PASS" if rep["mismatches"] == 0 else "FAIL")
    sys.exit(0 if rep["mismatches"] == 0 else 1)
