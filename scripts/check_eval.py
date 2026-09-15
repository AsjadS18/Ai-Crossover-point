"""Validate data/eval.jsonl before it is used for any measurement.

A bad prompt set silently invalidates every acceptance number downstream, so
this runs as a gate.  It checks schema, counts, duplicates and -- the one that
actually matters -- whether the set is machine-uniform.

    python scripts/check_eval.py
    python scripts/check_eval.py --min-handwritten 50
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any

EVAL_PATH = "data/eval.jsonl"
REQUIRED_KEYS = {"id", "domain", "prompt", "source"}
EXPECTED_DOMAINS = {
    "structured", "code", "reasoning", "prose", "math", "translation",
}
MIN_WORDS = 4
MAX_WORDS = 45
INSTRUCTION_MARKERS = (
    "register for",
    "hard requirement",
    "one per line",
    "do not answer",
    "no numbering",
    "every prompt must",
    "evaluation set",
)


def caps_ratio(text: str) -> float:
    """Share of alphabetic characters that are upper case, 0.0 to 1.0."""
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def is_title_case(text: str) -> bool:
    """True if nearly every word starts with a capital, as in Title Case."""
    words = [w for w in text.split() if w[:1].isalpha()]
    if len(words) < 4:
        return False
    return sum(1 for w in words if w[0].isupper()) / len(words) > 0.8


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{n} is not valid JSON: {exc}") from exc
    return out


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def check(records: list[dict[str, Any]], min_handwritten: int) -> list[str]:
    """Return a list of failure strings.  Empty means the set is usable."""
    problems: list[str] = []

    # Schema and ids.
    ids = [r.get("id") for r in records]
    for rec in records:
        missing = REQUIRED_KEYS - set(rec)
        if missing:
            problems.append(f"{rec.get('id', '?')}: missing keys {sorted(missing)}")
    dup_ids = [i for i, c in collections.Counter(ids).items() if c > 1]
    if dup_ids:
        problems.append(f"duplicate ids: {dup_ids[:10]}")
    print(f"[{'PASS' if not problems else 'FAIL'}] schema and ids "
          f"({len(records)} records)")

    # Domains.
    counts = collections.Counter(r.get("domain") for r in records)
    unknown = set(counts) - EXPECTED_DOMAINS
    if unknown:
        problems.append(f"unknown domains: {sorted(unknown)}")
    missing_domains = EXPECTED_DOMAINS - set(counts)
    if missing_domains:
        problems.append(f"domains with no prompts: {sorted(missing_domains)}")
    print("[INFO] per-domain counts:")
    for domain in sorted(counts):
        print(f"    {domain:12s} {counts[domain]}")
    spread = max(counts.values()) - min(counts.values()) if counts else 0
    if spread > 5:
        problems.append(f"domain counts are unbalanced, spread of {spread}")

    # Duplicates.
    seen: dict[str, str] = {}
    dupes = []
    for rec in records:
        k = norm(rec.get("prompt", ""))
        if k in seen:
            dupes.append((seen[k], rec.get("id")))
        else:
            seen[k] = rec.get("id", "?")
    if dupes:
        problems.append(f"{len(dupes)} duplicate prompts, e.g. {dupes[:5]}")
    print(f"[{'PASS' if not dupes else 'FAIL'}] no duplicate prompts")

    # Length sanity.
    bad_len = [
        (r.get("id"), len(r.get("prompt", "").split()))
        for r in records
        if not MIN_WORDS <= len(r.get("prompt", "").split()) <= MAX_WORDS
    ]
    if bad_len:
        problems.append(f"{len(bad_len)} prompts outside "
                        f"{MIN_WORDS}-{MAX_WORDS} words, e.g. {bad_len[:5]}")
    lengths = sorted(len(r.get("prompt", "").split()) for r in records)
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(f"[{'PASS' if not bad_len else 'FAIL'}] word counts "
              f"(min {lengths[0]}, median {mid}, max {lengths[-1]})")

    # Machine uniformity: the thing that quietly ruins the results.
    openers = collections.Counter(
        " ".join(r.get("prompt", "").split()[:2]).lower() for r in records
    )
    top = openers.most_common(5)
    print("[INFO] most repeated two-word openings:")
    for opener, n in top:
        share = n / len(records) if records else 0
        print(f"    {opener!r:28s} {n:3d}  ({share:.0%})")
    if top and top[0][1] / len(records) > 0.20:
        problems.append(
            f"{top[0][1]} of {len(records)} prompts start with {top[0][0]!r} "
            "-- the set is machine-uniform"
        )

    # Two defect classes the 32B actually produced: it echoed the instruction
    # header into its output, and it read "capitalised" as ALL CAPS.
    echoes = [
        r.get("id") for r in records
        if any(m in r.get("prompt", "").lower() for m in INSTRUCTION_MARKERS)
    ]
    if echoes:
        problems.append(f"{len(echoes)} prompts echo the generator's own "
                        f"instructions, e.g. {echoes[:5]}")
    print(f"[{'PASS' if not echoes else 'FAIL'}] no echoed instruction text")

    shouting = [
        r.get("id") for r in records if caps_ratio(r.get("prompt", "")) > 0.6
    ]
    if shouting:
        problems.append(f"{len(shouting)} prompts are ALL CAPS, which tokenizes "
                        f"differently and would skew acceptance, "
                        f"e.g. {shouting[:5]}")
    print(f"[{'PASS' if not shouting else 'FAIL'}] no ALL CAPS prompts")

    titled = [
        r.get("id") for r in records if is_title_case(r.get("prompt", ""))
    ]
    if titled:
        problems.append(f"{len(titled)} prompts are in Title Case, which "
                        f"tokenizes differently from ordinary text, "
                        f"e.g. {titled[:5]}")
    print(f"[{'PASS' if not titled else 'FAIL'}] no Title Case prompts")

    sources = collections.Counter(r.get("source") for r in records)
    print("[INFO] sources:")
    for source in sorted(sources, key=str):
        print(f"    {str(source):24s} {sources[source]}")
    human = sum(
        n for s, n in sources.items()
        if s and ("handwritten" in s or "edited" in s)
    )
    if human < min_handwritten:
        problems.append(
            f"only {human} records are hand-written or hand-edited, "
            f"{min_handwritten} required -- edit prompts and set their source "
            'to "hand-edited"'
        )
    print(f"[{'PASS' if human >= min_handwritten else 'FAIL'}] "
          f"human-touched records: {human}/{min_handwritten}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=EVAL_PATH)
    parser.add_argument("--min-handwritten", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"{args.path} does not exist")
        return 1

    records = read_jsonl(args.path)
    problems = check(records, args.min_handwritten)

    print()
    if problems:
        print(f"VERDICT: FAIL -- {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
