"""Draft the evaluation prompt set with the local 32B via Ollama.

Machine-generated prompts are a STARTING POINT, not the deliverable.  A
uniform prompt set produces suspiciously clean acceptance curves, so this
script deliberately asks for varied register and length, and the plan requires
hand-editing at least 50 records afterwards.  scripts/check_eval.py reports how
many are still untouched.

Writes data/eval_raw.jsonl incrementally (flushed per record, so an interrupted
run loses nothing) and then assembles data/eval.jsonl.

    python scripts/make_eval.py                  all domains to target count
    python scripts/make_eval.py --domain prose   just one
    python scripts/make_eval.py --assemble-only  rebuild eval.jsonl from raw
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:32b"
RAW_PATH = "data/eval_raw.jsonl"
EVAL_PATH = "data/eval.jsonl"
PER_DOMAIN = 35
BATCH = 12
MAX_CALLS_PER_DOMAIN = 8
REQUEST_TIMEOUT = 900

# Provisional taxonomy: guide 07 was not available when this was written.
# The Phase 6 sweep showed these six separate cleanly by acceptance rate,
# from 0.94 (structured) down to 0.32 (translation).
DOMAINS: dict[str, tuple[str, str]] = {
    "structured": (
        "st",
        "requests for machine-readable output only: JSON, YAML, TOML, CSV, a "
        "SQL DDL statement, an HTTP request body, a config file",
    ),
    "code": (
        "cd",
        "requests to write, fix, or explain a specific piece of code in "
        "Python, JavaScript, SQL or shell",
    ),
    "reasoning": (
        "rs",
        "multi-step reasoning puzzles, scheduling and logic problems, "
        "trade-off analysis, questions that need a chain of thought",
    ),
    "prose": (
        "pr",
        "natural-language writing and explanation: summaries, changelog "
        "entries, documentation paragraphs, emails, plain-English answers",
    ),
    "math": (
        "mt",
        "arithmetic, algebra, calculus and probability problems where the "
        "working should be shown",
    ),
    "translation": (
        "tr",
        "translation between English and other languages, especially Urdu, "
        "Arabic, Chinese and French, in both directions",
    ),
}

# Asking one model for "varied" prompts in a single call does not work: the
# first batch came back uniformly terse and lowercase. So the register is
# dictated per call and rotated, which spreads variety across the whole set.
STYLES: list[str] = [
    "Terse, lowercase, no final punctuation, like someone typing fast.",
    # "capitalised" alone was read as ALL CAPS and produced 31 shouting
    # prompts, so this spells out sentence case explicitly.
    "Normal sentence case - first letter capitalised, the rest lower case, "
    "NEVER all capitals - fully punctuated, often phrased as a question, "
    "sometimes with a short pleasantry.",
    "Multi-clause and specific: name concrete files, numbers, column names, "
    "versions or error strings, 20 to 40 words.",
    # "capitalised first word" produced 23 Title Case prompts, so this says
    # explicitly that the remaining words stay lower case.
    "Blunt imperative commands under 12 words. ONLY the very first word is "
    "capitalised; every other word stays lower case. Not Title Case.",
    "Slightly sloppy: one typo or a missing apostrophe, informal wording, "
    "as a real user would type in a hurry.",
    "Framed as context plus a request: one clause of situation, then what is "
    "wanted, separated by a comma or dash.",
]

INSTRUCTION = """You are helping build an evaluation set of PROMPTS for testing a code assistant.

Write {n} different prompts in the "{domain}" category: {description}.

REGISTER FOR THIS BATCH -- every prompt must be written this way:
{style}

Hard requirements:
- Output ONLY the prompts, one per line. No numbering, no bullets, no commentary.
- Each prompt must be a single line, between 4 and 40 words.
- No two prompts may ask for the same thing, and vary the opening words: do not
  start more than two prompts with the same word.
- Do NOT answer the prompts. Write only the prompts themselves.

Avoid anything resembling these, which are already in the set:
{avoid}
"""


def ollama(prompt: str, temperature: float = 0.95) -> str:
    """POST to the local Ollama generate endpoint.  Returns the response text."""
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.95},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"]


# The model echoed the instruction header into its output, and 39 of those
# lines were accepted as prompts. The content AFTER the header was fine, so the
# prefix is stripped rather than the line discarded.
ECHO_PREFIX = re.compile(
    r"^\s*register for (?:this )?batch\s*[-:–—]*\s*", re.IGNORECASE)

# Phrases that only ever appear when the model is repeating the instructions.
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
    """True if nearly every word starts with a capital, as in Title Case.

    Title Case tokenizes differently from an ordinary sentence, so it is the
    same class of artifact as ALL CAPS, just quieter.
    """
    words = [w for w in text.split() if w[:1].isalpha()]
    if len(words) < 4:
        return False
    return sum(1 for w in words if w[0].isupper()) / len(words) > 0.8


def clean(line: str) -> str:
    """Strip list markers, echoed instruction headers, quotes and whitespace."""
    line = line.strip()
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
    line = ECHO_PREFIX.sub("", line)
    line = line.strip().strip('"').strip("'").strip()
    return line


def usable(text: str) -> bool:
    """Is this line plausibly a prompt rather than commentary or an answer?"""
    words = text.split()
    if not 4 <= len(words) <= 45:
        return False
    if text.startswith(("```", "#", "Here are", "Sure", "Certainly")):
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in INSTRUCTION_MARKERS):
        return False
    # ALL CAPS and Title Case prompts tokenize very differently from ordinary
    # text and would skew acceptance for reasons unrelated to domain.
    if caps_ratio(text) > 0.6:
        return False
    if is_title_case(text):
        return False
    return True


def key(text: str) -> str:
    """Normalised form used for duplicate detection."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


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


def append_raw(record: dict[str, Any]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(RAW_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def seed_records() -> list[dict[str, Any]]:
    """Hand-written prompts already in data/eval.jsonl, kept and relabelled."""
    seeds = []
    for rec in read_jsonl(EVAL_PATH):
        if rec.get("source", "").startswith(("interim-handwritten", "handwritten")):
            seeds.append({
                "domain": rec["domain"],
                "prompt": rec["prompt"],
                "source": "handwritten",
            })
    return seeds


def generate(domain: str, want: int, taken: set[str], style_offset: int = 0) -> int:
    """Generate up to `want` new prompts for `domain`.  Returns how many landed.

    `style_offset` rotates which register this domain starts with.
    """
    _, description = DOMAINS[domain]
    got = 0
    for call in range(MAX_CALLS_PER_DOMAIN):
        if got >= want:
            break
        avoid = "\n".join(list(taken)[-8:]) or "(nothing yet)"
        # Offset the style by domain as well as call, so domain N does not
        # always open with the same register as domain N-1.
        style = STYLES[(call + style_offset) % len(STYLES)]
        started = time.perf_counter()
        try:
            raw = ollama(INSTRUCTION.format(
                n=BATCH, domain=domain, description=description,
                style=style, avoid=avoid))
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  [{domain}] call {call + 1} failed: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - started

        fresh = 0
        for line in raw.splitlines():
            text = clean(line)
            if not usable(text):
                continue
            k = key(text)
            if k in taken:
                continue
            taken.add(k)
            append_raw({"domain": domain, "prompt": text, "source": "hand-edited"})
            fresh += 1
            got += 1
            if got >= want:
                break
        print(f"  [{domain}] call {call + 1}: +{fresh} new "
              f"({got}/{want}) in {elapsed:.0f}s", flush=True)
    return got


def assemble() -> dict[str, int]:
    """Build data/eval.jsonl from the hand-written seeds plus the raw pool.

    Returns a per-domain count.  Ids are <prefix>_NNN, numbered per domain with
    hand-written records first so they keep stable low ids.
    """
    pool = seed_records() + [
        r for r in read_jsonl(RAW_PATH) if r.get("source") != "handwritten"
    ]

    seen: set[str] = set()
    by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in DOMAINS}
    for rec in pool:
        domain = rec.get("domain")
        if domain not in by_domain:
            continue
        k = key(rec["prompt"])
        if k in seen:
            continue
        seen.add(k)
        if len(by_domain[domain]) < PER_DOMAIN:
            by_domain[domain].append(rec)

    records = []
    for domain, (prefix, _) in DOMAINS.items():
        for i, rec in enumerate(by_domain[domain], start=1):
            records.append({
                "id": f"{prefix}_{i:03d}",
                "domain": domain,
                "prompt": rec["prompt"],
                "source": rec["source"],
            })

    with open(EVAL_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    return {d: len(v) for d, v in by_domain.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(DOMAINS), default=None)
    parser.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()

    if not args.assemble_only:
        taken = {key(r["prompt"]) for r in seed_records() + read_jsonl(RAW_PATH)}
        have: dict[str, int] = {d: 0 for d in DOMAINS}
        for rec in seed_records() + read_jsonl(RAW_PATH):
            if rec.get("domain") in have:
                have[rec["domain"]] += 1

        targets = [args.domain] if args.domain else list(DOMAINS)
        for offset, domain in enumerate(targets):
            want = max(0, args.per_domain - have[domain])
            print(f"[{domain}] have {have[domain]}, want {want} more", flush=True)
            if want:
                generate(domain, want, taken, style_offset=offset)

    counts = assemble()
    total = sum(counts.values())
    print(f"\nwrote {EVAL_PATH}: {total} prompts")
    for domain, n in counts.items():
        print(f"  {domain:12s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
