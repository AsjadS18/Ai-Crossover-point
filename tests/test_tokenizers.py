"""Draft and target must share a vocabulary.

If token id n means different strings to the two models, comparing their
predictions is meaningless and the entire approach fails for this pair.
This gate runs before any decoding work.
"""

from __future__ import annotations

import os
import sys

import pytest
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_PATH = "models/target-7b"
DRAFT_PATH = "models/draft-0.5b"

PROBES: dict[str, str] = {
    "python_nested": (
        'def outer(xs):\n'
        '    """Sum the positive values.\n\n'
        '    Returns an int.\n'
        '    """\n'
        '    total = 0\n'
        '    for x in xs:\n'
        '        if x > 0:\n'
        '            total += x\n'
        '    return total\n'
    ),
    "json_escapes": '{"key": "a \\"quoted\\" value", "unicode": "\u00e9\u4e2d", "n": 1.5e-3}',
    "english_punctuation": (
        "It was -- and this matters -- the third attempt; nobody, least of all "
        "Dr. Hayes (who'd predicted it), expected 97.5% accuracy!"
    ),
    "emoji": "shipping it 🚀 looks good 👍🏽 done ✅🎉",
    "urdu": "یہ ایک آزمائشی جملہ ہے جو اردو میں لکھا گیا ہے۔",
    "whitespace_tabs": "a\t\tb\n\n\n    c\t \t d      e\n\t\n",
}

MESSAGES = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Write a function that reverses a list."},
]

_loaded: tuple | None = None


def _load() -> tuple:
    """Load both tokenizers once per process.  Returns (target_tok, draft_tok)."""
    global _loaded
    if _loaded is None:
        _loaded = (
            AutoTokenizer.from_pretrained(TARGET_PATH),
            AutoTokenizer.from_pretrained(DRAFT_PATH),
        )
    return _loaded


@pytest.fixture(scope="module")
def toks():
    return _load()


def _vocab_report(tgt, drf) -> tuple[bool, list[str]]:
    """Compare vocabs.  Returns (identical, list of human-readable difference lines)."""
    vt, vd = tgt.get_vocab(), drf.get_vocab()
    lines: list[str] = []
    only_t = sorted(set(vt) - set(vd))
    only_d = sorted(set(vd) - set(vt))
    remapped = sorted(k for k in set(vt) & set(vd) if vt[k] != vd[k])
    for k in only_t:
        lines.append(f"  target-only: {k!r} -> {vt[k]}")
    for k in only_d:
        lines.append(f"  draft-only:  {k!r} -> {vd[k]}")
    for k in remapped:
        lines.append(f"  id differs:  {k!r} target={vt[k]} draft={vd[k]}")
    return (not lines), lines


def test_vocab_identical(toks):
    tgt, drf = toks
    identical, lines = _vocab_report(tgt, drf)
    if not identical:
        print(f"[FAIL] vocab: {len(lines)} differing keys")
        for line in lines:
            print(line)
    else:
        print(f"[PASS] vocab: identical, {len(tgt.get_vocab())} tokens")
    assert identical, f"{len(lines)} vocab differences (printed above)"


@pytest.mark.parametrize("name", list(PROBES))
def test_probe_encodes_identically(toks, name):
    tgt, drf = toks
    text = PROBES[name]
    it = tgt(text).input_ids
    idr = drf(text).input_ids
    if it == idr:
        print(f"[PASS] encode {name}: {len(it)} tokens identical")
    else:
        first = next(
            (i for i, (a, b) in enumerate(zip(it, idr)) if a != b),
            min(len(it), len(idr)),
        )
        print(f"[FAIL] encode {name}: diverges at index {first}")
        print(f"  target: {it[max(0, first - 3):first + 3]}")
        print(f"  draft:  {idr[max(0, first - 3):first + 3]}")
    assert it == idr


def test_eos_token_id_matches(toks):
    tgt, drf = toks
    ok = tgt.eos_token_id == drf.eos_token_id
    print(f"[{'PASS' if ok else 'FAIL'}] eos_token_id: "
          f"target={tgt.eos_token_id} draft={drf.eos_token_id}")
    assert ok


def test_chat_template_matches(toks):
    tgt, drf = toks
    st = tgt.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    sd = drf.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    # transformers v5 returns a BatchEncoding here, not a bare list of ids.
    it = tgt.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True)["input_ids"]
    idr = drf.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True)["input_ids"]
    ok = st == sd and it == idr
    if ok:
        print(f"[PASS] chat template: identical, {len(it)} tokens")
    else:
        print("[FAIL] chat template differs")
        print(f"  target text: {st!r}")
        print(f"  draft text:  {sd!r}")
    assert ok


if __name__ == "__main__":
    tgt, drf = _load()
    results: list[bool] = []

    identical, lines = _vocab_report(tgt, drf)
    print(f"[{'PASS' if identical else 'FAIL'}] vocab "
          f"({len(tgt.get_vocab())} target / {len(drf.get_vocab())} draft tokens)")
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"  ... {len(lines) - 50} more")
    results.append(identical)

    for name, text in PROBES.items():
        ok = tgt(text).input_ids == drf(text).input_ids
        print(f"[{'PASS' if ok else 'FAIL'}] encode {name} "
              f"({len(tgt(text).input_ids)} tokens)")
        results.append(ok)

    ok = tgt.eos_token_id == drf.eos_token_id
    print(f"[{'PASS' if ok else 'FAIL'}] eos_token_id "
          f"target={tgt.eos_token_id} draft={drf.eos_token_id}")
    results.append(ok)

    st = tgt.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    sd = drf.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    ok = st == sd
    print(f"[{'PASS' if ok else 'FAIL'}] chat template")
    results.append(ok)

    print("VERDICT:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
