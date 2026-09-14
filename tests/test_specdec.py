"""Decoder mechanics.

test_self_draft is the diagnostic: pass the target as its own draft and
acceptance must be exactly 1.0 with exactly gamma+1 tokens per round.  If that
fails the verification window offset is wrong and nothing downstream can work.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline import baseline_generate
from src.models import chat_ids, load_pair
from src.specdec import spec_generate

PROMPT = "Write a Python function that reverses a list in place."
SHORT_PROMPT = "Reply with exactly one word: the capital city of Japan."

_pair: tuple | None = None


def _load() -> tuple:
    """Load the target/draft pair once per process."""
    global _pair
    if _pair is None:
        _pair = load_pair()
    return _pair


@pytest.fixture(scope="module")
def pair():
    return _load()


def _text(tokenizer, seq, prompt_len: int) -> str:
    return tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True)


def test_self_draft_acceptance_is_exactly_one(pair):
    """The target drafting for itself must agree with itself on every token."""
    tokenizer, target, _ = pair
    ids = chat_ids(tokenizer, PROMPT)
    gamma = 4
    rounds_wanted = 5

    _, stats = spec_generate(
        target, target, ids,
        max_new_tokens=rounds_wanted * (gamma + 1),
        gamma=gamma,
        eos_id=None,
    )
    print(f"[self-draft] acceptance_rate={stats['acceptance_rate']} "
          f"tokens_per_round={stats['tokens_per_round']} "
          f"rounds={stats['rounds']} tokens={stats['tokens']}")

    assert stats["acceptance_rate"] == 1.0
    assert stats["tokens_per_round"] == gamma + 1


def test_gamma_zero_matches_baseline(pair):
    """gamma=0 is a plain target greedy step and must equal baseline exactly."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)

    base_seq, _ = baseline_generate(
        target, ids, max_new_tokens=48, eos_id=tokenizer.eos_token_id)
    spec_seq, stats = spec_generate(
        target, draft, ids, max_new_tokens=48, gamma=0,
        eos_id=tokenizer.eos_token_id)

    base_text = _text(tokenizer, base_seq, ids.shape[1])
    spec_text = _text(tokenizer, spec_seq, ids.shape[1])
    print(f"[gamma=0] proposed={stats['proposed']} rounds={stats['rounds']} "
          f"match={base_text == spec_text}")

    assert stats["proposed"] == 0
    assert spec_text == base_text


@pytest.mark.parametrize("gamma", [1, 3, 5, 8])
def test_all_gammas_agree(pair, gamma):
    """Every gamma must produce the same text as the baseline for one prompt."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)

    base_seq, _ = baseline_generate(
        target, ids, max_new_tokens=64, eos_id=tokenizer.eos_token_id)
    spec_seq, stats = spec_generate(
        target, draft, ids, max_new_tokens=64, gamma=gamma,
        eos_id=tokenizer.eos_token_id)

    base_text = _text(tokenizer, base_seq, ids.shape[1])
    spec_text = _text(tokenizer, spec_seq, ids.shape[1])
    if base_text != spec_text:
        first = next((i for i, (a, b) in enumerate(zip(base_text, spec_text))
                      if a != b), min(len(base_text), len(spec_text)))
        print(f"[gamma={gamma}] MISMATCH at char {first}")
        print(f"  baseline: {base_text[max(0, first - 40):first + 40]!r}")
        print(f"  spec:     {spec_text[max(0, first - 40):first + 40]!r}")
    else:
        print(f"[gamma={gamma}] identical, "
              f"acceptance={stats['acceptance_rate']:.3f} "
              f"tokens_per_round={stats['tokens_per_round']:.2f} "
              f"tok_per_s={stats['tok_per_s']:.2f}")

    assert spec_text == base_text


def test_stats_are_internally_consistent(pair):
    """accepted <= proposed, and a round cannot emit more than gamma+1 tokens."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    gamma = 5

    _, stats = spec_generate(
        target, draft, ids, max_new_tokens=64, gamma=gamma,
        eos_id=tokenizer.eos_token_id)
    print(f"[consistency] {stats}")

    assert stats["accepted"] <= stats["proposed"]
    assert stats["rounds"] * (gamma + 1) >= stats["tokens"]
    assert stats["proposed"] == stats["rounds"] * gamma


def test_terminates_on_eos(pair):
    """A short answer must stop at EOS rather than run to max_new_tokens."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, SHORT_PROMPT)

    seq, stats = spec_generate(
        target, draft, ids, max_new_tokens=256, gamma=5,
        eos_id=tokenizer.eos_token_id)
    last = int(seq[0, -1].item())
    print(f"[eos] tokens={stats['tokens']} last_token={last} "
          f"eos={tokenizer.eos_token_id} "
          f"text={_text(tokenizer, seq, ids.shape[1])!r}")

    assert stats["tokens"] < 256
    assert last == tokenizer.eos_token_id


def test_trace_requires_tokenizer(pair):
    """trace records carry token text, so a tokenizer is mandatory."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    with pytest.raises(ValueError):
        spec_generate(target, draft, ids, max_new_tokens=4, gamma=2, trace=[])


def test_trace_origins(pair):
    """Every traced token is labelled accept, correct or bonus, consistently."""
    tokenizer, target, draft = pair
    ids = chat_ids(tokenizer, PROMPT)
    trace: list = []

    _, stats = spec_generate(
        target, draft, ids, max_new_tokens=48, gamma=4,
        eos_id=tokenizer.eos_token_id, trace=trace, tokenizer=tokenizer)

    assert len(trace) == stats["rounds"]
    seen: dict[str, int] = {}
    for rec in trace:
        assert rec["accepted"] <= rec["gamma"]
        assert rec["rate"] == rec["accepted"] / rec["gamma"]
        accepts = [t for t in rec["tokens"] if t["origin"] == "accept"]
        assert len(accepts) == min(rec["accepted"], len(rec["tokens"]))
        for t in rec["tokens"]:
            assert t["origin"] in ("accept", "correct", "bonus")
            seen[t["origin"]] = seen.get(t["origin"], 0) + 1
            if t["origin"] == "bonus":
                assert t["draft_guessed"] is None
            else:
                assert isinstance(t["draft_guessed"], str)
    print(f"[trace] rounds={len(trace)} origins={seen}")
    assert sum(seen.values()) == stats["tokens"]


if __name__ == "__main__":
    tokenizer, target, draft = _load()
    ids = chat_ids(tokenizer, PROMPT)
    gamma = 4
    _, stats = spec_generate(target, target, ids, max_new_tokens=5 * (gamma + 1),
                             gamma=gamma, eos_id=None)
    ok = stats["acceptance_rate"] == 1.0 and stats["tokens_per_round"] == gamma + 1
    print(f"self-draft acceptance={stats['acceptance_rate']} "
          f"tokens_per_round={stats['tokens_per_round']}")
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
