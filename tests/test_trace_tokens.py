"""Token origin labelling, tested without a GPU.

The UI colours tokens by origin and the architecture notes require the field
from the first implementation, so the labelling is worth testing directly rather
than only through a full generation.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.specdec import trace_tokens


class FakeTokenizer:
    """Decodes an id to a printable stand-in, so no model is needed."""

    def decode(self, ids: list[int]) -> str:
        return f"<{ids[0]}>"


TOK = FakeTokenizer()


def origins(records):
    return [r["origin"] for r in records]


def guesses(records):
    return [r["draft_guessed"] for r in records]


def test_partial_acceptance_labels_the_correction():
    """2 of 3 accepted: two accepts, then the target's correction."""
    recs = trace_tokens(
        new_tokens=[10, 11, 99], draft_toks=[10, 11, 12],
        accepted=2, gamma=3, tokenizer=TOK)
    assert origins(recs) == ["accept", "accept", "correct"]
    # The correction must report what the DRAFT guessed at that position.
    assert guesses(recs) == ["<10>", "<11>", "<12>"]
    assert [r["text"] for r in recs] == ["<10>", "<11>", "<99>"]


def test_full_acceptance_labels_the_bonus():
    """All gamma accepted: the extra token is free, with no draft guess."""
    recs = trace_tokens(
        new_tokens=[10, 11, 12, 77], draft_toks=[10, 11, 12],
        accepted=3, gamma=3, tokenizer=TOK)
    assert origins(recs) == ["accept", "accept", "accept", "bonus"]
    assert guesses(recs)[-1] is None


def test_zero_acceptance_is_a_single_correction():
    recs = trace_tokens(
        new_tokens=[55], draft_toks=[10, 11, 12],
        accepted=0, gamma=3, tokenizer=TOK)
    assert origins(recs) == ["correct"]
    assert guesses(recs) == ["<10>"]


def test_gamma_zero_is_a_bonus_only_round():
    """gamma=0 drafts nothing, so the one token is the target's own."""
    recs = trace_tokens(new_tokens=[42], draft_toks=[], accepted=0, gamma=0,
                        tokenizer=TOK)
    assert origins(recs) == ["bonus"]
    assert guesses(recs) == [None]


def test_truncated_round_drops_labels_for_dropped_tokens():
    """After an EOS or budget cut, only the kept tokens are labelled."""
    recs = trace_tokens(
        new_tokens=[10, 11], draft_toks=[10, 11, 12],
        accepted=2, gamma=3, tokenizer=TOK)
    assert origins(recs) == ["accept", "accept"]
    assert len(recs) == 2


@pytest.mark.parametrize("gamma,accepted", [(1, 0), (1, 1), (5, 0), (5, 3),
                                            (5, 5), (12, 11), (12, 12)])
def test_every_record_is_well_formed(gamma, accepted):
    draft = list(range(100, 100 + gamma))
    new = draft[:accepted] + [999]
    recs = trace_tokens(new, draft, accepted, gamma, TOK)
    assert len(recs) == accepted + 1
    for rec in recs:
        assert set(rec) == {"text", "origin", "draft_guessed"}
        assert rec["origin"] in ("accept", "correct", "bonus")
        assert isinstance(rec["text"], str)
        if rec["origin"] == "bonus":
            assert rec["draft_guessed"] is None
        else:
            assert isinstance(rec["draft_guessed"], str)
    # Exactly one non-accept token per round: the target's own emission.
    assert origins(recs).count("accept") == accepted
    assert sum(1 for o in origins(recs) if o != "accept") == 1


def test_accept_records_agree_with_their_own_text():
    """An accepted token's draft guess must equal the token itself."""
    recs = trace_tokens([7, 8, 9, 123], [7, 8, 9], 3, 3, TOK)
    for rec in recs[:3]:
        assert rec["draft_guessed"] == rec["text"]


if __name__ == "__main__":
    for name, args in (
        ("partial", ([10, 11, 99], [10, 11, 12], 2, 3)),
        ("full", ([10, 11, 12, 77], [10, 11, 12], 3, 3)),
        ("none", ([55], [10, 11, 12], 0, 3)),
    ):
        recs = trace_tokens(*args, tokenizer=TOK)
        print(f"{name:8s} {origins(recs)}")
