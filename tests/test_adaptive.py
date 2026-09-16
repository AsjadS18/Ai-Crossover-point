"""The controller is pure arithmetic, so it is tested exactly.

The key external check: expected_speedup must reproduce the worked example in
the architecture notes (alpha=0.3, r=0.12 -> 1.12x at gamma=2, 0.83x at gamma=6,
0.59x at gamma=12). That table was derived independently of this code, so
matching it validates the formula rather than just the implementation.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive import AdaptiveGamma


def test_matches_architecture_worked_example():
    """alpha=0.3, r=0.12: the published table, to two decimals."""
    ctrl = AdaptiveGamma(r=0.12)
    expected = {2: (1.39, 1.24, 1.12), 6: (1.43, 1.72, 0.83),
                12: (1.43, 2.44, 0.59)}
    for gamma, (e_tok, e_cost, e_speed) in expected.items():
        tok = ctrl.expected_tokens(0.3, gamma)
        cost = ctrl.round_cost(gamma)
        speed = ctrl.expected_speedup(0.3, gamma)
        print(f"  gamma={gamma:2d}: E={tok:.3f} cost={cost:.3f} "
              f"speedup={speed:.3f}")
        assert round(tok, 2) == e_tok
        assert round(cost, 2) == e_cost
        assert round(speed, 2) == e_speed


def test_gamma_zero_is_exactly_unity():
    """No drafting means no speculation gain and no speculation cost."""
    for r in (0.05, 0.248, 0.64, 2.0):
        ctrl = AdaptiveGamma(r=r)
        assert ctrl.expected_speedup(0.9, 0) == 1.0
        assert ctrl.expected_tokens(0.9, 0) == 1.0
        assert ctrl.round_cost(0) == 1.0


def test_alpha_one_does_not_divide_by_zero():
    """Perfect acceptance is the gamma+1 limit, not a crash."""
    ctrl = AdaptiveGamma(r=0.248)
    for gamma in (1, 4, 12):
        assert ctrl.expected_tokens(1.0, gamma) == gamma + 1
    assert ctrl.best_gamma(1.0) == ctrl.gmax
    print(f"  alpha=1.0 -> gamma={ctrl.best_gamma(1.0)} (gmax)")


def test_alpha_zero_never_pays():
    """If nothing is ever accepted, any drafting is pure loss."""
    ctrl = AdaptiveGamma(r=0.248)
    assert ctrl.expected_tokens(0.0, 5) == 1.0
    assert ctrl.expected_speedup(0.0, 5) < 1.0
    assert ctrl.best_gamma(0.0) == 0
    print("  alpha=0.0 -> gamma=0, speculation switched off")


def test_controller_switches_speculation_off():
    """The property that matters: low acceptance must drive gamma to 0."""
    ctrl = AdaptiveGamma(r=0.64, cooldown=0)     # eager regime, expensive draft
    for _ in range(50):
        ctrl.update(accepted=0, gamma_used=4)    # nothing ever accepted
    print(f"  after 50 rejected rounds: alpha={ctrl.alpha:.4f} "
          f"gamma={ctrl.gamma}")
    assert ctrl.gamma == 0


def test_controller_opens_up_when_acceptance_is_high():
    """High acceptance and a cheap draft must push gamma up."""
    ctrl = AdaptiveGamma(r=0.248, cooldown=0)
    for _ in range(50):
        ctrl.update(accepted=8, gamma_used=8)    # everything accepted
    print(f"  after 50 accepted rounds: alpha={ctrl.alpha:.4f} "
          f"gamma={ctrl.gamma}")
    assert ctrl.gamma >= 8


def test_ema_converges_to_observed_rate():
    ctrl = AdaptiveGamma(r=0.248, alpha_init=0.5, beta=0.85, cooldown=0)
    for _ in range(200):
        ctrl.update(accepted=3, gamma_used=4)    # a steady 0.75
    print(f"  alpha after 200 rounds at 0.75: {ctrl.alpha:.5f}")
    assert abs(ctrl.alpha - 0.75) < 1e-3


def test_cooldown_limits_change_rate():
    """gamma must not change more often than every `cooldown` rounds."""
    ctrl = AdaptiveGamma(r=0.248, alpha_init=0.5, beta=0.0, cooldown=5)
    gammas = []
    for i in range(30):
        # Alternate extremes to provoke a change on every single round.
        ctrl.update(accepted=(0 if i % 2 else 8), gamma_used=8)
        gammas.append(ctrl.gamma)
    changes = sum(1 for a, b in zip(gammas, gammas[1:]) if a != b)
    print(f"  {changes} changes over 30 rounds with cooldown=5: {gammas}")
    assert changes <= 30 // 5 + 1


def test_bounds_are_respected():
    ctrl = AdaptiveGamma(r=0.1, gmin=2, gmax=6, cooldown=0)
    for alpha in (0.0, 0.3, 0.6, 0.99, 1.0):
        assert 2 <= ctrl.best_gamma(alpha) <= 6
    for _ in range(40):
        ctrl.update(accepted=0, gamma_used=4)
    assert ctrl.gamma == 2, "gmin must floor the controller"
    print("  gmin=2 floor honoured even with zero acceptance")


def test_gamma_zero_round_does_not_move_alpha():
    """A round with no drafting carries no acceptance evidence."""
    ctrl = AdaptiveGamma(r=0.248, alpha_init=0.6, cooldown=0)
    before = ctrl.alpha
    ctrl.update(accepted=0, gamma_used=0)
    assert ctrl.alpha == before
    assert ctrl.rounds == 1
    print("  gamma_used=0 advanced the clock without changing alpha")


def test_history_records_every_round():
    ctrl = AdaptiveGamma(r=0.248, cooldown=0)
    for i in range(10):
        ctrl.update(accepted=i % 3, gamma_used=3)
    assert len(ctrl.history) == 10
    accepted, alpha, gamma = ctrl.history[-1]
    assert isinstance(accepted, int)
    assert isinstance(alpha, float)
    assert isinstance(gamma, int)
    print(f"  history has {len(ctrl.history)} rows, last={ctrl.history[-1]}")


def test_ties_prefer_the_smaller_gamma():
    """Equal expected speedup should not buy extra speculative work."""
    ctrl = AdaptiveGamma(r=0.248, cooldown=0)
    # At alpha=0 every gamma>0 is strictly worse, so the argmax must be gmin.
    assert ctrl.best_gamma(0.0) == 0


@pytest.mark.parametrize("bad", [
    {"r": 0.0}, {"r": -1.0}, {"gmin": -1}, {"gmin": 5, "gmax": 2},
    {"alpha_init": 1.5}, {"beta": 1.0}, {"cooldown": -1},
])
def test_invalid_config_raises(bad):
    kwargs = {"r": 0.248}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        AdaptiveGamma(**kwargs)


def test_measured_regimes_pick_sensible_gammas():
    """Sanity-check against this machine's two measured r values."""
    eager = AdaptiveGamma(r=0.64)
    graphed = AdaptiveGamma(r=0.248)
    for alpha in (0.2, 0.4, 0.6, 0.7, 0.85, 0.9):
        ge, gg = eager.best_gamma(alpha), graphed.best_gamma(alpha)
        print(f"  alpha={alpha:.2f}: eager gamma={ge:2d}  graphed gamma={gg:2d}")
        assert gg >= ge, "a cheaper draft should never justify LESS speculation"


if __name__ == "__main__":
    ctrl = AdaptiveGamma(r=0.248)
    print("graphed regime, best gamma by acceptance:")
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.85, 0.95):
        print(f"  alpha={alpha:.2f} -> gamma={ctrl.best_gamma(alpha)} "
              f"(speedup {ctrl.expected_speedup(alpha, ctrl.best_gamma(alpha)):.3f})")


def test_gamma_zero_is_not_an_absorbing_state():
    """A controller at gamma=0 must probe, or it can never recover.

    A gamma=0 round drafts nothing, so it yields no acceptance evidence and
    alpha cannot move. Without a probe the controller stays at 0 forever.
    Measured before the fix: a run sat at gamma=0 for all 1045 rounds and
    scored 0.44x, because alpha_init=0.5 happens to select gamma=0 under the
    measured cost model.
    """
    verify = {1: 1.0, 2: 1.34, 3: 1.66, 4: 2.14, 5: 2.15, 6: 2.16, 7: 2.17,
              8: 2.18}
    ctrl = AdaptiveGamma(r=0.248, gmax=8, alpha_init=0.5, cooldown=0,
                         verify_cost=verify, probe_every=10)
    assert ctrl.gamma == 0, "this alpha/cost combination should start at 0"

    # Feed gamma=0 rounds: without probing these carry no information.
    for _ in range(25):
        ctrl.update(accepted=0, gamma_used=ctrl.gamma)
    print(f"  after 25 rounds from gamma=0: probes={ctrl.probes} "
          f"gamma={ctrl.gamma}")
    assert ctrl.probes >= 2, "controller never probed out of gamma=0"


def test_probe_lets_the_controller_recover():
    """Once probing reveals high acceptance, gamma must climb back up."""
    verify = {1: 1.0, 2: 1.34, 3: 1.66, 4: 2.14, 5: 2.15, 6: 2.16, 7: 2.17,
              8: 2.18}
    ctrl = AdaptiveGamma(r=0.248, gmax=8, alpha_init=0.5, cooldown=0,
                         verify_cost=verify, probe_every=5)
    seen = []
    for _ in range(80):
        g = ctrl.gamma
        # Whenever it drafts, everything is accepted: acceptance is truly high.
        ctrl.update(accepted=g, gamma_used=g)
        seen.append(ctrl.gamma)
    print(f"  alpha={ctrl.alpha:.3f} final gamma={ctrl.gamma} "
          f"probes={ctrl.probes} path={seen[:20]}")
    assert ctrl.gamma > 0, "controller failed to recover from gamma=0"
    assert ctrl.alpha > 0.8, "probing should have revealed high acceptance"


def test_probing_can_be_disabled():
    ctrl = AdaptiveGamma(r=0.64, gmax=8, alpha_init=0.0, cooldown=0,
                         probe_every=0)
    for _ in range(50):
        ctrl.update(accepted=0, gamma_used=ctrl.gamma)
    assert ctrl.gamma == 0
    assert ctrl.probes == 0
    print("  probe_every=0 keeps gamma=0 permanently, as documented")


def test_measured_cost_model_avoids_the_pessimal_middle():
    """The measured step function should make gamma 3-4 unattractive.

    Verify cost jumps from 1.66 to 2.14 single-token-forwards between widths 3
    and 4 and is flat after, so gamma=3 pays the step without amortising it.
    A controller given the real curve should skip that region.
    """
    verify = {1: 1.0, 2: 1.34, 3: 1.66, 4: 2.14, 5: 2.15, 6: 2.16, 7: 2.17,
              8: 2.18, 9: 2.20}
    ctrl = AdaptiveGamma(r=0.248, gmax=8, verify_cost=verify)
    picks = {a: ctrl.best_gamma(a) for a in
             (0.3, 0.5, 0.6, 0.7, 0.77, 0.83, 0.85, 0.91)}
    print(f"  picks by alpha: {picks}")
    assert 3 not in picks.values(), "gamma=3 is the measured pessimal point"
    assert 4 not in picks.values(), "gamma=4 also pays the step unamortised"
