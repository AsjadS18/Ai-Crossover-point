"""The gamma controller: tune speculation live, including switching it off.

The economics of one speculative round, with acceptance rate alpha, draft/target
cost ratio r, and gamma drafted tokens:

    E[tokens per round] = (1 - alpha^(gamma+1)) / (1 - alpha)
    cost per round      = gamma * r + 1
    speedup             = E[tokens] / cost

E[tokens] SATURATES as gamma grows -- once acceptance is low, extra draft tokens
almost never survive -- while cost grows linearly forever.  Hence a crossover,
and hence a controller: estimate alpha from what actually happened, then pick the
gamma that maximises expected speedup.

gamma=0 is deliberately in range.  A controller that can disable speculation
entirely is more useful than one that can only tune it, and on this machine three
of six domains never pay at any gamma.

This module is pure arithmetic: no torch, no model, no I/O.
"""

from __future__ import annotations


class AdaptiveGamma:
    """Closed-loop gamma controller driven by an EMA of acceptance.

    Usage from the decoder: read ``.gamma`` at the start of each round, call
    ``.update(accepted, gamma_used)`` at the end.
    """

    def __init__(
        self,
        r: float,
        gmin: int = 0,
        gmax: int = 12,
        alpha_init: float = 0.5,
        beta: float = 0.85,
        cooldown: int = 3,
        verify_cost: dict[int, float] | None = None,
        probe_every: int = 20,
    ) -> None:
        """Configure the controller.

        r          draft cost as a fraction of one target forward (0.248 graphed
                   on this machine, 0.64 eager)
        gmin/gmax  inclusive bounds on gamma; gmin=0 allows switching off
        alpha_init starting acceptance estimate, before any evidence
        beta       EMA weight on the existing estimate; higher is smoother
        cooldown   minimum rounds between gamma changes, to stop oscillation

        verify_cost optional MEASURED cost of one verify forward, keyed by token
                   width (gamma+1), in units of a single-token forward. The
                   default None uses the textbook linear model, cost = gamma*r+1.

                   That linear model is WRONG on this machine, measurably.
                   scripts/measure_verify_width.py found the graphed verify cost
                   is a STEP function: 23.5 / 31.5 / 39.0 ms at widths 1/2/3,
                   then 50.4 ms at width 4 and almost flat to 52.7 ms at width
                   13. So gamma=3 pays the entire step while verifying only four
                   tokens, and gamma>=5 pays the same step while amortising it
                   over more. A linear model cannot represent that, and a
                   controller using it parks in gamma 2-4, the worst region.

        probe_every how often to force a probe round at gamma=1 while sitting at
                   gamma=0.  Without this gamma=0 is an ABSORBING STATE: a round
                   that drafts nothing produces no acceptance evidence, so alpha
                   freezes and gamma can never rise again.  Measured: a
                   controller that reached gamma=0 stayed there for all 1045
                   rounds of a run and scored 0.44x.  Set 0 to disable probing
                   (and accept that gamma=0 becomes permanent).
        """
        if not 0.0 < r:
            raise ValueError(f"r must be positive, got {r}")
        if gmin < 0 or gmax < gmin:
            raise ValueError(f"need 0 <= gmin <= gmax, got {gmin}, {gmax}")
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"alpha_init must be in [0, 1], got {alpha_init}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")

        self.r = r
        self.gmin = gmin
        self.gmax = gmax
        self.alpha = alpha_init
        self.beta = beta
        self.cooldown = cooldown
        self.verify_cost = dict(verify_cost) if verify_cost else None
        self.probe_every = probe_every
        self._rounds_at_zero: int = 0
        self.probes: int = 0

        self.gamma: int = self.best_gamma(self.alpha)
        self.rounds: int = 0
        self._last_change: int = 0
        self.history: list[tuple[int, float, int]] = []

    def expected_tokens(self, alpha: float, gamma: int) -> float:
        """Expected tokens emitted in one round (tokens, including the free one).

        Between 1 and gamma+1.  The alpha=1 limit is gamma+1, handled explicitly
        because the closed form divides by 1-alpha.
        """
        if gamma <= 0:
            return 1.0
        if alpha >= 1.0:
            return float(gamma + 1)
        return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)

    def round_cost(self, gamma: int) -> float:
        """Cost of one round in units of one single-token target forward.

        With a measured verify_cost table, the round is gamma draft steps plus
        the real cost of a width-(gamma+1) verify.  Without one, the textbook
        linear model gamma*r + 1 -- which is wrong on this machine, see
        __init__.  Widths beyond the table reuse its largest entry, since the
        measured curve is flat there.
        """
        if self.verify_cost:
            width = gamma + 1
            if width in self.verify_cost:
                verify = self.verify_cost[width]
            else:
                widest = max(self.verify_cost)
                verify = self.verify_cost[widest if width > widest
                                          else min(self.verify_cost)]
            return gamma * self.r + verify
        return gamma * self.r + 1.0

    def expected_speedup(self, alpha: float, gamma: int) -> float:
        """Expected speedup over plain decoding, dimensionless.

        Exactly 1.0 at gamma=0: one target forward, one token, no draft cost.
        """
        return self.expected_tokens(alpha, gamma) / self.round_cost(gamma)

    def best_gamma(self, alpha: float) -> int:
        """The gamma in [gmin, gmax] maximising expected speedup at ``alpha``.

        Ties go to the SMALLER gamma: identical expected speedup for less
        speculative work is strictly preferable, and it keeps the controller from
        drifting upward on flat regions of the curve.
        """
        best_gamma = self.gmin
        best_value = self.expected_speedup(alpha, self.gmin)
        for gamma in range(self.gmin + 1, self.gmax + 1):
            value = self.expected_speedup(alpha, gamma)
            if value > best_value:
                best_gamma, best_value = gamma, value
        return best_gamma

    def update(self, accepted: int, gamma_used: int) -> int:
        """Fold one round's result into the estimate and return the next gamma.

        ``accepted`` of ``gamma_used`` draft tokens survived.  A round with
        gamma_used == 0 carries no information about acceptance, so it advances
        the clock without touching the estimate.
        """
        self.rounds += 1

        if gamma_used > 0:
            observed = accepted / gamma_used
            self.alpha = self.beta * self.alpha + (1.0 - self.beta) * observed
            self._rounds_at_zero = 0
        else:
            # No drafting means no acceptance evidence, so alpha cannot move.
            self._rounds_at_zero += 1

        candidate = self.best_gamma(self.alpha)
        if candidate != self.gamma:
            if self.rounds - self._last_change >= self.cooldown:
                self.gamma = candidate
                self._last_change = self.rounds

        # Escape the gamma=0 absorbing state. Sitting at 0 generates no
        # evidence, so without an occasional probe the controller can never
        # learn that acceptance has improved and speculation would stay off for
        # the rest of the session.
        if (self.gamma == 0 and self.probe_every > 0
                and self._rounds_at_zero >= self.probe_every):
            self.gamma = max(1, self.gmin)
            self._rounds_at_zero = 0
            self._last_change = self.rounds
            self.probes += 1

        self.history.append((accepted, self.alpha, self.gamma))
        return self.gamma


if __name__ == "__main__":
    # Reproduce the worked example from the architecture notes: alpha=0.3,
    # r=0.12 should show speculation going net-negative as gamma grows.
    ctrl = AdaptiveGamma(r=0.12)
    print(f"{'gamma':>6} {'E[tokens]':>10} {'cost':>7} {'speedup':>9}")
    for gamma in (0, 2, 6, 12):
        print(f"{gamma:6d} {ctrl.expected_tokens(0.3, gamma):10.3f} "
              f"{ctrl.round_cost(gamma):7.3f} "
              f"{ctrl.expected_speedup(0.3, gamma):9.3f}")

    print("\nmeasured regimes on this machine:")
    for label, r in (("eager  r=0.64", 0.64), ("graphed r=0.248", 0.248)):
        c = AdaptiveGamma(r=r)
        picks = {a: c.best_gamma(a) for a in (0.2, 0.4, 0.6, 0.8, 0.9)}
        print(f"  {label}: best gamma by alpha {picks}")
