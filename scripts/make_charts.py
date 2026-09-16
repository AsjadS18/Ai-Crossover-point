"""Static PNG charts from the sweep results, for the README and the video.

Reads whichever sweeps exist:
    results/sweep.jsonl          eager regime
    results/sweep_graphed.jsonl  CUDA-graph regime

Writes into charts/.  matplotlib only, no seaborn, no styling packages.

    python scripts/make_charts.py
    python scripts/make_charts.py --regime graphed
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")           # no display on this box; write files only
import matplotlib.pyplot as plt

from src.adaptive import AdaptiveGamma

SWEEPS = {
    "eager": ("results/sweep.jsonl", 0.64),
    "graphed": ("results/sweep_graphed.jsonl", 0.248),
}
CHART_DIR = "charts"
DOMAINS = ["structured", "code", "math", "reasoning", "translation", "prose"]
COLOURS = {
    "structured": "#1f77b4", "code": "#2ca02c", "math": "#9467bd",
    "reasoning": "#ff7f0e", "translation": "#8c564b", "prose": "#d62728",
}


def read_runs(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record") == "run":
                rows.append(rec)
    return rows


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def series(runs: list[dict], domain: str, field: str, gammas: list[int]):
    """Mean of `field` per gamma for one domain.  None where there is no data."""
    out = []
    for gamma in gammas:
        vals = [r[field] for r in runs
                if r["mode"] == "spec" and r["domain"] == domain
                and r["gamma"] == gamma and r.get(field) is not None]
        out.append(mean(vals) if vals else None)
    return out


def chart_acceptance(runs, gammas, regime: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    for domain in DOMAINS:
        ys = series(runs, domain, "acceptance_rate", gammas)
        if all(y is None for y in ys):
            continue
        ax.plot(gammas, ys, marker="o", label=domain, color=COLOURS[domain])
    ax.set_xlabel("gamma (draft tokens per round)")
    ax.set_ylabel("mean acceptance rate")
    ax.set_title(f"Acceptance falls with gamma, and separates by domain "
                 f"({regime})")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = f"{CHART_DIR}/acceptance_{regime}.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def chart_speedup(runs, gammas, regime: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    for domain in DOMAINS:
        ys = series(runs, domain, "speedup", gammas)
        if all(y is None for y in ys):
            continue
        ax.plot(gammas, ys, marker="o", label=domain, color=COLOURS[domain])
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1)
    ax.annotate("break-even", (gammas[0], 1.0), textcoords="offset points",
                xytext=(4, 5), color="#d62728", fontsize=8)
    ax.set_xlabel("gamma (draft tokens per round)")
    ax.set_ylabel("mean speedup vs that prompt's own baseline")
    ax.set_title(f"The crossover: where speculation stops paying ({regime})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = f"{CHART_DIR}/speedup_{regime}.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def chart_model_vs_measured(runs, gammas, regime: str, r: float) -> str:
    """Measured speedup against the cost model, using each domain's own alpha.

    If the model tracks the measurement, the economics in adaptive.py are sound
    and the controller is optimising something real.
    """
    ctrl = AdaptiveGamma(r=r)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    for domain in ("structured", "prose"):
        acc = series(runs, domain, "acceptance_rate", gammas)
        meas = series(runs, domain, "speedup", gammas)
        if all(a is None for a in acc):
            continue
        pred = [ctrl.expected_speedup(a, g) if a is not None else None
                for a, g in zip(acc, gammas)]
        ax.plot(gammas, meas, marker="o", color=COLOURS[domain],
                label=f"{domain} measured")
        ax.plot(gammas, pred, marker="x", linestyle=":", color=COLOURS[domain],
                label=f"{domain} predicted")
    ax.axhline(1.0, color="#999", linestyle="--", linewidth=1)
    ax.set_xlabel("gamma")
    ax.set_ylabel("speedup")
    ax.set_title(f"Cost model vs measurement, r={r} ({regime})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = f"{CHART_DIR}/model_vs_measured_{regime}.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def chart_regime_comparison() -> str | None:
    """Absolute throughput: eager baseline through to graphed speculation."""
    data = {}
    for regime, (path, _) in SWEEPS.items():
        runs = read_runs(path)
        if not runs:
            continue
        base = [r["tok_per_s"] for r in runs if r["mode"] == "baseline"]
        spec = [r for r in runs if r["mode"] == "spec"]
        if not base or not spec:
            continue
        gammas = sorted({r["gamma"] for r in spec})
        best = max(
            (mean([r["tok_per_s"] for r in spec if r["gamma"] == g]), g)
            for g in gammas
        )
        data[regime] = (mean(base), best[0], best[1])
    if len(data) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=140)
    labels, values, colours = [], [], []
    for regime in ("eager", "graphed"):
        if regime not in data:
            continue
        base, best, gamma = data[regime]
        labels += [f"{regime}\nbaseline", f"{regime}\nspec g={gamma}"]
        values += [base, best]
        colours += ["#8b93a3", COLOURS["structured"]]
    bars = ax.bar(labels, values, color=colours)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("tokens / second")
    ax.set_title("Where the throughput actually comes from")
    ax.grid(alpha=0.25, axis="y")
    path = f"{CHART_DIR}/regime_comparison.png"
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=sorted(SWEEPS), default=None)
    args = parser.parse_args()

    os.makedirs(CHART_DIR, exist_ok=True)
    written: list[str] = []
    regimes = [args.regime] if args.regime else list(SWEEPS)

    for regime in regimes:
        path, r = SWEEPS[regime]
        runs = read_runs(path)
        if not runs:
            print(f"{regime}: {path} not present, skipping")
            continue
        gammas = sorted({rec["gamma"] for rec in runs if rec["mode"] == "spec"})
        prompts = len({rec["id"] for rec in runs if rec["mode"] == "baseline"})
        print(f"{regime}: {len(runs)} runs, {prompts} prompts, gammas {gammas}")
        written += [
            chart_acceptance(runs, gammas, regime),
            chart_speedup(runs, gammas, regime),
            chart_model_vs_measured(runs, gammas, regime, r),
        ]

    comparison = chart_regime_comparison()
    if comparison:
        written.append(comparison)
    else:
        print("regime comparison needs BOTH sweeps; skipping")

    for path in written:
        size = os.path.getsize(path)
        print(f"wrote {path} ({size / 1024:.0f} KB)")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
