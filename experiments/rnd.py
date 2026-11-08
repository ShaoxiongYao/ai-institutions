"""Experiment 2 - R&D concentration under uncertainty.

Fix total budget R and split it equally across k symmetric projects (q0 = 0, h = 0,
all closed, broad ownership so distribution is not the issue). Vary k, the return
to input alpha, outcome uncertainty sigma and correlation rho.

Outcomes:
  frontier   E[max_j q_j]          - best capability achieved
  value      E[S]                  - value with closed models and finite kappa
  P(q* > b)  breakthrough prob.    - frontier exceeds bar b

Run: uv run python -m experiments.rnd
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ai_institutions import Economy, evaluate
from math import erf, log, sqrt

from ai_institutions.technology import draw_shocks, latent_correlation, progress

from .common import GROUPS, N, SERIES, Y0, md_table, save, style

NAME = "rnd"
SEED, N_DRAWS = 2, 200_000
R_TOTAL = 0.3
KS = (1, 2, 3, 4, 6, 8)
ALPHAS = (0.5, 1.0, 1.5)
SIGMAS = (0.0, 0.5, 1.0, 2.0)
RHOS = (0.0, 0.6)
STARTUP_COSTS = (0.0, 0.03)
KAPPA, THETA = 3.0, 1.2


def bar(alpha: float) -> float:
    """Breakthrough bar: 1.5x the deterministic single-project frontier."""
    return 1.5 * float(progress(R_TOTAL, 1.0, alpha))


def expected_max2(sigma: float, rho: float) -> float:
    """Analytic E[max(Z1, Z2)] for mean-1 lognormal shocks: 2 Phi(s sqrt((1 - rho_X) / 2))."""
    if sigma == 0:
        return 1.0
    s = sqrt(log(1 + sigma**2))
    x = s * sqrt((1 - latent_correlation(sigma, rho, 2)) / 2)
    return 1 + erf(x / sqrt(2))  # = 2 Phi(x)


def economy(k: int, alpha: float, sigma: float, rho: float, c: float) -> Economy:
    omega = np.repeat(N[:, None], k, axis=1)
    return Economy(
        n=N, y0=Y0, a=N, omega=omega, beta=omega, d=N, delta=N,
        r=np.full(k, R_TOTAL / k), q0=np.zeros(k), h=np.zeros(k), open=np.zeros(k, bool),
        A=1.0, alpha=alpha, sigma=sigma, rho=rho, startup_cost=c,
        kappa=KAPPA, theta=THETA, gamma=1.0, chi=0.0, tau=0.5,
    )


def main() -> None:
    rows = []
    for c in STARTUP_COSTS:
        for alpha in ALPHAS:
            for rho in RHOS:
                for sigma in SIGMAS:
                    for k in KS:
                        if sigma > 0 and k > 1 and rho < 0:
                            continue
                        rng = np.random.default_rng(SEED)  # same seed per cell
                        Z = draw_shocks(rng, N_DRAWS, k, sigma, rho)
                        econ = economy(k, alpha, sigma, rho, c)
                        out = evaluate(econ, Z)
                        qmax = out.q.max(axis=1)
                        se = lambda x: float(x.std(ddof=1) / np.sqrt(x.size))  # noqa: E731
                        rows.append({
                            "startup_cost": c, "alpha": alpha, "rho": rho, "sigma": sigma, "k": k,
                            "frontier": float(qmax.mean()), "frontier_se": se(qmax),
                            "value": float(out.S.mean()), "value_se": se(out.S),
                            "p_breakthrough": float((qmax > bar(alpha)).mean()),
                            "frontier_sd": float(qmax.std(ddof=1)),
                        })

    # Analytic checks (c = 0): k = 1 gives R^alpha in expectation; k = 2 gives
    # (R/2)^alpha E[max(Z1, Z2)]; sigma = 0 is exact.
    for r in rows:
        if r["startup_cost"] != 0 or r["k"] > 2:
            continue
        m = 1.0 if r["k"] == 1 else expected_max2(r["sigma"], r["rho"])
        exact = (R_TOTAL / r["k"]) ** r["alpha"] * m
        r["frontier_exact"] = exact
        assert abs(r["frontier"] - exact) <= 4 * r["frontier_se"] + 1e-12, r

    best = []
    for c in STARTUP_COSTS:
        for alpha in ALPHAS:
            for rho in RHOS:
                for sigma in SIGMAS:
                    sub = [r for r in rows if (r["startup_cost"], r["alpha"], r["rho"], r["sigma"])
                           == (c, alpha, rho, sigma)]
                    # k = 1 has an exact expected frontier; a k > 1 counts as better
                    # only if its estimate beats that by more than 2 Monte Carlo SE.
                    ref = float(progress(R_TOTAL, 1.0, alpha, c))
                    better = [r for r in sub if r["k"] > 1
                              and r["frontier"] - ref > 2 * r["frontier_se"]]
                    bf = max(better, key=lambda r: r["frontier"]) if better else None
                    bv = max(sub, key=lambda r: r["value"])
                    bp = max(sub, key=lambda r: r["p_breakthrough"])
                    best.append([c, alpha, rho, sigma, bf["k"] if bf else 1, bv["k"], bp["k"],
                                 100 * (bf["frontier"] / ref - 1) if bf else 0.0,
                                 100 * bf["frontier_se"] / ref if bf else 0.0])
    header = ["startup c", "alpha", "rho", "sigma", "best k (frontier)", "best k (value)",
              "best k (P breakthrough)", "frontier gain vs k=1 (%)", "MC SE (%)"]
    config = {"R_total": R_TOTAL, "ks": list(KS), "alphas": list(ALPHAS), "sigmas": list(SIGMAS),
              "rhos": list(RHOS), "startup_costs": list(STARTUP_COSTS), "kappa": KAPPA,
              "theta": THETA, "breakthrough_bar": "1.5 * A * R^alpha", "A": 1.0,
              "common_random_numbers": "same seed per cell"}
    out_dir = save(NAME, config, rows, SEED, N_DRAWS, md_table(header, best))
    plot(rows, out_dir / "figure.png")
    print(md_table(header, best))


def plot(rows, path) -> None:
    """E[frontier] relative to k=1, vs k; columns alpha, rows rho; lines sigma (c = 0)."""
    style()
    fig, axes = plt.subplots(len(RHOS), len(ALPHAS), figsize=(9.5, 5.4), sharex=True)
    for i, rho in enumerate(RHOS):
        for j, alpha in enumerate(ALPHAS):
            ax = axes[i, j]
            for s, sigma in enumerate(SIGMAS):
                sub = sorted((r for r in rows if r["startup_cost"] == 0 and r["alpha"] == alpha
                              and r["rho"] == rho and r["sigma"] == sigma), key=lambda r: r["k"])
                ref = float(progress(R_TOTAL, 1.0, alpha))  # exact k = 1 value
                ax.plot([r["k"] for r in sub], [r["frontier"] / ref for r in sub], marker="o",
                        color=SERIES[s], label=f"sigma = {sigma}")
            ax.axhline(1, color="#52514e", linewidth=0.8)
            ax.set_title(f"alpha = {alpha}, rho = {rho}")
            if i == len(RHOS) - 1:
                ax.set_xlabel("number of projects k (equal split of R)")
            if j == 0:
                ax.set_ylabel("E[max q] / E[max q | k=1]")
    axes[0, -1].legend(loc="upper right")
    fig.suptitle(f"Splitting a fixed budget R = {R_TOTAL}: expected frontier", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
