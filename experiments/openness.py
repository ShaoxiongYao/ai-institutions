"""Experiment 3 - openness (one period).

Keep investment, technology, financing and ownership fixed; vary only which projects
publish reusable results, across market sensitivity kappa. One period only: openness
works through usable capability q~ and market shares, not through accumulation.

Ownership: firm 0 (largest budget) owned by investors; firms 1-2 owned half by
investors, half by 'others'. Private financing (beta = omega).

Run: uv run python -m experiments.openness
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ai_institutions import evaluate, summarize
from ai_institutions.technology import draw_shocks

from .common import GROUPS, SERIES, baseline, econ_dict, md_table, save, style

NAME = "openness"
SEED, N_DRAWS = 3, 100_000
RULES = {
    "none": [False, False, False],
    "leader (0)": [True, False, False],
    "laggards (1,2)": [False, True, True],
    "all": [True, True, True],
}
KAPPAS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
OMEGA = np.array([[1.0, 0.5, 0.5],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.5]])


def main() -> None:
    base = baseline(omega=OMEGA, beta=OMEGA)
    Z = draw_shocks(np.random.default_rng(SEED), N_DRAWS, base.M, base.sigma, base.rho)
    rows = []
    for kappa in KAPPAS:
        ref = None
        for rule, is_open in RULES.items():
            econ = base.with_(open=is_open, kappa=kappa)
            out = evaluate(econ, Z)
            s = summarize(econ, out)
            hhi = (out.m**2).sum(axis=1)
            frontier_used = (out.m * out.q_eff).sum(axis=1) / out.q.max(axis=1)
            row = {"rule": rule, "kappa": kappa,
                   "S": s["S"][0], "S_se": s["S"][1], "H": s["H"][0],
                   "hhi": float(hhi.mean()), "frontier_used": float(frontier_used.mean()),
                   "welfare_linear": s["welfare"][0], "loser_share": s["loser_share"][0]}
            for j in range(econ.M):
                row[f"profit_{j}"] = float(out.profit[:, j].mean())
                row[f"share_{j}"] = float(out.m[:, j].mean())
                row[f"loss_prob_{j}"] = s["firm_loss_prob"][j]
            for g, name in enumerate(GROUPS):
                row[f"y_mean_{name}"] = s["group_y_mean"][g]
            if rule == "none":
                ref = out.S
            # Paired (common random numbers) difference vs. no openness.
            dS = out.S - ref
            row["dS_vs_none"] = float(dS.mean())
            row["dS_vs_none_se"] = float(dS.std(ddof=1) / np.sqrt(dS.size))
            rows.append(row)

    # Note: opening a non-frontier project can in principle lower S (it lifts a laggard
    # that then takes share from a stronger closed firm), so dS >= 0 is not asserted.
    header = ["kappa", "rule", "E[S]", "dS vs none", "SE", "frontier used", "HHI",
              "profit firm0", "profit firm1", "profit firm2", *[f"E[y] {g}" for g in GROUPS]]
    body = [[r["kappa"], r["rule"], r["S"], r["dS_vs_none"], r["dS_vs_none_se"],
             r["frontier_used"], r["hhi"], r["profit_0"], r["profit_1"], r["profit_2"],
             *[r[f"y_mean_{g}"] for g in GROUPS]] for r in rows]
    config = {"baseline": econ_dict(base), "rules": RULES, "kappas": list(KAPPAS),
              "frontier_used": "E[sum_j m_j q~_j / max_j q_j]", "common_random_numbers": True}
    out_dir = save(NAME, config, rows, SEED, N_DRAWS, md_table(header, body))
    plot(rows, out_dir / "figure.png")
    print(md_table(header, body))


def plot(rows, path) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panels = [("dS_vs_none", "E[S] gain vs. no openness"),
              ("frontier_used", "usage-weighted capability / frontier"),
              ("profit_0", "expected profit, leader firm 0")]
    for ax, (key, title) in zip(axes, panels):
        for i, rule in enumerate(RULES):
            sub = sorted((r for r in rows if r["rule"] == rule), key=lambda r: r["kappa"])
            ax.plot([r["kappa"] for r in sub], [r[key] for r in sub], marker="o",
                    color=SERIES[i], label=rule)
        ax.set_xscale("log")
        ax.set_xlabel("market sensitivity kappa (log scale)")
        ax.set_title(title)
    axes[0].axhline(0, color="#52514e", linewidth=0.8)
    axes[2].axhline(0, color="#52514e", linewidth=0.8)
    axes[1].legend(title="open projects", title_fontsize=8, loc="lower right")
    fig.suptitle("Same investment, ownership and financing; only reuse rules differ", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
