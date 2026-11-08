"""Experiment 1 - distribution.

Hold technology, investment, openness and therefore S, C, R fixed draw by draw.
Vary only ownership (concentrated vs broad) and financing (private vs public),
and check how group outcomes, risk and welfare under concave utility change.
Linear welfare must be identical across all cells (pure transfer).

Run: uv run python -m experiments.distribution
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ai_institutions import evaluate, summarize
from ai_institutions.technology import draw_shocks

from .common import (GROUPS, OWN_BROAD, OWN_CONCENTRATED, SERIES, TAX_SHARES, baseline,
                     econ_dict, md_table, save, style)

NAME = "distribution"
SEED, N_DRAWS = 1, 50_000
OWNERSHIP = {"concentrated": OWN_CONCENTRATED, "broad": OWN_BROAD}
FINANCING = ("private", "public")
ETAS = (0.0, 0.3)  # eta > 0 adds the structural income-transfer channel
UTILITIES = {"linear": ("linear", 1.0), "log": ("crra", 1.0), "crra2": ("crra", 2.0)}


def cell(ownership: str, financing: str, eta: float):
    base = baseline(eta=eta)
    omega = np.repeat(OWNERSHIP[ownership][:, None], base.M, axis=1)
    beta = omega if financing == "private" else np.repeat(TAX_SHARES[:, None], base.M, axis=1)
    return base.with_(omega=omega, beta=beta)


def main() -> None:
    base = baseline()
    Z = draw_shocks(np.random.default_rng(SEED), N_DRAWS, base.M, base.sigma, base.rho)
    rows = []
    for eta in ETAS:
        for own in OWNERSHIP:
            for fin in FINANCING:
                econ = cell(own, fin, eta)
                out = evaluate(econ, Z)
                row = {"ownership": own, "financing": fin, "eta": eta}
                for label, (kind, p) in UTILITIES.items():
                    s = summarize(econ, out, kind=kind, param=p, floor=0.05)
                    row[f"W_{label}"], row[f"W_{label}_se"] = s["welfare"]
                    if kind == "crra":
                        row["floor_hit_share"] = s["floor_hit_share"][0]
                row["total_resources"] = s["total_resources"][0]
                row["loser_share"], row["loser_share_se"] = s["loser_share"]
                for g, name in enumerate(GROUPS):
                    row[f"y_mean_{name}"] = s["group_y_mean"][g]
                    row[f"y_p05_{name}"] = s["group_y_p05"][g]
                    row[f"y_sd_{name}"] = float(out.y[:, g].std(ddof=1))
                    row[f"loss_prob_{name}"] = float((out.y[:, g] < econ.y0[g]).mean())
                rows.append(row)

    # Pure-transfer check: linear welfare identical within each eta.
    for eta in ETAS:
        w = [r["W_linear"] for r in rows if r["eta"] == eta]
        assert np.ptp(w) < 1e-10, "linear welfare changed under pure transfer"

    header = ["eta", "ownership", "financing", "W linear", "W log", "W CRRA(2)", "loser share",
              *[f"E[y] {g}" for g in GROUPS], *[f"P(loss) {g}" for g in GROUPS]]
    body = [[r["eta"], r["ownership"], r["financing"], r["W_linear"], r["W_log"], r["W_crra2"],
             r["loser_share"], *[r[f"y_mean_{g}"] for g in GROUPS],
             *[r[f"loss_prob_{g}"] for g in GROUPS]] for r in rows]
    config = {"baseline": econ_dict(base), "ownership": {k: v.tolist() for k, v in OWNERSHIP.items()},
              "tax_shares": TAX_SHARES.tolist(), "etas": list(ETAS), "crra_floor": 0.05,
              "common_random_numbers": True}
    out_dir = save(NAME, config, rows, SEED, N_DRAWS, md_table(header, body))
    plot(rows, base.y0, out_dir / "figure.png")
    print(md_table(header, body))


def plot(rows, y0, path) -> None:
    """Mean per-capita gain y - y0 by group, one panel per eta, bars per institution."""
    style()
    fig, axes = plt.subplots(1, len(ETAS), figsize=(9, 3.4), sharey=True)
    cells = [(o, f) for o in OWNERSHIP for f in FINANCING]
    width = 0.8 / len(cells)
    x = np.arange(len(GROUPS))
    for ax, eta in zip(axes, ETAS):
        sub = [r for r in rows if r["eta"] == eta]
        for k, (o, f) in enumerate(cells):
            r = next(r for r in sub if r["ownership"] == o and r["financing"] == f)
            gain = [r[f"y_mean_{g}"] - y0[i] for i, g in enumerate(GROUPS)]
            ax.bar(x + (k - 1.5) * width, gain, width * 0.92, color=SERIES[k], label=f"{o} / {f}")
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_xticks(x, GROUPS)
        ax.set_title(f"eta = {eta}" + (" (no income transfer)" if eta == 0 else ""))
    axes[0].set_ylabel("mean per-capita gain, y - y0")
    axes[-1].legend(title="ownership / financing", title_fontsize=8, loc="upper right")
    fig.suptitle("Same S, C, R; only ownership and financing differ", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
