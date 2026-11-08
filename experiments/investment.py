"""Experiment 6 - endogenous private investment (model.md §13).

Private institutions: firms choose their own R&D each period (InvestmentRule).
Public institutions: policy budget R = 0.3 per period, split equally (as before).
Both pricing rules; payoff horizon L swept.

Run: uv run python -m experiments.investment
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ai_institutions.dynamics import simulate_path
from ai_institutions.investment import InvestmentRule

from . import dynamics as dyn
from .common import GROUPS, SERIES, md_table, save, style

NAME = "investment"
SEED, DRAWS = 7, 1_000
PRICINGS = ("fixed", "lead")
HORIZONS = (1.0, 3.0, 5.0, 10.0)
MAIN_L = 5.0


def run(key: str, pricing: str, L: float) -> dict:
    spec = dyn.INSTITUTIONS[key]
    econ = dyn.economy(spec["open"], spec["public"], 0.0).with_(pricing=pricing)
    rule = None if spec["public"] else InvestmentRule(horizon=L)
    p = simulate_path(econ, dyn.T, DRAWS, SEED, owners_fund=not spec["public"], K0=dyn.K0, investment=rule)
    n_act = p.active.sum(axis=2)
    used = np.where(n_act == 0, p.H, (p.m * p.q_eff).sum(axis=2))
    cum = p.gain.cumsum(axis=0)
    L_ = dyn.T - 1
    return {
        "R_t": p.R.mean(axis=1), "used_t": used.mean(axis=1),
        "row": {
            "institution": spec["label"], "key": key, "pricing": pricing, "L": L if rule else None,
            "R_first5": float(p.R[:5].mean()), "R_last5": float(p.R[-5:].mean()),
            "R_cum": float(p.R.sum(axis=0).mean()), "p_zero_rd_last5": float((p.R[-5:].sum(axis=0) == 0).mean()),
            "used_T": float(used[L_].mean()), "n_active_T": float(n_act[L_].mean()),
            "firm_share_T": float(p.capital[L_].sum(axis=1).sum() / p.S[L_].sum()),
            "net_cum": float((p.S - p.C - p.R).sum(axis=0).mean()),
            **{f"cum_gain_{g}": float(cum[L_, :, i].mean()) for i, g in enumerate(GROUPS)},
        },
    }


def main() -> None:
    rows, series = [], {}
    for pricing in PRICINGS:
        for key, spec in dyn.INSTITUTIONS.items():
            for L in (HORIZONS if not spec["public"] else (MAIN_L,)):
                res = run(key, pricing, L)
                rows.append(res["row"])
                if L == MAIN_L:
                    series[(key, pricing)] = res
                print(f"{pricing:5s} {spec['label']} L={L}: R last5 {res['row']['R_last5']:.3f} "
                      f"used_T {res['row']['used_T']:.2f} firms {res['row']['n_active_T']:.2f}")

    main_rows = [r for r in rows if r["L"] in (None, MAIN_L)]
    head = ["pricing", "institution", "R&D/period (t1-5)", "R&D/period (t21-25)", "cum R&D",
            "P(no R&D, t21-25)", "capability used (T)", "firms alive (T)", "firm share (T)",
            "cum net value", *[f"cum gain {g}" for g in GROUPS]]
    t1 = md_table(head, [[r["pricing"], r["institution"], r["R_first5"], r["R_last5"], r["R_cum"],
                          r["p_zero_rd_last5"], r["used_T"], r["n_active_T"], r["firm_share_T"],
                          r["net_cum"], *[r[f"cum_gain_{g}"] for g in GROUPS]] for r in main_rows])
    t2 = md_table(["pricing", "institution", "L", "R&D/period (t21-25)", "cum R&D", "capability used (T)",
                   "firms alive (T)", *[f"cum gain {g}" for g in GROUPS]],
                  [[r["pricing"], r["institution"], r["L"], r["R_last5"], r["R_cum"], r["used_T"],
                    r["n_active_T"], *[r[f"cum_gain_{g}"] for g in GROUPS]] for r in rows if r["L"] is not None])
    table = (f"## Main (L = {MAIN_L}; public institutions use the fixed budget 0.3/period)\n\n{t1}\n\n"
             f"## Payoff horizon L (private institutions)\n\n{t2}\n\n"
             f"Fixed public budget for comparison: 0.3 per period, 7.5 cumulative.")
    config = {"seed": SEED, "draws": DRAWS, "T": dyn.T, "K0": dyn.K0, "horizons": list(HORIZONS),
              "main_L": MAIN_L, "grid": list(InvestmentRule().grid), "nodes": InvestmentRule().nodes,
              "public_budget": 0.3}
    out_dir = save(NAME, config, rows, SEED, DRAWS, table)
    plot(series, out_dir / "figure.png")
    print(table)


def plot(series, path) -> None:
    style()
    keys = list(dyn.INSTITUTIONS)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True, sharey="col")
    t = np.arange(1, dyn.T + 1)
    for i, pricing in enumerate(PRICINGS):
        for j, key in enumerate(keys):
            s = series[(key, pricing)]
            ls = "--" if dyn.INSTITUTIONS[key]["public"] else "-"
            lab = key.replace("_", " ") + (" (fixed budget)" if dyn.INSTITUTIONS[key]["public"] else "")
            axes[i, 0].plot(t, s["R_t"], color=SERIES[j], linestyle=ls, label=lab)
            axes[i, 1].plot(t, s["used_t"], color=SERIES[j], linestyle=ls)
        axes[i, 0].set_ylabel(f"{pricing} pricing")
        axes[i, 0].set_title(f"R&D spending per period ({pricing} pricing)")
        axes[i, 1].set_title(f"capability used ({pricing} pricing)")
    for ax in axes[1]:
        ax.set_xlabel("period")
    axes[0, 0].legend(loc="upper right", fontsize=7.5)
    fig.suptitle(f"Private firms choose their own R&D (payoff horizon L = {MAIN_L:g}); public budget fixed",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
