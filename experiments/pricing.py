"""Experiment 5 - endogenous pricing share (model.md §12).

A. One period: the openness experiment (experiments/openness.py) with fixed tau = 0.5
   versus lead-based pricing. S is identical under both; only who captures it changes.
B. Multi-period: the four institutions of experiments/dynamics.py under fixed vs
   lead-based pricing, plus a tau_min sweep for the private open institution.

Run: uv run python -m experiments.pricing
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ai_institutions import evaluate
from ai_institutions.dynamics import simulate_path
from ai_institutions.technology import draw_shocks

from . import dynamics as dyn
from . import openness as opn
from .common import GROUPS, INK2, SERIES, econ_dict, md_table, save, style

NAME = "pricing"
SEED_A, DRAWS_A = 5, 50_000
SEED_B, DRAWS_B = 6, 2_000
KAPPAS = (1.0, 3.0, 10.0)
PRICING = {"fixed": {"pricing": "fixed"}, "lead": {"pricing": "lead"}}
TAU_MIN_SWEEP = (0.05, 0.1, 0.15, 0.2, 0.3)
TAU_MAX_SWEEP = (0.4, 0.55, 0.7, 0.9)  # closed private vs open private, lead pricing


def part_a() -> list[dict]:
    base = opn.baseline(omega=opn.OMEGA, beta=opn.OMEGA)
    Z = draw_shocks(np.random.default_rng(SEED_A), DRAWS_A, base.M, base.sigma, base.rho)
    rows = []
    for kappa in KAPPAS:
        for rule, is_open in opn.RULES.items():
            for pname, pk in PRICING.items():
                econ = base.with_(open=is_open, kappa=kappa, tau=0.5, **pk)
                out = evaluate(econ, Z)
                firm_take = (out.tau * out.v).sum(axis=1)
                row = {"part": "A", "kappa": kappa, "rule": rule, "pricing": pname,
                       "S": float(out.S.mean()),
                       "tau_eff": float(firm_take.sum() / out.S.sum()),
                       "firm_take": float(firm_take.mean()),
                       "user_side": float((out.S - firm_take).mean())}
                for j in range(econ.M):
                    row[f"profit_{j}"] = float(out.profit[:, j].mean())
                for g, name in enumerate(GROUPS):
                    row[f"y_mean_{name}"] = float(out.y[:, g].mean())
                rows.append(row)
    return rows


def run_dyn(key: str, spec: dict, **changes):
    econ = dyn.economy(spec["open"], spec["public"], 0.0).with_(**changes)
    p = simulate_path(econ, dyn.T, DRAWS_B, SEED_B, owners_fund=not spec["public"], K0=dyn.K0)
    n_act = p.active.sum(axis=2)
    used = np.where(n_act == 0, p.H, (p.m * p.q_eff).sum(axis=2))
    firm_take = p.capital.sum(axis=2)                        # (T, D): total sum_j tau_j v_j
    cum = p.gain.cumsum(axis=0)
    return econ, {
        "used": used.mean(axis=1), "n_active": n_act.mean(axis=1),
        "p_all_exit": (n_act == 0).mean(axis=1),
        "tau_eff": firm_take.sum(axis=1) / p.S.sum(axis=1),
        "cum_gain": cum.mean(axis=1),                        # (T, G)
        "R_cum": p.R.cumsum(axis=0).mean(axis=1),
        "net_cum": (p.S - p.C - p.R).cumsum(axis=0).mean(axis=1),
    }


def part_b():
    rows, series = [], {}
    for key, spec in dyn.INSTITUTIONS.items():
        for pname, pk in PRICING.items():
            econ, s = run_dyn(key, spec, **pk)
            series[(key, pname)] = s
            L = dyn.T - 1
            row = {"part": "B", "institution": spec["label"], "key": key, "pricing": pname,
                   "used_T": float(s["used"][L]), "n_active_T": float(s["n_active"][L]),
                   "p_all_exit_T": float(s["p_all_exit"][L]), "tau_eff_T": float(s["tau_eff"][L]),
                   "R_cum_T": float(s["R_cum"][L]), "net_cum_T": float(s["net_cum"][L])}
            for g, name in enumerate(GROUPS):
                row[f"cum_gain_{name}"] = float(s["cum_gain"][L, g])
            rows.append(row)
    sweep = []
    spec = dyn.INSTITUTIONS["open_private"]
    for tmin in TAU_MIN_SWEEP:
        _, s = run_dyn("open_private", spec, pricing="lead", tau_min=tmin)
        L = dyn.T - 1
        sweep.append({"part": "B-sweep", "tau_min": tmin, "used_T": float(s["used"][L]),
                      "n_active_T": float(s["n_active"][L]), "p_all_exit_T": float(s["p_all_exit"][L]),
                      "R_cum_T": float(s["R_cum"][L]),
                      "cum_gain_investors": float(s["cum_gain"][L, 0]),
                      "cum_gain_workers": float(s["cum_gain"][L, 1])})
    for tmax in TAU_MAX_SWEEP:
        for key in ("closed_private", "open_private"):
            spec = dyn.INSTITUTIONS[key]
            _, s = run_dyn(key, spec, pricing="lead", tau_max=tmax)
            L = dyn.T - 1
            sweep.append({"part": "B-sweep-max", "tau_max": tmax, "institution": spec["label"],
                          "used_T": float(s["used"][L]), "n_active_T": float(s["n_active"][L]),
                          "tau_eff_T": float(s["tau_eff"][L]),
                          **{f"cum_gain_{g}": float(s["cum_gain"][L, i]) for i, g in enumerate(GROUPS)}})
    return rows, sweep, series


def main() -> None:
    a = part_a()
    b, sweep, series = part_b()

    # Sanity: pricing never changes S in one period.
    for kappa in KAPPAS:
        for rule in opn.RULES:
            s = [r["S"] for r in a if r["kappa"] == kappa and r["rule"] == rule]
            assert abs(s[0] - s[1]) < 1e-12

    ta = md_table(["kappa", "rule", "pricing", "E[S]", "firm share", "firms take", "user side",
                   "profit firm0", "profit firm1", "profit firm2", *[f"E[y] {g}" for g in GROUPS]],
                  [[r["kappa"], r["rule"], r["pricing"], r["S"], r["tau_eff"], r["firm_take"], r["user_side"],
                    r["profit_0"], r["profit_1"], r["profit_2"], *[r[f"y_mean_{g}"] for g in GROUPS]] for r in a])
    tb = md_table(["institution", "pricing", "capability used (T)", "firms alive (T)", "P(all exit)",
                   "firm share (T)", "cum R&D", "cum net value", *[f"cum gain {g}" for g in GROUPS]],
                  [[r["institution"], r["pricing"], r["used_T"], r["n_active_T"], r["p_all_exit_T"],
                    r["tau_eff_T"], r["R_cum_T"], r["net_cum_T"], *[r[f"cum_gain_{g}"] for g in GROUPS]] for r in b])
    ts = md_table(["tau_min", "capability used (T)", "firms alive (T)", "P(all exit)", "cum R&D",
                   "cum gain investors", "cum gain workers"],
                  [[r["tau_min"], r["used_T"], r["n_active_T"], r["p_all_exit_T"], r["R_cum_T"],
                    r["cum_gain_investors"], r["cum_gain_workers"]] for r in sweep if r["part"] == "B-sweep"])
    tm = md_table(["tau_max", "institution", "capability used (T)", "firms alive (T)", "firm share (T)",
                   *[f"cum gain {g}" for g in GROUPS]],
                  [[r["tau_max"], r["institution"], r["used_T"], r["n_active_T"], r["tau_eff_T"],
                    *[r[f"cum_gain_{g}"] for g in GROUPS]] for r in sweep if r["part"] == "B-sweep-max"])
    table = (f"## A. One period (openness experiment), fixed tau = 0.5 vs lead pricing\n\n{ta}\n\n"
             f"## B. Multi-period institutions (eta = 0), T = {dyn.T}\n\n{tb}\n\n"
             f"## B-sweep. Private open, lead pricing, varying tau_min\n\n{ts}\n\n"
             f"## B-sweep-max. Lead pricing, varying tau_max\n\n{tm}")
    lead_defaults = dyn.economy(False, False).with_(pricing="lead")
    config = {"part_a": {"seed": SEED_A, "draws": DRAWS_A, "kappas": list(KAPPAS), "rules": opn.RULES,
                         "baseline": econ_dict(opn.baseline(omega=opn.OMEGA, beta=opn.OMEGA)), "fixed_tau": 0.5},
              "part_b": {"seed": SEED_B, "draws": DRAWS_B, "T": dyn.T, "K0": dyn.K0,
                         "tau_min_sweep": list(TAU_MIN_SWEEP), "tau_max_sweep": list(TAU_MAX_SWEEP)},
              "lead_pricing": {k: getattr(lead_defaults, k) for k in ("tau_min", "tau_max", "tau_scale")}}
    out_dir = save(NAME, config, a + b + sweep, SEED_A, DRAWS_A, table)
    plot_a(a, out_dir / "figure_one_period.png")
    plot_b(series, out_dir / "figure_dynamics.png")
    print(table)


def plot_a(rows, path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rules = list(opn.RULES)
    x = np.arange(len(rules))
    for ax, (key, title) in zip(axes, [("tau_eff", "share of value captured by firms"),
                                       ("profit_0", "expected profit, leader firm 0")]):
        for i, pname in enumerate(PRICING):
            vals = [next(r[key] for r in rows if r["kappa"] == 3.0 and r["rule"] == ru and r["pricing"] == pname)
                    for ru in rules]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=SERIES[i], label=f"{pname} pricing")
        ax.axhline(0, color=INK2, linewidth=0.8)
        if key == "tau_eff":
            ax.set_ylim(0, 0.66)  # leave room for the legend above the bars
        ax.set_xticks(x, rules)
        ax.set_xlabel("open projects")
        ax.set_title(title + " (kappa = 3)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_b(series, path) -> None:
    style()
    keys = list(dyn.INSTITUTIONS)
    fig, axes = plt.subplots(2, len(keys), figsize=(12, 5.4), sharey="row", sharex=True)
    t = np.arange(1, dyn.T + 1)
    for j, key in enumerate(keys):
        for i, pname in enumerate(PRICING):
            s = series[(key, pname)]
            axes[0, j].plot(t, s["used"], color=SERIES[i], label=f"{pname} pricing")
            axes[1, j].plot(t, s["n_active"], color=SERIES[i])
        axes[0, j].set_title(key.replace("_", " "))
        axes[1, j].set_xlabel("period")
    axes[0, 0].set_ylabel("capability used")
    axes[1, 0].set_ylabel("firms alive (mean)")
    axes[0, -1].legend(loc="upper left")
    fig.suptitle("Same budget and shocks; pricing fixed (tau = 0.5) vs set by lead", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
