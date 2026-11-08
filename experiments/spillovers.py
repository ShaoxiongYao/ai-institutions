"""Experiment 7 - frictions in open spillovers (model.md §14).

In the frictionless multi-period model every open institution whose firms survive
ends at the same capability (~10 after 25 periods): all firms start each period from
the best open result, so the frontier rises by f(R/M) E[max_j Z_j] per period,
independent of ownership, pricing or transfers. This experiment relaxes the three
assumptions behind that number, one at a time and then together:

  psi       absorption: firms can use only a share psi of the gap to the open frontier
  rho_open  homogenisation: open firms' R&D shocks are more correlated (rho_open > rho)
  phi       difficulty: progress per period scaled by (1 + q)^(-phi)

Same dynamics setup as experiment 4 (T = 25, M = 4, budget R = 0.3 split equally,
eta = 0), both pricing rules, common random numbers.

Run: uv run python -m experiments.spillovers
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ai_institutions.dynamics import simulate_path

from . import dynamics as dyn
from .common import GROUPS, INK2, SERIES, md_table, save, style

NAME = "spillovers"
SEED, DRAWS = 11, 2_000
PRICINGS = ("fixed", "lead")
SWEEPS = {
    "psi": (1.0, 0.75, 0.5, 0.25, 0.0),
    "rho_open": (0.2, 0.4, 0.6, 0.8, 0.95),
    "phi": (-0.25, 0.0, 0.25, 0.5, 1.0),
}
BASE = {"psi": 1.0, "rho_open": 0.2, "phi": 0.0}  # rho_open = rho: frictionless
COMBINED = {"psi": 0.5, "rho_open": 0.6, "phi": 0.5}
RHO_SCAN = tuple(np.round(np.arange(0.2, 1.0001, 0.05), 2))  # crossing open vs closed
LABEL = {"psi": "absorption psi", "rho_open": "open-shock correlation rho_open", "phi": "difficulty phi"}


def run(key: str, pricing: str, changes: dict, draws: int = DRAWS) -> dict:
    spec = dyn.INSTITUTIONS[key]
    econ = dyn.economy(spec["open"], spec["public"], 0.0).with_(pricing=pricing, **changes)
    p = simulate_path(econ, dyn.T, draws, SEED, owners_fund=not spec["public"], K0=dyn.K0)
    n_act = p.active.sum(axis=2)
    used = np.where(n_act == 0, p.H, (p.m * p.q_eff).sum(axis=2))
    cum = p.gain.cumsum(axis=0)
    net = (p.S - p.C - p.R).sum(axis=0)
    last = dyn.T - 1
    return {
        "used_draws": used[last], "net_draws": net,
        "row": {
            "institution": spec["label"], "key": key, "pricing": pricing, **changes,
            "used_T": float(used[last].mean()), "net_cum": float(net.mean()),
            "n_active_T": float(n_act[last].mean()),
            "loser_share_T": float(((cum[last] < 0).astype(float) @ econ.n).mean()),
            **{f"cum_gain_{g}": float(cum[last, :, i].mean()) for i, g in enumerate(GROUPS)},
        },
    }


def gap_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))


def main() -> None:
    rows, cases = [], {}
    for pricing in PRICINGS:
        for param, values in SWEEPS.items():
            for val in values:
                changes = {**BASE, param: val}
                for key in dyn.INSTITUTIONS:
                    res = run(key, pricing, changes)
                    rows.append({**res["row"], "sweep": param})
                    cases[(pricing, param, val, key)] = res
        for key in dyn.INSTITUTIONS:
            res = run(key, pricing, COMBINED)
            rows.append({**res["row"], "sweep": "combined"})
            cases[(pricing, "combined", None, key)] = res
        print(f"{pricing}: done")

    # Where does homogenisation make open worse than closed? (fixed pricing; paired by draw)
    scan = []
    for c in RHO_SCAN:
        ch = {**BASE, "rho_open": float(c)}
        res = {k: run(k, "fixed", ch) for k in dyn.INSTITUTIONS}
        for own in ("public", "private"):
            g, se = gap_se(res[f"open_{own}"]["used_draws"], res[f"closed_{own}"]["used_draws"])
            gn, sen = gap_se(res[f"open_{own}"]["net_draws"], res[f"closed_{own}"]["net_draws"])
            scan.append({"rho_open": float(c), "ownership": own, "used_gap": g, "used_gap_se": se,
                         "net_gap": gn, "net_gap_se": sen})

    table = tables(rows, cases, scan)
    config = {"seed": SEED, "draws": DRAWS, "T": dyn.T, "K0": dyn.K0, "M": dyn.M, "eta": 0.0,
              "investment": "budget (R = 0.3 split equally)", "sweeps": {k: list(v) for k, v in SWEEPS.items()},
              "base": BASE, "combined": COMBINED, "rho_scan": list(map(float, RHO_SCAN)),
              "note": "rho_open = rho = 0.2 is the frictionless case; closed institutions ignore rho_open."}
    out_dir = save(NAME, config, rows + [{"scan": s} for s in scan], SEED, DRAWS, table)
    plot(cases, out_dir / "figure.png")
    print(table)


def tables(rows, cases, scan) -> str:
    keys = list(dyn.INSTITUTIONS)
    labels = [dyn.INSTITUTIONS[k]["label"] for k in keys]
    parts = []
    for pricing in PRICINGS:
        for param, values in [*SWEEPS.items(), ("combined", (None,))]:
            body = []
            for val in values:
                used = [cases[(pricing, param, val, k)]["row"]["used_T"] for k in keys]
                net = [cases[(pricing, param, val, k)]["row"]["net_cum"] for k in keys]
                alive = [cases[(pricing, param, val, k)]["row"]["n_active_T"] for k in keys]
                gp, sp = gap_se(cases[(pricing, param, val, "open_public")]["used_draws"],
                                cases[(pricing, param, val, "closed_public")]["used_draws"])
                body.append([val if val is not None else str(COMBINED), *used, *net, alive[0], alive[1],
                             f"{gp:.2f} ± {2 * sp:.2f}"])
            head = [param, *[f"used {x}" for x in labels], *[f"net {x}" for x in labels],
                    f"firms {labels[0]}", f"firms {labels[1]}", "open−closed used (public) ±2SE"]
            parts.append(f"### {pricing} pricing, sweep {param}\n\n{md_table(head, body)}")
    scan_body = [[s["rho_open"], s["ownership"], f"{s['used_gap']:.2f} ± {2 * s['used_gap_se']:.2f}",
                  f"{s['net_gap']:.2f} ± {2 * s['net_gap_se']:.2f}"] for s in scan]
    parts.append("### Open minus closed vs rho_open (fixed pricing, same ownership; ±2 SE, paired draws)\n\n"
                 + md_table(["rho_open", "ownership", "capability used at T", "cum net value"], scan_body))
    detail = [r for r in rows if r["sweep"] == "combined" or (r["sweep"] == "psi" and r["psi"] == 1.0)]
    parts.append("### Group gains, frictionless vs combined frictions\n\n" + md_table(
        ["pricing", "case", "institution", "loser share", *[f"cum gain {g}" for g in GROUPS]],
        [[r["pricing"], "combined" if r["sweep"] == "combined" else "frictionless", r["institution"],
          r["loser_share_T"], *[r[f"cum_gain_{g}"] for g in GROUPS]] for r in detail]))
    return "\n\n".join(parts)


def plot(cases, path) -> None:
    style()
    keys = list(dyn.INSTITUTIONS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
    for i, pricing in enumerate(PRICINGS):
        for j, (param, values) in enumerate(SWEEPS.items()):
            ax = axes[i, j]
            for c, key in enumerate(keys):
                y = [cases[(pricing, param, v, key)]["row"]["used_T"] for v in values]
                ax.plot(values, y, marker="o", color=SERIES[c],
                        linestyle="--" if dyn.INSTITUTIONS[key]["public"] else "-",
                        label=key.replace("_", " "))
            ax.axvline(BASE[param], color=INK2, linewidth=0.8, linestyle=":")
            ax.set_title(f"{pricing} pricing: {LABEL[param]}")
            if param == "psi":
                ax.invert_xaxis()
            if j == 0:
                ax.set_ylabel("capability used after 25 periods")
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Relaxing frictionless spillovers, one at a time (dotted line: frictionless baseline; "
                 "with fixed pricing open private and open public coincide)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
