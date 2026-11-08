"""Experiment 8 - a public lab next to private firms: crowding out (model.md §15).

Three privately funded closed firms choose their own R&D (InvestmentRule, §13). A fourth
firm is a public lab: tax-funded, broadly owned, fixed policy budget R_pub per period,
never exits. Compare, with common random numbers:

  none    the lab is absent (private firms alone)
  closed  public lab keeps its results
  open    public lab publishes; every firm and user can use its results

Crowding in/out = private R&D with the lab minus private R&D without it (> 0: crowding in). closed vs none
isolates "one more competitor"; open vs closed isolates the lab's openness. The lab's
R&D efficiency e (progress scaled by e) captures weaker effort in public labs.

Also: private OPEN firms next to the lab, and a per-period Nash investment rule (§13 item 4)
instead of the adaptive one, as robustness checks.

Run: uv run python -m experiments.mixed
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ai_institutions.dynamics import simulate_path
from ai_institutions.investment import NASH_RESIDUALS, InvestmentRule

from . import dynamics as dyn
from .common import GROUPS, OWN_BROAD, OWN_CONCENTRATED, SERIES, TAX_SHARES, baseline, md_table, save, style

NAME = "mixed"
SEED, DRAWS = 7, 1_000
PRICINGS = ("lead", "fixed")
LABS = ("none", "closed", "open")
BUDGETS = (0.075, 0.15, 0.3)       # public lab budget per period
EFFICIENCIES = (1.0, 0.75, 0.5, 0.25)
MAIN_BUDGET, MAIN_E = 0.15, 1.0
HORIZONS = (2.0, 5.0, 10.0)        # payoff horizon L of private firms (main: 5)
MAIN_L = 5.0
PRIVATE_R0 = 0.075                 # private firms' starting R&D (as in experiment 4)
FUNDED = np.array([True, True, True, False])
LAB_LABEL = {"none": "no public lab", "closed": "closed public lab", "open": "open public lab"}
# Investment rules: the adaptive rule of §13, and a per-period Nash equilibrium (§13 item 4).
RULES = {"adaptive": {}, "nash": {"rounds": 8, "damping": 0.5}}


def economy(lab: str, budget: float, e: float, pricing: str, fric: str = "none", popen: bool = False):
    omega = np.column_stack([OWN_CONCENTRATED] * 3 + [OWN_BROAD])
    beta = np.column_stack([OWN_CONCENTRATED] * 3 + [TAX_SHARES])
    return baseline(
        M=4, omega=omega, beta=beta, open=[popen] * 3 + [lab == "open"],
        r=[PRIVATE_R0] * 3 + [budget if lab != "none" else 0.0],
        q0=[0.3, 0.25, 0.2, 0.15], h=[0.1, 0.05, 0.0, 0.0],
        sigma=1.0, rho=0.2, gamma=0.5, theta=1.0, eta=0.0, pricing=pricing,
        efficiency=[1.0, 1.0, 1.0, e], **dyn.FRICTIONS[fric],
    )


def run(lab, budget, e, pricing, fric="none", L=MAIN_L, popen=False, rule="adaptive") -> dict:
    econ = economy(lab, budget, e, pricing, fric, popen)
    NASH_RESIDUALS.clear()
    p = simulate_path(econ, dyn.T, DRAWS, SEED, owners_fund=FUNDED, K0=dyn.K0,
                      investment=InvestmentRule(horizon=L, **RULES[rule]), active0=[True, True, True, lab != "none"])
    resid = np.array(NASH_RESIDUALS) if NASH_RESIDUALS else np.zeros((1, 2))
    r_pub = np.where(p.active[:, :, 3], econ.r[3], 0.0)  # (T, D); the lab spends its budget every period
    r_priv = p.R - r_pub
    n_act = p.active.sum(axis=2)
    used = np.where(n_act == 0, p.H, (p.m * p.q_eff).sum(axis=2))
    cum = p.gain.cumsum(axis=0)
    last = dyn.T - 1
    qp = np.sort(p.q_eff[:, :, :3], axis=2)                  # private firms' capability, ascending
    return {
        "priv_draws": r_priv.sum(axis=0), "used_draws": used[last], "net_draws": (p.S - p.C - p.R).sum(axis=0),
        "priv_t": r_priv.mean(axis=1), "used_t": used.mean(axis=1),
        "row": {
            "lab": lab, "budget": budget if lab != "none" else 0.0, "efficiency": e if lab != "none" else None,
            "pricing": pricing, "frictions": fric, "L": L, "private_open": popen, "rule": rule,
            "nash_resid_mean": float(resid[:, 0].mean()), "nash_resid_share": float(resid[:, 1].mean()),
            "R_private_cum": float(r_priv.sum(axis=0).mean()), "R_public_cum": float(r_pub.sum(axis=0).mean()),
            "R_private_last5": float(r_priv[-5:].mean()),
            "used_T": float(used[last].mean()), "net_cum": float((p.S - p.C - p.R).sum(axis=0).mean()),
            "private_alive_T": float(p.active[last, :, :3].sum(axis=1).mean()),
            "lab_share_T": float(p.m[last, :, 3].mean()),
            # Leader's lead over the second private firm, and how close the race is.
            "private_lead_T": float((qp[last, :, 2] - qp[last, :, 1]).mean()),
            "private_lead_mid": float((qp[12, :, 2] - qp[12, :, 1]).mean()),
            "private_tau_T": float((p.tau[last, :, :3] * p.m[last, :, :3]).sum(axis=1).mean()
                                   / max(p.m[last, :, :3].sum(axis=1).mean(), 1e-12)),
            **{f"cum_gain_{g}": float(cum[last, :, i].mean()) for i, g in enumerate(GROUPS)},
        },
    }


def paired(a, b):
    d = a - b
    return float(d.mean()), float(2 * d.std(ddof=1) / np.sqrt(d.size))


def main() -> None:
    cases, rows = {}, []

    def do(lab, budget, e, pricing, fric="none", L=MAIN_L, popen=False, rule="adaptive"):
        key = (lab, budget if lab != "none" else 0.0, e if lab != "none" else None, pricing, fric, L, popen, rule)
        if key not in cases:
            cases[key] = run(lab, budget, e, pricing, fric, L, popen, rule)
            rows.append(cases[key]["row"])
            r = cases[key]["row"]
            print(f"{rule:8s} {'open' if popen else 'closed'} L={L:g} {pricing:5s} {fric:4s} {lab:6s} R_pub {r['budget']:.3f} e {e}: private R&D {r['R_private_cum']:.2f} "
                  f"used {r['used_T']:.2f} net {r['net_cum']:.2f}")
        return cases[key]

    for pricing in PRICINGS:
        for fric in ("none", "all"):
            do("none", 0, 1.0, pricing, fric)
            for lab in ("closed", "open"):
                do(lab, MAIN_BUDGET, MAIN_E, pricing, fric)
                if fric == "none":
                    for b in BUDGETS:
                        do(lab, b, MAIN_E, pricing)
                    for e in EFFICIENCIES:
                        do(lab, MAIN_BUDGET, e, pricing)
        for L in HORIZONS:
            for lab in LABS:
                do(lab, MAIN_BUDGET, MAIN_E, pricing, L=L)
        # Private open firms next to a public lab; and the per-period Nash rule.
        for popen in (False, True):
            for rule in RULES:
                do("none", 0, 1.0, pricing, popen=popen, rule=rule)
                for lab in ("closed", "open"):
                    for b in (BUDGETS if rule == "adaptive" else (MAIN_BUDGET,)):
                        do(lab, b, MAIN_E, pricing, popen=popen, rule=rule)

    def get(lab, b, e, pricing, fric="none", L=MAIN_L, popen=False, rule="adaptive"):
        return cases[(lab, b if lab != "none" else 0.0, e if lab != "none" else None, pricing, fric, L, popen, rule)]

    parts = []
    head = ["public lab", "R_pub", "e", "private R&D (cum)", "Δ private R&D vs none ±2SE",
            "Δ private per unit public R&D", "public R&D (cum)", "private R&D t21-25", "private firms alive",
            "lab share (T)", "leader's lead t13", "leader's lead T", "private markup (T)", "capability used (T)", "Δ used vs none ±2SE", "cum net value",
            "Δ net vs none ±2SE", *[f"cum gain {g}" for g in GROUPS]]

    def line(c, base):
        r = c["row"]
        dp, sp = paired(c["priv_draws"], base["priv_draws"])
        du, su = paired(c["used_draws"], base["used_draws"])
        dn, sn = paired(c["net_draws"], base["net_draws"])
        per = (dp / r["R_public_cum"]) if r["R_public_cum"] > 0 else float("nan")
        return [LAB_LABEL[r["lab"]], r["budget"], r["efficiency"] if r["efficiency"] is not None else "-",
                r["R_private_cum"], f"{dp:.2f} ± {sp:.2f}", per, r["R_public_cum"], r["R_private_last5"],
                r["private_alive_T"], r["lab_share_T"], r["private_lead_mid"], r["private_lead_T"], r["private_tau_T"], r["used_T"], f"{du:.2f} ± {su:.2f}",
                r["net_cum"], f"{dn:.2f} ± {sn:.2f}", *[r[f"cum_gain_{g}"] for g in GROUPS]]

    for pricing in PRICINGS:
        base = get("none", 0, 1.0, pricing)
        body = [line(base, base)]
        body += [line(get(lab, b, MAIN_E, pricing), base) for lab in ("closed", "open") for b in BUDGETS]
        parts.append(f"### {pricing} pricing: public lab budget (e = {MAIN_E})\n\n{md_table(head, body)}")
        body = [line(get(lab, MAIN_BUDGET, e, pricing), base) for lab in ("closed", "open") for e in EFFICIENCIES]
        parts.append(f"### {pricing} pricing: public lab efficiency (R_pub = {MAIN_BUDGET})\n\n{md_table(head, body)}")
        base_f = get("none", 0, 1.0, pricing, "all")
        body = [line(get(lab, MAIN_BUDGET, MAIN_E, pricing, "all"), base_f) for lab in LABS]
        body_L = []
        for L in HORIZONS:
            bL = get("none", 0, 1.0, pricing, L=L)
            body_L += [[f"L = {L:g}", *line(get(lab, MAIN_BUDGET, MAIN_E, pricing, L=L), bL)] for lab in LABS]
        parts.append(f"### {pricing} pricing: private payoff horizon L (R_pub = {MAIN_BUDGET}, e = {MAIN_E})\n\n"
                     f"{md_table(['L', *head], body_L)}")
        for popen in (False, True):
            body_r = []
            for rule in RULES:
                b0 = get("none", 0, 1.0, pricing, popen=popen, rule=rule)
                body_r.append([rule, *line(b0, b0)])
                for lab in ("closed", "open"):
                    for b in (BUDGETS if rule == "adaptive" else (MAIN_BUDGET,)):
                        body_r.append([rule, *line(get(lab, b, MAIN_E, pricing, popen=popen, rule=rule), b0)])
            res = [get(l, MAIN_BUDGET, MAIN_E, pricing, popen=popen, rule="nash")["row"] for l in LABS]
            parts.append(f"### {pricing} pricing: private firms {'OPEN' if popen else 'closed'}, adaptive vs per-period Nash rule\n\n"
                         f"{md_table(['rule', *head], body_r)}\n\nNash iteration residual (mean |best response − choice|, "
                         f"share of firm-periods > half a grid step): "
                         + ", ".join(f"{r['lab']} {r['nash_resid_mean']:.4f} / {r['nash_resid_share']:.3f}" for r in res))
        parts.append(f"### {pricing} pricing, all three spillover frictions (§14), R_pub = {MAIN_BUDGET}, e = {MAIN_E}\n\n"
                     f"{md_table(head, body)}")
    table = "\n\n".join(parts) + ("\n\nPrivate firms: 3 closed, owned by investors, choose R&D (L = 5, λ = 0.5, cash-limited). "
                                  "Δ private R&D > 0: crowding in; < 0: crowding out. Paired differences over the same draws.")
    config = {"seed": SEED, "draws": DRAWS, "T": dyn.T, "K0": dyn.K0, "pricings": list(PRICINGS),
              "budgets": list(BUDGETS), "efficiencies": list(EFFICIENCIES), "main_budget": MAIN_BUDGET,
              "main_efficiency": MAIN_E, "rules": RULES, "horizons": list(HORIZONS), "main_L": MAIN_L, "private_r0": PRIVATE_R0, "investment_rule": "InvestmentRule() defaults",
              "frictions_robustness": dyn.FRICTIONS["all"]}
    out_dir = save(NAME, config, rows, SEED, DRAWS, table)
    plot(get, out_dir / "figure.png")
    print(table)


def plot(get, path) -> None:
    style()
    t = np.arange(1, dyn.T + 1)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for i, pricing in enumerate(PRICINGS):
        for c, lab in enumerate(LABS):
            res = get(lab, MAIN_BUDGET, MAIN_E, pricing)
            axes[i, 0].plot(t, res["priv_t"], color=SERIES[c], label=LAB_LABEL[lab])
            axes[i, 1].plot(t, res["used_t"], color=SERIES[c])
        for c, lab in enumerate(("closed", "open")):
            ys = [get(lab, MAIN_BUDGET, e, pricing)["row"]["used_T"] for e in EFFICIENCIES]
            axes[i, 2].plot(EFFICIENCIES, ys, marker="o", color=SERIES[c + 1], label=LAB_LABEL[lab])
        axes[i, 2].axhline(get("none", 0, 1.0, pricing)["row"]["used_T"], color=SERIES[0], linestyle="--",
                           label=LAB_LABEL["none"])
        axes[i, 2].invert_xaxis()
        axes[i, 0].set_title(f"{pricing} pricing: private R&D per period")
        axes[i, 1].set_title(f"{pricing} pricing: capability used")
        axes[i, 2].set_title(f"{pricing} pricing: capability at T vs lab efficiency")
    for ax in axes[1]:
        ax.set_xlabel("period")
    axes[1, 2].set_xlabel("public lab efficiency e")
    axes[0, 0].legend(fontsize=7.5)
    axes[0, 2].legend(fontsize=7.5)
    fig.suptitle(f"Three private closed firms choose R&D; a public lab spends {MAIN_BUDGET}/period (e = {MAIN_E} unless swept)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
