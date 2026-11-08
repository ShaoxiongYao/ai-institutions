"""Experiment 4 - institutions over time (multi-period draft, model.md §11).

2 x 2 institutions, all with the same per-period budget R and the same shocks:
  openness:            closed  | open (every firm publishes reusable results)
  ownership/financing: private (investors own and fund; firms exit after losses)
                       public  (broad ownership, tax-funded; no exit)

Scenario dimensions: pricing (§12), labour-income transfer eta, private investment
(policy budget | firms decide, §13), spillover frictions (§14), and in private institutions
whether firm D is a public lab (none | closed | open, §15). eta changes only who gets
labour income, so each path is simulated once and re-summarised for every eta.

Exports results/dynamics/trajectories.json for the animated explainer.

Run: uv run python -m experiments.dynamics
"""

from __future__ import annotations

import json

import numpy as np

from ai_institutions.dynamics import simulate_path, with_eta
from ai_institutions.investment import InvestmentRule

from .common import GROUPS, OWN_BROAD, OWN_CONCENTRATED, RESULTS, TAX_SHARES, Y0, baseline, econ_dict
from ai_institutions import run_info

NAME = "dynamics"
SEED, N_DRAWS, T = 4, 4_000, 25
M = 4
K0 = 0.3
ETAS = (0.0, 0.3)  # income transfer from workers to AI owners: off / on
PRICINGS = ("fixed", "lead")  # model.md §12
INVESTMENTS = ("budget", "firms")  # private R&D: policy budget split equally | firms decide (§13)
# Spillover frictions (model.md §14, experiment 7). "none" is the frictionless model.
FRICTIONS = {
    "none": {},
    "absorb": {"psi": 0.5},       # firms build on only half of the gap to the open frontier
    "herd": {"rho_open": 0.6},    # open firms' research paths converge
    "hard": {"phi": 0.5},         # progress gets harder as capability rises (all institutions)
    "all": {"psi": 0.5, "rho_open": 0.6, "phi": 0.5},
}
N_SAMPLES = 3  # example histories exported in full
N_FAN = 50     # histories exported as thin paths (capability, cumulative gains)
FLOWS = ("app", "capital", "t_in", "t_out", "adjust", "cost")
INSTITUTIONS = {
    "closed_private": {"label": "私人闭源", "open": False, "public": False},
    "open_private": {"label": "私人开源", "open": True, "public": False},
    "closed_public": {"label": "公共闭源", "open": False, "public": True},
    "open_public": {"label": "公共开源", "open": True, "public": True},
}


# Public lab (model.md §15): in private institutions firm D can be a tax-funded, broadly
# owned public lab with the same policy budget share (0.075); "none" = all four private.
LABS = ("none", "closed", "open")


def economy(open_: bool, public: bool, eta: float = 0.0, lab: str = "none"):
    own = OWN_BROAD if public else OWN_CONCENTRATED
    omega = np.repeat(own[:, None], M, axis=1)
    beta = np.repeat(TAX_SHARES[:, None], M, axis=1) if public else omega.copy()
    is_open = [open_] * M
    if lab != "none" and not public:
        omega[:, -1], beta[:, -1], is_open[-1] = OWN_BROAD, TAX_SHARES, lab == "open"
    return baseline(
        M=M, omega=omega, beta=beta, open=is_open,
        r=[0.075] * M, q0=[0.3, 0.25, 0.2, 0.15], h=[0.1, 0.05, 0.0, 0.0],
        sigma=1.0, rho=0.2, gamma=0.5, theta=1.0, eta=eta,
    )


def funding(public: bool, lab: str):
    """owners_fund argument: one flag, or per firm when firm D is a public lab."""
    if public:
        return False
    return True if lab == "none" else np.array([True, True, True, False])


def r4(x):
    return np.round(np.asarray(x, dtype=float), 4).tolist()


def rounded(x, d):
    return np.round(np.asarray(x, dtype=float), d).tolist()


def main() -> None:
    out = {"T": T, "M": M, "groups": list(GROUPS), "n": OWN_BROAD.tolist(), "y0": Y0.tolist(), "K0": K0,
           "etas": [str(e) for e in ETAS], "pricings": list(PRICINGS), "investments": list(INVESTMENTS),
           "frictions": FRICTIONS, "labs": list(LABS), "blocks": [], "scenarios": {}}
    # scenarios["pricing|eta|inv|fric|lab"][institution] is an index into blocks; identical
    # blocks (e.g. public institutions, which have no separate lab) are stored once.
    seen: dict[str, int] = {}

    def store(key: str, inst: str, block: dict) -> None:
        text = json.dumps(block, ensure_ascii=False, separators=(",", ":"))
        if text not in seen:
            seen[text] = len(out["blocks"])
            out["blocks"].append(block)
        out["scenarios"].setdefault(key, {})[inst] = seen[text]

    configs = {}
    for fric in FRICTIONS:
        for pricing in PRICINGS:
            public_blocks = {}  # public institutions: same for every investment mode and lab
            for inv in INVESTMENTS:
                for key, spec in INSTITUTIONS.items():
                    for lab in (("none",) if spec["public"] else LABS):
                        if spec["public"] and key in public_blocks:
                            blocks = public_blocks[key]
                        else:
                            blocks = run_one(spec, pricing, inv, fric, lab, configs, key)
                            if spec["public"]:
                                public_blocks[key] = blocks
                        for eta, block in blocks.items():
                            labs = LABS if spec["public"] else (lab,)
                            for lb in labs:
                                store(f"{pricing}|{eta}|{inv}|{fric}|{lb}", key, block)
    d = RESULTS / NAME
    d.mkdir(parents=True, exist_ok=True)
    out["run_info"] = run_info(SEED, N_DRAWS)
    (d / "trajectories.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    (d / "config.json").write_text(json.dumps(
        {"T": T, "K0": K0, "etas": list(ETAS), "pricings": list(PRICINGS), "investments": list(INVESTMENTS),
         "frictions": FRICTIONS, "labs": list(LABS),
         "investment_rule": {k: getattr(InvestmentRule(), k) for k in ("grid", "horizon", "nodes", "adjust", "cash_limited")},
         "institutions": INSTITUTIONS, "economies": configs,
         "run_info": out["run_info"]}, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(out['blocks'])} blocks, {len(out['scenarios'])} scenarios")


def run_one(spec: dict, pricing: str, invest: str, fric: str, lab: str, configs: dict, key: str) -> dict:
    """Simulate once (eta = 0) and summarise for every eta; eta only moves labour income."""
    econ0 = economy(spec["open"], spec["public"], 0.0, lab).with_(pricing=pricing, **FRICTIONS[fric])
    rule = InvestmentRule() if invest == "firms" and not spec["public"] else None
    p0 = simulate_path(econ0, T, N_DRAWS, SEED, owners_fund=funding(spec["public"], lab), K0=K0, investment=rule)
    blocks = {}
    for eta in ETAS:
        econ = econ0.with_(eta=eta)
        configs[f"{key}_{pricing}_eta{eta}_{invest}_{fric}_{lab}"] = econ_dict(econ)
        p = p0 if eta == 0 else with_eta(p0, econ, eta)
        blocks[str(eta)] = summarise(p, econ, spec, lab)
        m = blocks[str(eta)]["mean"]
        print(f"{fric:6s} {pricing:5s} {invest:6s} lab={lab:6s} eta={eta} {spec['label']}: "
              f"used {m['used'][-1]:.2f} firms {m['n_active'][-1]:.2f} R_cum {m['R_cum'][-1]:.2f}")
    return blocks


def summarise(p, econ, spec: dict, lab: str) -> dict:
    frontier = np.where(p.active, p.q_eff, -np.inf).max(axis=2)
    frontier = np.where(np.isfinite(frontier), frontier, p.H)
    n_alive = p.active.sum(axis=2)
    used = np.where(n_alive == 0, p.H, (p.m * p.q_eff).sum(axis=2))  # commons if no firms
    hhi = (p.m**2).sum(axis=2)
    n_active = p.active.sum(axis=2)
    cum_gain = p.gain.cumsum(axis=0)                     # (T, D, G)
    losers = (cum_gain < 0).astype(float) @ econ.n       # (T, D)
    net = (p.S - p.C - p.R).cumsum(axis=0)                # cumulative net value
    mean = {
        "frontier": r4(frontier.mean(1)), "used": r4(used.mean(1)), "S": r4(p.S.mean(1)),
        "hhi": r4(hhi.mean(1)), "n_active": r4(n_active.mean(1)),
        "loser_share": r4(losers.mean(1)), "net_cum": r4(net.mean(1)),
        "cum_gain": r4(cum_gain.mean(1)),                  # (T, G)
        "R_cum": r4(p.R.cumsum(axis=0).mean(1)),
        "firm_share": r4(p.capital.sum(axis=2).sum(axis=1) / np.maximum(p.S.sum(axis=1), 1e-12)),
        "cum_gain_p10": r4(np.quantile(cum_gain, 0.1, axis=1)),
        "cum_gain_p90": r4(np.quantile(cum_gain, 0.9, axis=1)),
    }
    samples = []
    for d in range(N_SAMPLES):
        samples.append({
            "q_eff": r4(p.q_eff[:, d]), "m": r4(p.m[:, d]), "active": p.active[:, d].tolist(),
            "H": r4(p.H[:, d]), "S": r4(p.S[:, d]), "used": r4(used[:, d]),
            "n_active": n_active[:, d].tolist(), "cum_gain": r4(cum_gain[:, d]),
            "loser_share": r4(losers[:, d]),
            # Cumulative group totals by channel, (T, G); see Path in dynamics.py.
            "flows": {f: r4(getattr(p, f)[:, d].cumsum(axis=0)) for f in FLOWS},
        })
    # Thin "fan" paths: 2 decimals for capability, 1 for cumulative gains (plotted ~150 px tall).
    fan = {"used": rounded(used[:, :N_FAN].T, 2),                          # (N_FAN, T)
           "cum_gain": rounded(cum_gain[:, :N_FAN].transpose(1, 0, 2), 1)}  # (N_FAN, T, G)
    return {**spec, "lab": lab, "mean": mean, "samples": samples, "fan": fan}


if __name__ == "__main__":
    main()
