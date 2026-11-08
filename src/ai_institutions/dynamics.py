"""Multi-period extension (model.md §11). Draft for v0.2.

Each period repeats the one-period model, with three links between periods:
  1. Knowledge carries over: q0_{j,t+1} = q~_{j,t} for surviving firms.
  2. Open results persist: H_t = max(H_{t-1}, max_{j open, active} q_{j,t}).
  3. Exit: if owners fund R&D, firm j exits once cumulative profit falls below -K0.
     Under public funding owners pay nothing, so profit tau v_j >= 0 and no firm exits.
The per-period budget R = econ.R is split equally among surviving firms. If no firm
survives, users still use open knowledge: S = s(H), with no firm share (closed
knowledge is lost with its firm). Pricing shares tau_j follow econ.pricing (§12).
Spillover frictions (§14): psi limits how much of the open frontier a firm builds
on next period (it can always serve H this period), rho_open sets the shock
correlation of open firms, phi scales the progress step by (1 + q_start)^(-phi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distribution import value_fn
from .params import Economy
from .pricing import markups
from .technology import difficulty, draw_economy_shocks, progress


@dataclass(frozen=True)
class Path:
    """Arrays indexed (period, draw, ...)."""

    q: np.ndarray        # (T, D, M) own capability before using open results
    q_eff: np.ndarray    # (T, D, M) usable capability
    m: np.ndarray        # (T, D, M) market share (0 for exited firms)
    active: np.ndarray   # (T, D, M) firm active during period t
    K: np.ndarray        # (T, D, M) capital buffer at end of period t
    H: np.ndarray        # (T, D) public frontier
    S: np.ndarray        # (T, D) new output
    C: np.ndarray        # (T, D) adjustment loss
    R: np.ndarray        # (T, D) R&D spent
    gain: np.ndarray     # (T, D, G) per-capita net gain y_g - y0_g in period t
    # Group totals (not per capita), all >= 0, each (T, D, G):
    #   n_g * gain_g = app + capital + t_in - t_out - adjust - cost
    app: np.ndarray      # application-side gains (1 - tau) S a_g
    capital: np.ndarray  # firm revenue to owners tau v omega
    t_in: np.ndarray     # income transfer received eta S sum_j omega_gj m_j
    t_out: np.ndarray    # income transfer paid eta S delta_g
    adjust: np.ndarray   # adjustment loss d_g C
    cost: np.ndarray     # R&D cost beta r
    tau: np.ndarray      # (T, D, M) pricing share of each firm


def masked_shares(q_eff: np.ndarray, h: np.ndarray, kappa: float, active: np.ndarray) -> np.ndarray:
    z = np.where(active, kappa * (q_eff + h), -np.inf)
    zmax = z.max(axis=-1, keepdims=True)
    zmax = np.where(np.isfinite(zmax), zmax, 0.0)
    w = np.where(active, np.exp(z - zmax), 0.0)
    tot = w.sum(axis=-1, keepdims=True)
    return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)


def run_path(econ: Economy, Z: np.ndarray, owners_fund, K0: float,
             investment=None, active0=None) -> Path:
    """Run T periods for D draws given shocks Z of shape (T, D, M).

    owners_fund: bool for all firms, or a bool array (M,) marking privately funded firms.
      All private or all public: the budget econ.R is split equally among surviving firms
      (or, with an InvestmentRule and private funding, firms choose r_j; model.md §13).
      Mixed (§15): each public firm spends its own policy budget econ.r[j] every period and
      never exits; private firms spend econ.r[j] or, with an InvestmentRule, choose r_j.
    active0: optional bool (M,); firms marked False are absent from the start.
    """
    from .investment import choose_investment
    T, D, M = Z.shape
    funded = np.broadcast_to(np.asarray(owners_fund, bool), (M,))
    mixed = bool(funded.any() and not funded.all())
    q_start = np.broadcast_to(econ.q0, (D, M)).copy()
    H = np.full(D, econ.H0)
    active = np.ones((D, M), bool) if active0 is None else np.broadcast_to(np.asarray(active0, bool), (D, M)).copy()
    K = np.full((D, M), float(K0))
    rec = {k: [] for k in Path.__dataclass_fields__}

    for t in range(T):
        n_act = active.sum(axis=1, keepdims=True)
        if mixed:
            # Public labs spend their own budget; private firms split the private budget
            # equally among surviving private firms (as in §11), or choose with the rule.
            priv_alive = (active & funded).sum(axis=1, keepdims=True)
            private_share = float(econ.r[funded].sum()) / np.maximum(priv_alive, 1)
            policy = np.where(active, np.where(funded, private_share, econ.r), 0.0)
            if investment is not None:
                r_prev = r if t > 0 else np.where(active, econ.r, 0.0)
                chosen = choose_investment(econ, q_start, H, active, np.where(active, r_prev, 0.0), investment,
                                           cap=K if investment.cash_limited else None, fixed=~funded)
                if investment.cash_limited:
                    chosen = np.minimum(chosen, np.maximum(K, 0.0))
                r = np.where(funded, chosen, policy)
            else:
                r = policy
        elif investment is not None and funded.all():
            r_prev = r if t > 0 else np.full((D, M), econ.R / M)
            r = choose_investment(econ, q_start, H, active, np.where(active, r_prev, 0.0), investment,
                                  cap=K if investment.cash_limited else None)
            if investment.cash_limited:
                r = np.minimum(r, np.maximum(K, 0.0))  # cannot spend more than cash on hand
        else:
            r = np.where(active, econ.R / np.maximum(n_act, 1), 0.0)
        step = progress(r, econ.A, econ.alpha, econ.startup_cost)
        if econ.phi != 0:
            step = step * difficulty(q_start, econ.phi)
        if econ.efficiency is not None:
            step = step * econ.efficiency
        q = q_start + step * Z[t]
        newly_open = np.where(active & econ.open, q, -np.inf).max(axis=1)
        H = np.maximum(H, newly_open)
        q_eff = np.where(active, np.maximum(q, H[:, None]), q)
        m = masked_shares(q_eff, econ.h, econ.kappa, active)

        v = m * value_fn(q_eff, econ.theta, econ.gamma)
        commons = np.where(n_act[:, 0] == 0, value_fn(H, econ.theta, econ.gamma), 0.0)
        S = v.sum(axis=1) + commons
        C = econ.chi * S
        tau = markups(econ, q_eff, active, H)
        # Commons value (no firms left) goes entirely to the application side.
        app_total = ((1 - econ.tau) * (S - commons) + commons if econ.pricing == "fixed"
                     else ((1 - tau) * v).sum(axis=1) + commons)
        app = np.outer(app_total, econ.a)
        capital = tau * v @ econ.omega.T
        t_in = econ.eta * S[:, None] * (m @ econ.omega.T)
        # No firms -> no AI owners to receive displaced labour income -> no transfer.
        t_out = np.outer(econ.eta * S * (n_act[:, 0] > 0), econ.delta)
        adjust = np.outer(C, econ.d)
        cost = r @ econ.beta.T
        flows = app + capital + t_in - t_out - adjust - cost
        profit = tau * v - np.where(funded, r, 0.0)  # public firms' R&D is paid by taxes
        K = K + profit

        for k, val in (("q", q), ("q_eff", q_eff), ("m", m), ("active", active.copy()), ("K", K.copy()),
                       ("H", H.copy()), ("S", S), ("C", C), ("R", r.sum(axis=1)),
                       ("gain", flows / econ.n), ("app", app), ("capital", capital),
                       ("t_in", t_in), ("t_out", t_out), ("adjust", adjust), ("cost", cost),
                       ("tau", tau)):
            rec[k].append(val)

        active = active & ((K >= 0) | ~funded)  # only privately funded firms exit
        # Knowledge carried forward: own result plus the absorbed share psi of the gap to H.
        q_start = q_eff if econ.psi == 1.0 else np.where(active, q + econ.psi * np.maximum(H[:, None] - q, 0.0), q)

    return Path(**{k: np.stack(v) for k, v in rec.items()})


def with_eta(p: Path, econ: Economy, eta: float) -> Path:
    """The same path with a different labour-income transfer eta.

    eta enters only the transfer t_in / t_out (and so gain): it changes neither
    technology, prices, profits, exit nor investment. Uses the expressions of run_path,
    so the result equals a fresh run with that eta.
    """
    from dataclasses import replace
    n_act = p.active.sum(axis=2)                                       # (T, D)
    t_in = eta * p.S[..., None] * (p.m @ econ.omega.T)
    t_out = (eta * p.S * (n_act > 0))[..., None] * econ.delta
    flows = p.app + p.capital + t_in - t_out - p.adjust - p.cost
    return replace(p, t_in=t_in, t_out=t_out, gain=flows / econ.n)


def simulate_path(econ: Economy, T: int, n_draws: int, seed: int, owners_fund,
                  K0: float, investment=None, active0=None) -> Path:
    rng = np.random.default_rng(seed)
    Z = draw_economy_shocks(rng, T * n_draws, econ).reshape(T, n_draws, econ.M)
    return run_path(econ, Z, owners_fund, K0, investment, active0)
