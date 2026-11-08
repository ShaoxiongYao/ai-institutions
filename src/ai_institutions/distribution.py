"""Value creation and its distribution across groups (model.md §5-§7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import effective_capability, market_shares, public_frontier
from .params import Economy
from .pricing import markups
from .technology import capability


@dataclass(frozen=True)
class Outcomes:
    """Realized outcomes for a batch of draws. Leading axis is the draw."""

    Z: np.ndarray        # (D, M) shocks
    q: np.ndarray        # (D, M) own capability
    H: np.ndarray        # (D,) public frontier
    q_eff: np.ndarray    # (D, M) usable capability
    m: np.ndarray        # (D, M) market shares
    v: np.ndarray        # (D, M) value created through firm j
    S: np.ndarray        # (D,) total new output
    C: np.ndarray        # (D,) real adjustment loss
    profit: np.ndarray   # (D, M) firm profit tau v_j - r_j
    # Group totals (not per capita), each (D, G); y = y0 + sum / n.
    app: np.ndarray
    capital: np.ndarray
    transfer: np.ndarray
    adjust: np.ndarray
    cost: np.ndarray
    y: np.ndarray        # (D, G) per-capita net resources
    tau: np.ndarray      # (D, M) pricing share of each firm (§12)


def value_fn(q: np.ndarray, theta: float, gamma: float) -> np.ndarray:
    """s(q) = theta q^gamma."""
    return theta * np.maximum(q, 0.0) ** gamma


def evaluate(econ: Economy, Z: np.ndarray) -> Outcomes:
    """Map shocks Z (D, M) to all outcomes, following model.md step by step."""
    Z = np.atleast_2d(Z)
    D = Z.shape[0]

    q = capability(econ.q0, econ.r, Z, econ.A, econ.alpha, econ.startup_cost, econ.phi, econ.efficiency)
    H = public_frontier(q, econ.open, econ.H0)
    q_eff = effective_capability(q, H)
    m = market_shares(q_eff, econ.h, econ.kappa)

    v = m * value_fn(q_eff, econ.theta, econ.gamma)
    S = v.sum(axis=-1)
    C = econ.chi * S
    tau = markups(econ, q_eff, np.ones_like(q_eff, dtype=bool), H)
    profit = tau * v - econ.r

    # Fixed pricing keeps the v0.1 expression so earlier results reproduce bit for bit.
    app_total = (1.0 - econ.tau) * S if econ.pricing == "fixed" else ((1.0 - tau) * v).sum(axis=-1)
    app = np.outer(app_total, econ.a)
    capital = tau * v @ econ.omega.T
    transfer = econ.eta * S[:, None] * (m @ econ.omega.T - econ.delta)
    adjust = -np.outer(C, econ.d)
    cost = -np.broadcast_to(econ.beta @ econ.r, (D, econ.G))

    y = econ.y0 + (app + capital + transfer + adjust + cost) / econ.n
    return Outcomes(Z, q, H, q_eff, m, v, S, C, profit, app, capital, transfer, adjust, cost, y, tau)


def accounting_gap(econ: Economy, out: Outcomes) -> np.ndarray:
    """sum_g n_g y_g - (sum_g n_g y0_g + S - C - R); zero up to rounding (§7)."""
    lhs = out.y @ econ.n
    rhs = econ.n @ econ.y0 + out.S - out.C - econ.R
    return lhs - rhs
