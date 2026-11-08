"""Endogenous private R&D investment (model.md §13). Draft.

Each period, every surviving privately funded firm j picks r_j from a grid to maximise

    L * E_Z[ tau_j v_j | r_j, rivals ] - r_j,

where the expectation is over firm j's own R&D shock (lognormal quadrature nodes),
rivals are assumed to repeat last period's investment and reach their expected
capability, and L ("payoff horizon") roughly counts the periods over which a
capability gain keeps earning revenue. Firms move only part of the way to that best
response each period (partial adjustment), which removes the synchronised two-period
cycle that pure adaptive expectations produce. This is a myopic, adaptive rule, not a
dynamic equilibrium.

With rounds > 0 (model.md §13, item 4) firms instead play an approximate Nash equilibrium of
each period's investment game: starting from last period's profile, every firm best-responds
to the rivals' current choices, damped, for `rounds` iterations (adjust is then unused). The
payoff is still the reduced-form L * E[tau v] - r, so this is a per-period equilibrium, not a
dynamic one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .distribution import value_fn
from .params import Economy
from .pricing import markups
from .technology import difficulty, progress


@dataclass(frozen=True)
class InvestmentRule:
    grid: tuple = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6)
    horizon: float = 5.0   # L
    nodes: int = 16        # quadrature nodes for the firm's own shock
    adjust: float = 0.5    # r_t = r_{t-1} + adjust * (best response - r_{t-1}); 1 = no inertia
    cash_limited: bool = True  # r_j <= cash on hand K_j (no external finance)
    rounds: int = 0        # > 0: per-period Nash by damped best-response iteration
    damping: float = 0.5   # step towards the best response in each round


# Diagnostics of the per-period equilibrium iteration: for each call, the mean and the share
# (over active firms and draws) of the last round's |best response - current choice|.
NASH_RESIDUALS: list[tuple[float, float]] = []


def shock_nodes(sigma: float, K: int) -> np.ndarray:
    """Equal-weight nodes for a mean-1 lognormal with sd sigma."""
    if sigma == 0:
        return np.ones(1)
    s = np.sqrt(np.log1p(sigma**2))
    u = np.array([NormalDist().inv_cdf((k + 0.5) / K) for k in range(K)])
    z = np.exp(s * u - 0.5 * s**2)
    return z / z.mean()


def _shares(q_eff, h, kappa, active):
    z = np.where(active, kappa * (q_eff + h), -np.inf)
    zmax = z.max(axis=-1, keepdims=True)
    zmax = np.where(np.isfinite(zmax), zmax, 0.0)
    w = np.where(active, np.exp(z - zmax), 0.0)
    tot = w.sum(axis=-1, keepdims=True)
    return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)


def expected_payoff(econ: Economy, q_start: np.ndarray, H: np.ndarray, active: np.ndarray,
                    r_prev: np.ndarray, rule: InvestmentRule) -> np.ndarray:
    """L * E[tau_j v_j] - r for every firm j and grid option; shape (D, M, G)."""
    D, M = q_start.shape
    grid = np.asarray(rule.grid, dtype=float)
    z = shock_nodes(econ.sigma, rule.nodes)
    f_opt = progress(grid, econ.A, econ.alpha, econ.startup_cost)          # (G,)
    ease = difficulty(q_start, econ.phi)                                   # (D, M), 1 if phi = 0
    if econ.efficiency is not None:
        ease = ease * econ.efficiency
    q_bar = q_start + progress(r_prev, econ.A, econ.alpha, econ.startup_cost) * ease  # rivals, (D, M)

    # Q[d, j, g, k, i]: capability of firm i when firm j picks option g and draws node k.
    Q = np.broadcast_to(q_bar[:, None, None, None, :], (D, M, grid.size, z.size, M)).copy()
    own = q_start[:, :, None, None] + (f_opt[None, None, :, None] * ease[:, :, None, None]) * z[None, None, None, :]
    idx = np.arange(M)
    Q[:, idx, :, :, idx] = np.moveaxis(own, 1, 0)                          # set diagonal j == i

    act = active[:, None, None, None, :]
    newly_open = np.where(act & econ.open, Q, -np.inf).max(axis=-1)
    Hn = np.maximum(H[:, None, None, None], newly_open)                    # (D, M, G, K)
    q_eff = np.where(act, np.maximum(Q, Hn[..., None]), Q)
    m = _shares(q_eff, econ.h, econ.kappa, act)
    v = m * value_fn(q_eff, econ.theta, econ.gamma)
    tau = markups(econ, q_eff, np.broadcast_to(act, q_eff.shape), Hn)
    rev = (tau * v)[:, idx, :, :, idx]                                     # (M, D, G, K)
    rev = np.moveaxis(rev, 0, 1).mean(axis=-1)                             # (D, M, G)
    return rule.horizon * rev - grid


def best_response(econ: Economy, q_start, H, active, r_rivals, rule: InvestmentRule,
                  chunk: int = 500) -> np.ndarray:
    """Grid best response of every firm to rivals playing r_rivals, (D, M). Chunked over draws."""
    D = q_start.shape[0]
    pay = np.concatenate([expected_payoff(econ, q_start[i:i + chunk], H[i:i + chunk], active[i:i + chunk],
                                          r_rivals[i:i + chunk], rule) for i in range(0, D, chunk)])
    return np.asarray(rule.grid, dtype=float)[pay.argmax(axis=-1)]


def choose_investment(econ: Economy, q_start, H, active, r_prev, rule: InvestmentRule,
                      chunk: int = 500, cap=None, fixed=None) -> np.ndarray:
    """Chosen r_j (D, M); 0 for exited firms.

    cap: optional (D, M) upper bound (cash on hand), applied inside the equilibrium iteration.
    fixed: optional (M,) bool; those firms (public labs) keep r_prev in the equilibrium iteration.
    """
    if rule.rounds <= 0:
        best = best_response(econ, q_start, H, active, r_prev, rule, chunk)
        r = r_prev + rule.adjust * (best - r_prev)
        return np.where(active, r, 0.0)
    r = np.where(active, r_prev, 0.0)
    for _ in range(rule.rounds):
        best = best_response(econ, q_start, H, active, r, rule, chunk)
        if cap is not None:
            best = np.minimum(best, np.maximum(cap, 0.0))
        if fixed is not None:
            best = np.where(fixed, r, best)
        step = best - r
        r = np.where(active, r + rule.damping * step, 0.0)
    moving = active if fixed is None else active & ~np.asarray(fixed, bool)
    resid = np.abs(step)[moving]
    NASH_RESIDUALS.append((float(resid.mean()) if resid.size else 0.0,
                           float((resid > 0.0125).mean()) if resid.size else 0.0))
    return r
