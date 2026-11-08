"""Spillover frictions psi, rho_open, phi (model.md §14)."""

import numpy as np
import pytest

from ai_institutions import evaluate
from ai_institutions.dynamics import run_path
from ai_institutions.technology import draw_economy_shocks, draw_shocks

from test_core import base_economy
from test_dynamics import shocks


def test_defaults_reproduce_frictionless_model():
    econ = base_economy()
    same = econ.with_(psi=1.0, rho_open=None, phi=0.0)
    Z = shocks(econ, 6, 200)
    a, b = run_path(econ, Z, True, 0.2), run_path(same, Z, True, 0.2)
    assert np.array_equal(a.q_eff, b.q_eff) and np.array_equal(a.gain, b.gain)
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    assert np.array_equal(draw_economy_shocks(rng1, 100, econ),
                          draw_shocks(rng2, 100, econ.M, econ.sigma, econ.rho))


def test_no_absorption_open_frontier_is_running_max_of_closed_paths():
    # psi = 0: firms still serve H, but each builds only on its own results, so knowledge
    # evolves exactly as under closed R&D and H is the running maximum of those paths.
    closed = base_economy(open=[False] * 3, psi=0.0)
    opened = closed.with_(open=[True] * 3)
    Z = shocks(closed, 10, 300)
    a, b = run_path(closed, Z, False, 0.0), run_path(opened, Z, False, 0.0)
    assert np.allclose(a.q, b.q)
    assert np.allclose(b.H, np.maximum.accumulate(a.q.max(axis=2), axis=0))
    assert np.all(b.q_eff >= a.q_eff - 1e-12)  # users still gain from open results


def test_partial_absorption_carries_a_share_of_the_gap():
    econ = base_economy(open=[True] * 3, psi=0.3, sigma=0.8)
    Z = shocks(econ, 12, 200)
    p = run_path(econ, Z, False, 0.0)
    step = econ.A * (econ.R / econ.M) ** econ.alpha
    q_start = p.q[:-1] + 0.3 * np.maximum(p.H[:-1, :, None] - p.q[:-1], 0)
    assert np.allclose(p.q[1:], q_start + step * Z[1:])
    assert np.allclose(p.q_eff, p.H[..., None])        # everyone serves the open frontier now
    one = evaluate(econ, Z[0])                          # one-period model: psi plays no role
    assert np.allclose(one.S, evaluate(econ.with_(psi=1.0), Z[0]).S)
    assert np.max(np.abs(p.gain @ econ.n - (p.S - p.C - p.R))) < 1e-10


@pytest.mark.parametrize("mask", [[True] * 3, [True, True, False]])
def test_rho_open_calibration(mask):
    econ = base_economy(open=mask, sigma=1.0, rho=0.2, rho_open=0.7)
    Z = draw_economy_shocks(np.random.default_rng(0), 400_000, econ)
    C = np.corrcoef(Z.T)
    assert np.allclose(Z.mean(axis=0), 1.0, atol=0.02)
    assert abs(C[0, 1] - 0.7) < 0.03
    if not all(mask):
        assert abs(C[0, 2] - 0.2) < 0.03 and abs(C[1, 2] - 0.2) < 0.03


def test_higher_rho_open_lowers_open_frontier():
    base = base_economy(open=[True] * 3, sigma=1.0, rho=0.2)
    T, D = 15, 2000
    front = []
    for c in (0.2, 0.9):
        econ = base.with_(rho_open=c)
        Z = draw_economy_shocks(np.random.default_rng(1), T * D, econ).reshape(T, D, 3)
        front.append(run_path(econ, Z, False, 0.0).H[-1].mean())
    assert front[1] < front[0]  # best-of-M search is worth less when results are correlated


def test_difficulty_scales_progress():
    econ = base_economy(phi=0.5, sigma=0.0)
    out = evaluate(econ, np.ones((1, 3)))
    step = econ.A * econ.r**econ.alpha * (1 + econ.q0) ** -0.5
    assert np.allclose(out.q[0], econ.q0 + step)
    harder = run_path(econ, shocks(econ, 10, 100), False, 0.0)
    easier = run_path(econ.with_(phi=-0.5), shocks(econ, 10, 100), False, 0.0)
    assert np.all(harder.H[-1] <= easier.H[-1]) and harder.q_eff[-1].mean() < easier.q_eff[-1].mean()


def test_parameter_validation():
    for bad in ({"psi": 1.5}, {"psi": -0.1}, {"rho_open": 2.0}, {"phi": np.inf}):
        with pytest.raises(ValueError):
            base_economy(**bad)
