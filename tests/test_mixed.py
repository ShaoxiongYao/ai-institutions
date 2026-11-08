"""Mixed institutions: per-firm funding, absent firms, R&D efficiency (model.md §15)."""

import numpy as np
import pytest

from ai_institutions import evaluate
from ai_institutions.dynamics import run_path
from ai_institutions.investment import InvestmentRule

from test_core import base_economy
from test_dynamics import shocks

PRIVATE = np.array([True, True, False])  # firms 0, 1 private; firm 2 public


def test_public_firm_keeps_its_budget_and_never_exits():
    econ = base_economy(sigma=1.0, r=[0.2, 0.2, 0.3])
    p = run_path(econ, shocks(econ, 20, 400), PRIVATE, K0=0.0)
    assert p.active[:, :, 2].all()
    assert (~p.active[:, :, :2]).any()                     # private firms do exit
    spent = p.R - 0.3                                        # public budget every period
    assert np.all(spent >= -1e-12) and np.all(spent <= 0.4 + 1e-12)
    assert np.max(np.abs(p.gain @ econ.n - (p.S - p.C - p.R))) < 1e-10


def test_only_private_firms_follow_the_investment_rule():
    econ = base_economy(sigma=1.0, r=[0.1, 0.1, 0.25], pricing="lead")
    Z = shocks(econ, 10, 200)
    none = run_path(econ, Z, PRIVATE, K0=0.3, investment=InvestmentRule(horizon=0.0, adjust=1.0))
    assert np.allclose(none.R, 0.25)  # private firms invest nothing, public lab spends its budget
    mid = run_path(econ, Z, PRIVATE, K0=0.3, investment=InvestmentRule())
    assert np.max(np.abs(mid.gain @ econ.n - (mid.S - mid.C - mid.R))) < 1e-10
    assert mid.active[:, :, 2].all()


def test_absent_firm_plays_no_role():
    econ = base_economy(sigma=1.0, r=[0.2, 0.2, 0.3], open=[False, False, True])
    p = run_path(econ, shocks(econ, 12, 300), PRIVATE, K0=0.3, active0=[True, True, False])
    assert not p.active[:, :, 2].any()
    assert np.all(p.m[:, :, 2] == 0) and np.allclose(p.R, p.R.clip(max=0.4))
    assert np.all(p.H == econ.H0)  # the absent open lab never publishes
    assert np.max(np.abs(p.gain @ econ.n - (p.S - p.C - p.R))) < 1e-10


def test_efficiency_scales_progress():
    econ = base_economy(sigma=0.0, efficiency=[1.0, 0.5, 0.0])
    out = evaluate(econ, np.ones((1, 3)))
    step = econ.A * econ.r**econ.alpha
    assert np.allclose(out.q[0], econ.q0 + step * np.array([1.0, 0.5, 0.0]))
    with pytest.raises(ValueError):
        base_economy(efficiency=[1.0, -0.1, 1.0])
    with pytest.raises(ValueError):
        base_economy(efficiency=[1.0, 1.0])


def test_scalar_and_uniform_array_funding_agree():
    econ = base_economy(sigma=1.0)
    Z = shocks(econ, 10, 200)
    for flag in (True, False):
        a = run_path(econ, Z, flag, K0=0.2)
        b = run_path(econ, Z, np.full(3, flag), K0=0.2)
        assert np.array_equal(a.gain, b.gain) and np.array_equal(a.active, b.active)


def test_private_budget_is_reallocated_after_exits():
    econ = base_economy(sigma=1.0, r=[0.2, 0.2, 0.3])
    p = run_path(econ, shocks(econ, 20, 400), PRIVATE, K0=0.0)
    one_left = p.active[:, :, :2].sum(axis=2) == 1
    assert one_left.any()
    assert np.allclose(p.R[one_left], 0.4 + 0.3)   # survivor gets the whole private budget


def test_nash_one_round_without_damping_is_a_best_response():
    from ai_institutions.investment import best_response, choose_investment
    econ = base_economy(sigma=1.0, pricing="lead")
    rng = np.random.default_rng(0)
    q, H = rng.uniform(0.2, 2, (50, 3)), rng.uniform(0, 0.5, 50)
    act, r0 = np.ones((50, 3), bool), rng.uniform(0, 0.3, (50, 3))
    a = choose_investment(econ, q, H, act, r0, InvestmentRule(adjust=1.0))
    b = choose_investment(econ, q, H, act, r0, InvestmentRule(rounds=1, damping=1.0))
    assert np.array_equal(a, b)
    assert np.allclose(a, best_response(econ, q, H, act, r0, InvestmentRule()))


def test_nash_respects_cash_and_public_budgets():
    from ai_institutions.investment import NASH_RESIDUALS
    econ = base_economy(sigma=1.0, r=[0.1, 0.1, 0.25], pricing="lead")
    Z = shocks(econ, 8, 150)
    NASH_RESIDUALS.clear()
    p = run_path(econ, Z, PRIVATE, K0=0.05, investment=InvestmentRule(rounds=6, horizon=1e3))
    assert len(NASH_RESIDUALS) == 8
    assert np.all(p.R[0] <= 0.05 * 2 + 0.25 + 1e-12)       # cash caps the private firms
    assert p.active[:, :, 2].all()
    assert np.max(np.abs(p.gain @ econ.n - (p.S - p.C - p.R))) < 1e-10
