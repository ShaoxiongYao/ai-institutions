"""Checks listed in docs/model.md §9."""

import numpy as np
import pytest

from ai_institutions import Economy, accounting_gap, evaluate, private_financing, public_financing, simulate, summarize
from ai_institutions.technology import draw_shocks


def base_economy(**changes) -> Economy:
    """Three groups (investors, workers, users) and three projects."""
    omega = [[0.8, 0.5, 0.2], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]]
    econ = Economy(
        n=[0.1, 0.6, 0.3], y0=[5.0, 1.0, 1.5],
        a=[0.2, 0.3, 0.5], omega=omega, beta=private_financing(omega),
        d=[0.0, 0.8, 0.2], delta=[0.0, 0.9, 0.1],
        r=[1.0, 0.5, 0.2], q0=[0.5, 0.3, 0.1], h=[0.2, 0.0, -0.1], open=[False, True, False],
        A=1.2, alpha=0.6, sigma=0.8, rho=0.3, kappa=2.0, theta=1.5, gamma=0.8,
        chi=0.1, tau=0.35, eta=0.2,
    )
    return econ.with_(**changes) if changes else econ


@pytest.mark.parametrize("seed", range(5))
def test_accounting_identity_random_parameters(seed):
    rng = np.random.default_rng(seed)
    G, M = 4, 3

    def shares(*shape):
        x = rng.random(shape)
        return x / x.sum(axis=0)

    econ = Economy(
        n=shares(G), y0=rng.random(G) * 3, a=shares(G), omega=shares(G, M), beta=shares(G, M),
        d=shares(G), delta=shares(G), r=rng.random(M) * 2, q0=rng.random(M),
        h=rng.normal(size=M), open=rng.random(M) < 0.5,
        A=rng.uniform(0.5, 2), alpha=rng.uniform(0.2, 1.5), startup_cost=rng.uniform(0, 0.5),
        sigma=rng.uniform(0, 1.5), rho=rng.uniform(0, 0.8), H0=rng.uniform(0, 0.5),
        kappa=rng.uniform(0, 5), theta=rng.uniform(0.5, 2), gamma=rng.uniform(0.3, 1),
        chi=rng.uniform(0, 0.5), tau=rng.uniform(0, 1), eta=rng.uniform(0, 0.5),
    )
    out = simulate(econ, n_draws=2_000, seed=seed)
    assert np.max(np.abs(accounting_gap(econ, out))) < 1e-10
    assert np.allclose(out.m.sum(axis=1), 1.0)
    assert np.allclose(out.transfer.sum(axis=1), 0.0)


def test_zero_investment_no_capability_leaves_everyone_at_y0():
    econ = base_economy(r=[0.0, 0.0, 0.0], q0=[0.0, 0.0, 0.0], H0=0.0)
    out = simulate(econ, n_draws=500, seed=1)
    assert np.allclose(out.q, 0.0)
    assert np.allclose(out.S, 0.0)
    assert np.allclose(out.y, econ.y0)


def test_zero_investment_keeps_starting_capability():
    econ = base_economy(r=[0.0, 0.0, 0.0])
    out = simulate(econ, n_draws=500, seed=1)
    assert np.allclose(out.q, econ.q0)


def test_deterministic_limit():
    econ = base_economy(sigma=0.0)
    out = simulate(econ, n_draws=50, seed=2)
    expected_q = econ.q0 + econ.A * econ.r**econ.alpha
    assert np.allclose(out.q, expected_q)
    assert np.allclose(out.y, out.y[0])


def test_startup_cost_blocks_progress_but_still_costs():
    econ = base_economy(sigma=0.0, startup_cost=0.6, eta=0.0, chi=0.0)
    out = simulate(econ, n_draws=1, seed=0)
    assert np.allclose(out.q[0, 1:], econ.q0[1:])          # r < c: no progress
    assert out.q[0, 0] > econ.q0[0]                         # r > c: progress
    assert np.allclose(out.cost[0], -(econ.beta @ econ.r))  # inputs still paid


def test_pure_transfer_leaves_linear_welfare_unchanged():
    econ = base_economy()
    alt_omega = [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3]]
    alt = econ.with_(
        a=[0.6, 0.2, 0.2], omega=alt_omega, beta=public_financing([0.3, 0.3, 0.4], econ.M),
        d=[0.5, 0.25, 0.25], delta=[0.4, 0.3, 0.3],
    )
    s0 = summarize(econ, simulate(econ, 5_000, seed=3))
    s1 = summarize(alt, simulate(alt, 5_000, seed=3))
    assert s0["welfare"][0] == pytest.approx(s1["welfare"][0], abs=1e-10)
    assert not np.allclose(s0["group_y_mean"], s1["group_y_mean"])
    # Concave utility does distinguish the two.
    c0 = summarize(econ, simulate(econ, 5_000, seed=3), kind="cara", param=1.0)
    c1 = summarize(alt, simulate(alt, 5_000, seed=3), kind="cara", param=1.0)
    assert abs(c0["welfare"][0] - c1["welfare"][0]) > 1e-3


@pytest.mark.parametrize("sigma,rho", [(0.5, 0.0), (1.0, 0.5), (0.8, -0.2)])
def test_shock_calibration(sigma, rho):
    Z = draw_shocks(np.random.default_rng(4), 400_000, 3, sigma, rho)
    assert Z.min() >= 0
    assert Z.mean() == pytest.approx(1.0, abs=0.01)
    assert Z.std(axis=0).mean() == pytest.approx(sigma, rel=0.03)
    corr = np.corrcoef(Z.T)[np.triu_indices(3, 1)]
    assert np.allclose(corr, rho, atol=0.02)


def test_infeasible_negative_correlation_rejected():
    with pytest.raises(ValueError):
        draw_shocks(np.random.default_rng(0), 10, 5, 1.0, -0.5)


def test_opening_frontier_raises_usable_capability_and_value():
    closed = base_economy(sigma=0.0, open=[False, False, False])
    opened = closed.with_(open=[True, False, False])  # project 0 is the frontier
    oc, oo = evaluate(closed, np.ones((1, 3))), evaluate(opened, np.ones((1, 3)))
    assert np.allclose(oo.q_eff, oo.q.max())
    assert oo.S[0] > oc.S[0]


def test_full_closure_high_kappa_approaches_frontier_value():
    econ = base_economy(sigma=0.0, open=[False] * 3, h=[0.0] * 3, kappa=200.0)
    out = evaluate(econ, np.ones((1, 3)))
    assert out.S[0] == pytest.approx(econ.theta * out.q.max() ** econ.gamma, rel=1e-6)


def test_independent_trials_success_probability_independent_of_M():
    """Claim 3 in the plan: with p(r)=1-exp(-lambda r), P(at least one) = 1-exp(-lambda R)."""
    lam, R = 0.7, 3.0
    for M in (1, 2, 5, 50):
        p = 1 - np.exp(-lam * R / M)
        assert 1 - (1 - p) ** M == pytest.approx(1 - np.exp(-lam * R))


def test_lead_pricing_open_frontier_and_monopoly():
    from ai_institutions.pricing import relative_lead
    q = np.array([[2.0, 2.0, 2.0], [3.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    act = np.array([[True] * 3, [True] * 3, [True, False, False]])
    lead = relative_lead(q, act, H=np.array([2.0, 0.0, 0.0]))
    assert np.allclose(lead[0], 0.0)                 # everyone at the open frontier
    assert np.allclose(lead[1], [2 / 3, 0.0, 0.0])   # leader vs best rival
    assert np.allclose(lead[2, 0], 1.0)              # closed monopoly, H = 0


def test_lead_pricing_accounting_one_period():
    econ = base_economy(pricing="lead")
    out = simulate(econ, n_draws=2_000, seed=7)
    assert np.max(np.abs(accounting_gap(econ, out))) < 1e-10
    opened = econ.with_(open=[True] * 3, sigma=0.0)
    o = evaluate(opened, np.ones((1, 3)))
    assert np.allclose(o.tau, opened.tau_min)
