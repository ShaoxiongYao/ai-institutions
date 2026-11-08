import numpy as np
import pytest

from ai_institutions import evaluate
from ai_institutions.dynamics import run_path
from ai_institutions.technology import draw_shocks

from test_core import base_economy


def shocks(econ, T, D, seed=0):
    rng = np.random.default_rng(seed)
    return draw_shocks(rng, T * D, econ.M, econ.sigma, econ.rho).reshape(T, D, econ.M)


def test_accounting_identity_every_period():
    econ = base_economy()
    p = run_path(econ, shocks(econ, 15, 300), owners_fund=True, K0=0.2)
    lhs = p.gain @ econ.n
    assert np.max(np.abs(lhs - (p.S - p.C - p.R))) < 1e-10


def test_first_period_matches_one_period_model():
    econ = base_economy(r=[0.2, 0.2, 0.2])  # dynamics splits R equally
    Z = shocks(econ, 1, 200)
    p = run_path(econ, Z, owners_fund=True, K0=1.0)
    out = evaluate(econ, Z[0])
    assert np.allclose(p.S[0], out.S)
    assert np.allclose(p.gain[0], out.y - econ.y0)


def test_public_funding_no_exit_and_open_knowledge_persists():
    econ = base_economy(open=[True] * 3)
    p = run_path(econ, shocks(econ, 10, 200), owners_fund=False, K0=0.0)
    assert p.active.all()
    assert np.all(np.diff(p.H, axis=0) >= 0)
    assert np.allclose(p.q_eff, p.q_eff.max(axis=2, keepdims=True))  # all at frontier


def test_exited_firms_have_no_share_and_no_budget():
    econ = base_economy(sigma=1.0)
    p = run_path(econ, shocks(econ, 20, 500), owners_fund=True, K0=0.0)
    assert (~p.active).any()
    assert np.all(p.m[~p.active] == 0)
    assert np.all(np.diff(p.active.astype(int), axis=0) <= 0)  # no re-entry


def test_flow_components_add_up_to_gain():
    econ = base_economy()
    p = run_path(econ, shocks(econ, 8, 200), owners_fund=True, K0=0.2)
    total = p.app + p.capital + p.t_in - p.t_out - p.adjust - p.cost
    assert np.allclose(total, p.gain * econ.n)
    for part in (p.app, p.capital, p.t_in, p.t_out, p.adjust, p.cost):
        assert np.all(part >= -1e-12)


def test_lead_pricing_accounting_and_bounds():
    econ = base_economy(pricing="lead", sigma=1.0)
    p = run_path(econ, shocks(econ, 15, 300), owners_fund=True, K0=0.0)
    assert np.max(np.abs(p.gain @ econ.n - (p.S - p.C - p.R))) < 1e-10
    act = p.active
    assert np.all((p.tau[act] >= econ.tau_min - 1e-12) & (p.tau[act] <= econ.tau_max + 1e-12))


def test_commons_when_all_firms_exit():
    # All open, tiny margins, owners pay: firms exit, open knowledge keeps serving users.
    econ = base_economy(pricing="lead", open=[True] * 3, tau_min=0.0, tau_max=0.0, eta=0.3)
    p = run_path(econ, shocks(econ, 10, 50), owners_fund=True, K0=0.0)
    gone = p.active.sum(axis=2) == 0
    assert gone.any()
    from ai_institutions.distribution import value_fn
    assert np.allclose(p.S[gone], value_fn(p.H[gone], econ.theta, econ.gamma))
    assert np.max(np.abs(p.gain @ econ.n - (p.S - p.C - p.R))) < 1e-10


def test_investment_rule_limits_and_accounting():
    from ai_institutions.investment import InvestmentRule, shock_nodes
    z = shock_nodes(1.0, 16)
    assert np.isclose(z.mean(), 1.0) and np.all(z > 0)
    econ = base_economy(pricing="lead", sigma=1.0)
    Z = shocks(econ, 10, 200)
    none = run_path(econ, Z, owners_fund=True, K0=0.3, investment=InvestmentRule(horizon=0.0, adjust=1.0))
    assert np.allclose(none.R, 0.0)  # no payoff weight -> invest nothing
    top = max(InvestmentRule().grid)
    huge = run_path(econ, Z, owners_fund=True, K0=1e9, investment=InvestmentRule(horizon=1e6, adjust=1.0))
    assert np.allclose(huge.R[0], top * econ.M)  # dominant payoff -> top of grid for every firm
    slow = run_path(econ, Z, owners_fund=True, K0=0.3, investment=InvestmentRule(horizon=0.0, adjust=0.5, cash_limited=False))
    assert np.allclose(slow.R[1], 0.5 * slow.R[0])  # partial adjustment halves the gap each period
    mid = run_path(econ, Z, owners_fund=True, K0=0.3, investment=InvestmentRule())
    assert np.max(np.abs(mid.gain @ econ.n - (mid.S - mid.C - mid.R))) < 1e-10
    capped = run_path(econ, Z, owners_fund=True, K0=0.3, investment=InvestmentRule(horizon=1e6))
    assert np.all(capped.R[0] <= 0.3 * econ.M + 1e-12)  # cash on hand caps spending
    assert capped.active.all()  # spending within cash never drives capital below zero
    pub = run_path(econ, Z, owners_fund=False, K0=0.3, investment=InvestmentRule())
    assert np.allclose(pub.R, econ.R)  # public budget ignores the private rule


@pytest.mark.parametrize("phi, eff", [(0.0, None), (0.7, None), (-0.3, [1.0, 0.4, 0.8])])
def test_expected_payoff_matches_brute_force(phi, eff):
    from ai_institutions.distribution import value_fn
    from ai_institutions.dynamics import masked_shares
    from ai_institutions.investment import InvestmentRule, expected_payoff, shock_nodes
    from ai_institutions.pricing import markups
    from ai_institutions.technology import progress
    econ = base_economy(pricing="lead", sigma=0.8, open=[True, False, False], phi=phi, efficiency=eff)
    rng = np.random.default_rng(3)
    D, M = 3, econ.M
    q_start = rng.uniform(0.2, 2.0, (D, M))
    H = rng.uniform(0.0, 0.5, D)
    active = np.array([[True, True, True], [True, False, True], [True, True, True]])
    r_prev = rng.uniform(0, 0.2, (D, M))
    rule = InvestmentRule(nodes=5)
    pay = expected_payoff(econ, q_start, H, active, r_prev, rule)
    z = shock_nodes(econ.sigma, rule.nodes)
    f = lambda r: progress(r, econ.A, econ.alpha, econ.startup_cost)  # noqa: E731
    for d in range(D):
        for j in range(M):
            for g, rg in enumerate(rule.grid):
                revs = []
                for zk in z:
                    ease = (1 + q_start[d]) ** (-phi) * (1.0 if eff is None else np.array(eff))
                    q = q_start[d] + f(r_prev[d]) * ease
                    q[j] = q_start[d, j] + f(rg) * ease[j] * zk
                    Hn = max(H[d], max([q[i] for i in range(M) if active[d, i] and econ.open[i]], default=-np.inf))
                    qe = np.where(active[d], np.maximum(q, Hn), q)
                    m = masked_shares(qe[None], econ.h, econ.kappa, active[d][None])[0]
                    v = m * value_fn(qe, econ.theta, econ.gamma)
                    tau = markups(econ, qe[None], active[d][None], np.array([Hn]))[0]
                    revs.append(tau[j] * v[j])
                assert np.isclose(pay[d, j, g], rule.horizon * np.mean(revs) - rg)


@pytest.mark.parametrize("owners_fund", [True, False])
def test_with_eta_matches_a_fresh_run(owners_fund):
    from ai_institutions.dynamics import with_eta
    econ = base_economy(sigma=1.0, eta=0.0, pricing="lead")
    Z = shocks(econ, 12, 300)
    p0 = run_path(econ, Z, owners_fund, K0=0.1)
    for eta in (0.0, 0.3):
        a = with_eta(p0, econ.with_(eta=eta), eta)
        b = run_path(econ.with_(eta=eta), Z, owners_fund, K0=0.1)
        for f in ("t_in", "t_out", "gain", "S", "active"):
            assert np.allclose(getattr(a, f), getattr(b, f), rtol=0, atol=1e-12), f
