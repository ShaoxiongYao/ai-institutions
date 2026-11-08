"""Technology production and R&D shocks (model.md §2)."""

from __future__ import annotations

import numpy as np


def progress(r, A: float, alpha: float, startup_cost: float = 0.0) -> np.ndarray:
    """Expected progress f(r) = A * max(r - c, 0)^alpha."""
    eff = np.maximum(np.asarray(r, dtype=float) - startup_cost, 0.0)
    return A * eff**alpha


def latent_correlation(sigma: float, rho: float, M: int) -> float:
    """Gaussian-copula correlation rho_X that yields Corr(Z_j, Z_k) = rho."""
    if sigma == 0:
        return 0.0
    s2 = np.log1p(sigma**2)
    arg = 1.0 + rho * np.expm1(s2)
    if arg <= 0:
        raise ValueError(f"rho={rho} infeasible for lognormal shocks with sigma={sigma}")
    rho_x = float(np.log(arg) / s2)
    lower = -1.0 / (M - 1) if M > 1 else -1.0
    if not lower - 1e-12 <= rho_x <= 1 + 1e-12:
        raise ValueError(f"rho={rho} needs latent correlation {rho_x:.3f} outside [{lower:.3f}, 1]")
    return min(max(rho_x, lower), 1.0)


def draw_shocks(rng: np.random.Generator, n_draws: int, M: int, sigma: float, rho: float) -> np.ndarray:
    """Draw Z of shape (n_draws, M): lognormal, mean 1, sd sigma, pairwise corr rho."""
    if sigma == 0:
        return np.ones((n_draws, M))
    s = np.sqrt(np.log1p(sigma**2))
    rho_x = latent_correlation(sigma, rho, M)
    if rho_x >= 0:
        common = rng.standard_normal((n_draws, 1))
        idio = rng.standard_normal((n_draws, M))
        X = np.sqrt(rho_x) * common + np.sqrt(1.0 - rho_x) * idio
    else:
        cov = np.full((M, M), rho_x)
        np.fill_diagonal(cov, 1.0)
        X = rng.multivariate_normal(np.zeros(M), cov, size=n_draws, method="eigh")
    return np.exp(s * X - 0.5 * s**2)


def difficulty(q_start, phi: float) -> np.ndarray:
    """Progress multiplier (1 + q_start)^(-phi) (§14). phi > 0: ideas get harder to find
    as capability rises; phi < 0: standing on shoulders. Exactly 1 when phi = 0."""
    q = np.maximum(np.asarray(q_start, dtype=float), 0.0)
    return np.ones_like(q) if phi == 0 else (1.0 + q) ** (-phi)


def capability(q0, r, Z, A: float, alpha: float, startup_cost: float = 0.0,
               phi: float = 0.0, efficiency=None) -> np.ndarray:
    """Realized capability q_j = q0_j + e_j f(r_j) (1 + q0_j)^(-phi) Z_j; broadcasts over draws."""
    step = progress(r, A, alpha, startup_cost)
    if phi != 0:
        step = step * difficulty(q0, phi)
    if efficiency is not None:
        step = step * efficiency
    return np.asarray(q0) + step * Z


def draw_economy_shocks(rng: np.random.Generator, n_draws: int, econ) -> np.ndarray:
    """Draw Z (n_draws, M) for an Economy, honouring rho_open (§14).

    Pairs of open firms have correlation rho_open; every other pair has rho. With
    rho_open unset (or no two open firms) this is exactly draw_shocks(..., rho).
    """
    M, sigma, rho, rho_open = econ.M, econ.sigma, econ.rho, econ.rho_open
    is_open = np.asarray(econ.open, bool)
    if rho_open is None or rho_open == rho or is_open.sum() < 2:
        return draw_shocks(rng, n_draws, M, sigma, rho)
    if is_open.all():
        return draw_shocks(rng, n_draws, M, sigma, rho_open)
    if sigma == 0:
        return np.ones((n_draws, M))
    s = np.sqrt(np.log1p(sigma**2))
    both = is_open[:, None] & is_open[None, :]

    def latent(c):  # Gaussian-copula correlation for a target lognormal correlation c
        arg = 1.0 + c * np.expm1(s**2)
        if arg <= 0:
            raise ValueError(f"correlation {c} infeasible for lognormal shocks with sigma={sigma}")
        return np.log(arg) / s**2

    cov = np.where(both, latent(rho_open), latent(rho))
    np.fill_diagonal(cov, 1.0)
    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("rho / rho_open give a correlation matrix that is not positive semi-definite")
    X = rng.multivariate_normal(np.zeros(M), cov, size=n_draws, method="eigh")
    return np.exp(s * X - 0.5 * s**2)
