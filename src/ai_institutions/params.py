"""Model parameters. Symbols follow docs/model.md."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

_TOL = 1e-9


def _arr(x, dtype=float) -> np.ndarray:
    a = np.array(x, dtype=dtype)
    a.setflags(write=False)
    return a


def _check_shares(name: str, x: np.ndarray, axis: int = 0) -> None:
    if np.any(x < -_TOL):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(x.sum(axis=axis), 1.0, atol=1e-8):
        raise ValueError(f"{name} must sum to 1 over groups, got {x.sum(axis=axis)}")


@dataclass(frozen=True)
class Economy:
    """One-period economy with G groups and M R&D projects (model.md §1-§6)."""

    # Groups (§1, §6): arrays indexed by g, or (g, j).
    n: np.ndarray            # population mass, sums to 1
    y0: np.ndarray           # initial per-capita resources
    a: np.ndarray            # share of application-side gains
    omega: np.ndarray        # (G, M) ownership of firm j
    beta: np.ndarray         # (G, M) share of project j's R&D cost
    d: np.ndarray            # share of adjustment loss C
    delta: np.ndarray        # source shares of income transfer

    # Projects (§2-§4): arrays indexed by j.
    r: np.ndarray            # R&D input
    q0: np.ndarray           # starting capability
    h: np.ndarray            # non-technical advantage
    open: np.ndarray         # bool, results reusable by others

    # Technology (§2).
    A: float = 1.0
    alpha: float = 0.5
    startup_cost: float = 0.0
    sigma: float = 0.0       # sd of Z
    rho: float = 0.0         # Corr(Z_j, Z_k)

    # Openness (§3), market (§4), value (§5), distribution (§6).
    H0: float = 0.0
    kappa: float = 1.0
    theta: float = 1.0
    gamma: float = 1.0
    chi: float = 0.0
    tau: float = 0.3
    eta: float = 0.0

    # Pricing share (§12): 'fixed' uses tau; 'lead' uses tau_min, tau_max, tau_scale.
    pricing: str = "fixed"
    tau_min: float = 0.1
    tau_max: float = 0.7
    tau_scale: float = 0.2

    # Spillover frictions (§14). Defaults reproduce the frictionless model exactly.
    psi: float = 1.0                 # absorption: share of the gap to H carried into next period
    rho_open: float | None = None    # Corr(Z_j, Z_k) between two open firms; None -> rho
    phi: float = 0.0                 # difficulty: progress scaled by (1 + q_start)^(-phi)

    # Mixed institutions (§15): per-project R&D efficiency, progress scaled by efficiency_j.
    efficiency: np.ndarray | None = None  # None -> 1 for every project

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "open":
                object.__setattr__(self, f.name, _arr(v, bool))
            elif isinstance(v, (list, tuple, np.ndarray)):
                object.__setattr__(self, f.name, _arr(v))
        self._validate()

    @property
    def G(self) -> int:
        return self.n.shape[0]

    @property
    def M(self) -> int:
        return self.r.shape[0]

    @property
    def R(self) -> float:
        return float(self.r.sum())

    def _validate(self) -> None:
        G, M = self.G, self.M
        for name in ("y0", "a", "d", "delta"):
            if getattr(self, name).shape != (G,):
                raise ValueError(f"{name} must have shape ({G},)")
        for name in ("omega", "beta"):
            if getattr(self, name).shape != (G, M):
                raise ValueError(f"{name} must have shape ({G}, {M})")
        for name in ("q0", "h", "open"):
            if getattr(self, name).shape != (M,):
                raise ValueError(f"{name} must have shape ({M},)")
        if np.any(self.n <= 0):
            raise ValueError("n must be positive")
        for name in ("n", "a", "d", "delta"):
            _check_shares(name, getattr(self, name))
        _check_shares("omega", self.omega)
        _check_shares("beta", self.beta)
        if np.any(self.r < 0) or np.any(self.q0 < 0):
            raise ValueError("r and q0 must be non-negative")
        if not self.alpha > 0:
            raise ValueError("alpha must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        if self.pricing not in ("fixed", "lead"):
            raise ValueError("pricing must be 'fixed' or 'lead'")
        if not 0 <= self.tau_min <= self.tau_max <= 1 or self.tau_scale <= 0:
            raise ValueError("need 0 <= tau_min <= tau_max <= 1 and tau_scale > 0")
        if not 0 <= self.chi < 1:
            raise ValueError("chi must be in [0, 1)")
        if self.efficiency is not None:
            if self.efficiency.shape != (M,) or np.any(self.efficiency < 0):
                raise ValueError(f"efficiency must have shape ({M},) and be non-negative")
        if not 0 <= self.psi <= 1:
            raise ValueError("psi must be in [0, 1]")
        if self.rho_open is not None and not -1 <= self.rho_open <= 1:
            raise ValueError("rho_open must be in [-1, 1] or None")
        if not np.isfinite(self.phi):
            raise ValueError("phi must be finite")
        if self.sigma < 0 or self.startup_cost < 0 or self.kappa < 0:
            raise ValueError("sigma, startup_cost, kappa must be non-negative")

    def with_(self, **changes) -> "Economy":
        """Return a copy with some parameters replaced (re-validated)."""
        return replace(self, **changes)


def private_financing(omega) -> np.ndarray:
    """Owners pay for the projects they own: beta_gj = omega_gj (§6)."""
    return np.array(omega, dtype=float)


def public_financing(tax_shares, M: int) -> np.ndarray:
    """Groups pay by tax share regardless of ownership: beta_gj = t_g (§6)."""
    t = np.asarray(tax_shares, dtype=float)
    return np.repeat(t[:, None], M, axis=1)
