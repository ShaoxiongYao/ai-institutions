"""Openness, reuse and market shares (model.md §3-§4). Arrays are (draws, M)."""

from __future__ import annotations

import numpy as np


def public_frontier(q: np.ndarray, is_open: np.ndarray, H0: float) -> np.ndarray:
    """H = max{H0, max_{j open} q_j}, shape (draws,)."""
    masked = np.where(is_open, q, -np.inf)
    return np.maximum(H0, masked.max(axis=-1))


def effective_capability(q: np.ndarray, H: np.ndarray) -> np.ndarray:
    """q~_j = max{q_j, H}."""
    return np.maximum(q, H[..., None])


def market_shares(q_eff: np.ndarray, h: np.ndarray, kappa: float) -> np.ndarray:
    """Logit shares m_j proportional to exp[kappa (q~_j + h_j)]."""
    z = kappa * (q_eff + h)
    z = z - z.max(axis=-1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=-1, keepdims=True)
