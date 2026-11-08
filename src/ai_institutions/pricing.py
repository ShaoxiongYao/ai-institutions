"""Firm pricing share tau_j (model.md §12).

'fixed': tau_j = tau for every firm (v0.1 default).
'lead':  tau_j rises with firm j's lead over its users' next-best option,
         A_j = max(H, best active rival's q~). Relative lead
         l_j = max(0, (q~_j - A_j) / q~_j), and
         tau_j = tau_min + (tau_max - tau_min) * (1 - exp(-l_j / tau_scale)).
         Everyone at the open frontier -> l_j = 0 -> tau_min.
         A closed monopolist with H = 0 -> l_j = 1 -> close to tau_max.
"""

from __future__ import annotations

import numpy as np

from .params import Economy


def relative_lead(q_eff: np.ndarray, active: np.ndarray, H: np.ndarray) -> np.ndarray:
    """l_j over the last axis (firms); 0 for inactive firms."""
    qa = np.where(active, q_eff, -np.inf)
    if qa.shape[-1] > 1:
        srt = np.sort(qa, axis=-1)
        top1, top2 = srt[..., -1:], srt[..., -2:-1]
        rival = np.where(qa >= top1, top2, top1)  # best active firm other than j
    else:
        rival = np.full_like(qa, -np.inf)
    alt = np.maximum(rival, np.asarray(H)[..., None])
    safe = np.where(q_eff > 0, q_eff, 1.0)
    lead = np.where(q_eff > 0, (q_eff - alt) / safe, 0.0)
    return np.where(active, np.clip(lead, 0.0, 1.0), 0.0)


def markups(econ: Economy, q_eff: np.ndarray, active: np.ndarray, H: np.ndarray) -> np.ndarray:
    """tau_j with the shape of q_eff."""
    if econ.pricing == "fixed":
        return np.full(q_eff.shape, econ.tau)
    lead = relative_lead(q_eff, active, H)
    return econ.tau_min + (econ.tau_max - econ.tau_min) * (1.0 - np.exp(-lead / econ.tau_scale))
