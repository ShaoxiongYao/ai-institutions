"""Utility, welfare and summary statistics (model.md §8)."""

from __future__ import annotations

import numpy as np

from .distribution import Outcomes
from .params import Economy


def utility(y: np.ndarray, kind: str = "linear", param: float = 1.0, floor: float = 1e-6) -> np.ndarray:
    """u(y): 'linear'; 'cara' with absolute risk aversion param; 'crra' with RRA param and floor."""
    if kind == "linear":
        return np.asarray(y, dtype=float)
    if kind == "cara":
        return -np.exp(-param * y) / param
    if kind == "crra":
        yc = np.maximum(y, floor)
        return np.log(yc) if param == 1 else yc ** (1 - param) / (1 - param)
    raise ValueError(f"unknown utility kind {kind!r}")


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0
    return float(x.mean()), float(se)


def summarize(econ: Economy, out: Outcomes, kind: str = "linear", param: float = 1.0,
              floor: float = 1e-6) -> dict:
    """Summary dict. Each scalar is reported as (mean, Monte Carlo SE) unless noted."""
    W_draw = utility(out.y, kind, param, floor) @ econ.n
    total = out.y @ econ.n
    losers = (out.y < econ.y0 - 1e-12).astype(float) @ econ.n
    summary = {
        "welfare": _mean_se(W_draw),
        "total_resources": _mean_se(total),
        "total_sd": float(total.std(ddof=1)) if total.size > 1 else 0.0,
        "S": _mean_se(out.S),
        "C": _mean_se(out.C),
        "R": econ.R,
        "H": _mean_se(out.H),
        "loser_share": _mean_se(losers),
        "firm_loss_prob": (out.profit < 0).mean(axis=0).tolist(),
        "group_y_mean": out.y.mean(axis=0).tolist(),
        "group_y_p05": np.quantile(out.y, 0.05, axis=0).tolist(),
        "group_y_p95": np.quantile(out.y, 0.95, axis=0).tolist(),
    }
    if kind == "crra":
        summary["floor_hit_share"] = _mean_se((out.y < floor).astype(float) @ econ.n)
    return summary
