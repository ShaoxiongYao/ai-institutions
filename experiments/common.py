"""Shared baseline, result saving and plot style for experiments.

Baseline parameters are illustrative, chosen to exercise the mechanisms; they are
not calibrated to any real economy.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ai_institutions import Economy, run_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

GROUPS = ("investors", "workers", "others")
N = np.array([0.1, 0.6, 0.3])
Y0 = np.array([4.0, 1.0, 2.0])

# Ownership / financing building blocks (rows = groups).
OWN_CONCENTRATED = np.array([1.0, 0.0, 0.0])
OWN_BROAD = N.copy()
TAX_SHARES = N * Y0 / (N * Y0).sum()  # proportional income tax


def baseline(M: int = 3, **changes) -> Economy:
    """Three groups, three asymmetric projects, concentrated private ownership."""
    omega = np.repeat(OWN_CONCENTRATED[:, None], M, axis=1)
    params = dict(
        n=N, y0=Y0,
        a=[0.2, 0.4, 0.4], omega=omega, beta=omega,
        d=[0.0, 0.8, 0.2], delta=[0.0, 0.8, 0.2],
        r=[0.15, 0.10, 0.05], q0=[0.2, 0.15, 0.1], h=[0.1, 0.0, 0.0],
        open=[False, False, False],
        A=1.0, alpha=0.6, sigma=0.8, rho=0.3, startup_cost=0.0, H0=0.0,
        kappa=3.0, theta=1.2, gamma=1.0, chi=0.1, tau=0.5, eta=0.0,
    )
    params.update(changes)
    return Economy(**params)


def econ_dict(econ: Economy) -> dict:
    out = {}
    for f in fields(econ):
        v = getattr(econ, f.name)
        out[f.name] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def save(name: str, config: dict, rows: list[dict], seed: int, n_draws: int,
         table_md: str) -> Path:
    """Write summary.json (config, run info, rows) and table.md under results/<name>/."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": name, "run_info": run_info(seed, n_draws), "config": config, "rows": rows}
    (d / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (d / "table.md").write_text(table_md + "\n")
    return d


def md_table(header: list[str], body: list[list]) -> str:
    def fmt(x):
        return f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in body]
    return "\n".join(lines)


# Plot style: reference palette (light mode), first four categorical slots.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.markersize": 5,
        "font.size": 9, "text.color": INK,
    })
