"""Monte Carlo driver and run metadata."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .distribution import Outcomes, evaluate
from .params import Economy
from .technology import draw_economy_shocks


def simulate(econ: Economy, n_draws: int = 10_000, seed: int = 0) -> Outcomes:
    """Draw shocks with a fixed seed and evaluate the economy."""
    rng = np.random.default_rng(seed)
    Z = draw_economy_shocks(rng, n_draws, econ)
    return evaluate(econ, Z)


def run_info(seed: int, n_draws: int) -> dict:
    """Metadata to store with every result: seed, draws, code version, time."""
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "seed": seed,
        "n_draws": n_draws,
        "git_commit": git("rev-parse", "HEAD"),
        # Code only: generated outputs under results/ (tracked or not) do not mark the run dirty.
        "git_dirty": git("status", "--porcelain", "--untracked-files=no", "--", ".",
                         ":(exclude)results") not in ("", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
