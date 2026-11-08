"""Run all v0.1 experiments. Usage: uv run python -m experiments.run_all"""

from . import distribution, openness, rnd

if __name__ == "__main__":
    for mod in (distribution, rnd, openness):
        print(f"\n## {mod.NAME}\n")
        mod.main()
