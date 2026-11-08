# ai-institutions

[中文说明](README.zh-CN.md)

**Research question.** With the same R&D budget, how do the openness of AI research, the ownership of AI firms and the way they are financed shape innovation, the distribution of income, and who bears the risk?

The project does not assume that any institution is better. Every result is a conditional statement derived from a stylized model, and the conditions under which it holds are written down next to it. Parameters are illustrative and uncalibrated; nothing here is a forecast.

## Interactive explainer

Four institutions (closed or open research × private or public funding and ownership) run for 25 periods with the same budget and the same random shocks. Flip the switches to see how 100 people fare, how four firms "climb the mountain", where the money flows, and 50 possible histories. Other switches change the assumptions: labor-income replacement, pricing, whether private firms choose their own R&D, frictions in open-research spillovers, and a public lab next to private firms.

**Open it online:** [English](https://shaoxiongyao.github.io/ai-institutions/results/dynamics/en/) · [中文](https://shaoxiongyao.github.io/ai-institutions/results/dynamics/)

Each page is self-contained (about 10 MB, data included), so you can also download [the English file](results/dynamics/en/index.html) or [the Chinese file](results/dynamics/index.html) and open it in a browser offline.

## Findings in brief

Full statements, assumptions, evidence and sensitivity checks are in [docs/findings.md](docs/findings.md) (in Chinese).

**One-period model (v0.1)**

- **F1.** Changing only ownership and financing leaves total (linear) welfare unchanged; it moves gains and risk between groups.
- **F2.** Public financing with concentrated private ownership does worst under concave utility: society pays for R&D while investors keep the returns.
- **F3.** Broad ownership with private financing shifts investment risk to workers.
- **F4–F5.** With a fixed budget, spreading R&D across projects raises the expected frontier only when returns are diminishing, outcomes are very uncertain and projects are weakly correlated.
- **F6–F7.** The value of openness comes from letting users reach the frontier; it is larger when the market is less concentrated on the best model. If openness erodes pricing power, it moves value from firm owners to users.

**Multi-period model (draft; drives the explainer)**

- **F8.** When openness erodes pricing power, privately funded open research can collapse: firms lose money and exit. Whether it survives depends on the revenue firms earn from non-technical differentiation. Private closed research drifts toward monopoly.
- **F9.** When private firms choose their own R&D, open firms hardly invest even with fixed pricing. The main mechanism is free riding; lower pricing power makes it worse.
- **F10.** "Open research always reaches about 10" comes from frictionless spillovers. Imperfect absorption, herding of research paths, or rising difficulty shrink the open advantage a lot; under public funding it rarely reverses, under private funding it can.
- **F11.** A public lab does not generally crowd out private closed R&D; an open public lab makes private closed firms invest more. When private firms are open too, an open public lab replaces their R&D entirely. The lab's results are mostly monetized by private owners.

## Repository layout

| Path | Contents |
|---|---|
| [docs/model.md](docs/model.md) | Model specification (in Chinese); section numbers match the code |
| [docs/findings.md](docs/findings.md) | Findings with assumptions, evidence and sensitivity (in Chinese) |
| [src/ai_institutions/](src/ai_institutions/) | Model code |
| [experiments/](experiments/) | One script per experiment; results go to [results/](results/) |
| [tests/](tests/) | Accounting identities, boundary cases, and checks of each mechanism |
| [viz/](viz/) | Explainer templates: `dynamics.html` (Chinese) and `dynamics.en.html` (English), translations of each other, so edit both |

Each `results/<experiment>/summary.json` (or `config.json`) records the configuration, random seed, number of draws and git commit.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                              # accounting identities and boundary cases
uv run python -m experiments.run_all       # one-period experiments: distribution, R&D, openness
uv run python -m experiments.pricing       # F7, F8
uv run python -m experiments.investment    # F9
uv run python -m experiments.spillovers    # F10
uv run python -m experiments.mixed         # F11
uv run python -m experiments.dynamics      # explainer data (about 14 minutes)
uv run python -m experiments.build_viz     # build both explainer pages
```

With the same seed, regenerated data match the committed files except for the run record (`run_info`).

Baseline parameters are in [experiments/common.py](experiments/common.py). Change one, rerun, and compare with the "sensitivity" column in the findings to see whether a conclusion survives.

## Citation

If you use this work, please cite it as below (also available from [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this repository"):

```bibtex
@software{yao_ai_institutions_2026,
  author = {Yao, Shaoxiong},
  title  = {ai-institutions: How openness, ownership and financing of AI R\&D shape innovation, distribution and risk},
  year   = {2026},
  url    = {https://github.com/ShaoxiongYao/ai-institutions}
}
```

## License

- Code (`src/`, `experiments/`, `tests/`, and the scripts in `viz/`): [Apache License 2.0](LICENSE).
- Documentation, figures, results and the text of the explainer pages: [CC BY 4.0](LICENSE-docs.md).
