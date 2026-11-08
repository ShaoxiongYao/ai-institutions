"""Build the animated explainer from viz/dynamics.html and the dynamics results.

Run after experiments.dynamics:
    uv run python -m experiments.build_viz

Writes, for the Chinese page (viz/dynamics.html) and the English page
(viz/dynamics.en.html), two files with the same content:
  explainer.html  page fragment, as published to the Claude Artifact
                  (the publisher adds <!doctype>, <head>, <body>)
  index.html      complete standalone page: open it directly in a browser,
                  or serve it (e.g. GitHub Pages)
Chinese output goes to results/dynamics/, English to results/dynamics/en/.
The two templates are translations of each other: edit both together.
"""

from __future__ import annotations

from .common import RESULTS, ROOT

DATA = RESULTS / "dynamics" / "trajectories.json"
# (template, output directory, html lang)
PAGES = (
    (ROOT / "viz" / "dynamics.html", RESULTS / "dynamics", "zh-CN"),
    (ROOT / "viz" / "dynamics.en.html", RESULTS / "dynamics" / "en", "en"),
)

# Matches the reset the Artifact publisher adds around a fragment.
HEAD = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}
body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def standalone(fragment: str, lang: str) -> str:
    """Move the fragment's <title>/<meta>/<link>/<style> preamble into <head>."""
    cut = fragment.index("</style>") + len("</style>")
    return f"{HEAD.replace('{lang}', lang)}{fragment[:cut]}\n</head>\n<body>\n{fragment[cut:].lstrip()}\n</body>\n</html>\n"


def main() -> None:
    data = DATA.read_text()
    if "</script" in data:
        raise SystemExit("data must not contain '</script'")
    for template, out, lang in PAGES:
        html = template.read_text()
        if "__DATA__" not in html:
            raise SystemExit(f"{template.name} is missing the __DATA__ placeholder")
        fragment = html.replace("__DATA__", data, 1)
        out.mkdir(parents=True, exist_ok=True)
        (out / "explainer.html").write_text(fragment)
        (out / "index.html").write_text(standalone(fragment, lang))
        for name in ("explainer.html", "index.html"):
            p = out / name
            print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
