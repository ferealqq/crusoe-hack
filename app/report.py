"""HTML report for one research run folder: how well each setup did, and how it evolved.

    uv run python -m app.report data/runs/<run>

Works on a finished or an in-progress run. Writes <run>/report.html.
"""

import html
import json
import sys
from pathlib import Path
from typing import Any

from app.config import SETUPS
from app.store import Store

LABELS = {
    "one_shot": "One-shot",
    "no_memory": "No memory",
    "raw_log": "Raw log",
    "shared_brain": "Shared Brain",
}

CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"


def _cost(usage: dict[str, int], price: dict[str, str]) -> float:
    def p(key: str) -> float:
        return float(price.get(key) or 0) / 1e6

    prompt, cached, output = usage.get("prompt", 0), usage.get("cached", 0), usage.get("output", 0)
    return (
        (prompt - cached) * p("prompt") + cached * p("input_cache_reads") + output * p("completion")
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _e(x: Any) -> str:
    return html.escape(str(x))


def load(root: Path) -> dict[str, Any]:
    store = Store(root)
    config = store.read_json("config")
    baseline = store.read_json("baseline") if (root / "baseline.json").exists() else None
    pricing = config.get("pricing", {})
    setups: dict[str, Any] = {}
    for setup in config["setups"]:
        s = Store(root / setup)
        if not (root / setup).exists():
            continue
        gens = s.read("generations")
        agents = s.read("agents")
        exps = s.read("experiments")
        cost = 0.0
        tokens = {"prompt": 0, "cached": 0, "output": 0}
        for a in agents:
            model = (
                config["curator_model"] if a["role"] == "curator" else config["researcher_model"]
            )
            cost += _cost(a.get("usage", {}), pricing.get(model, {}))
            for k in tokens:
                tokens[k] += a.get("usage", {}).get(k, 0)
        setups[setup] = {
            "generations": gens,
            "snapshots": s.read("memory_snapshots"),
            "agents": [{k: v for k, v in a.items() if k != "messages"} for a in agents],
            "runs": len(exps),
            "errors": sum(e["status"] != "ok" for e in exps),
            "result": s.read_json("result") if (root / setup / "result.json").exists() else None,
            "cost": cost,
            "tokens": tokens,
        }
    return {"config": config, "baseline": baseline, "setups": setups}


def chart_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = data["config"]
    n_gens = cfg["research"]["generations"]
    l0 = data["baseline"]["l0"] if data["baseline"] else None
    series = []
    for setup in SETUPS:
        s = data["setups"].get(setup)
        if not s:
            continue
        gens = s["generations"]
        test = [0.0] + [g["test_gain"] for g in gens]
        val = [l0] + [g["end_best_val"] for g in gens]
        mem = [g["memory_tokens"] for g in gens]
        if setup == "one_shot" and gens:  # one generation; draw it flat across the x axis
            test += [test[-1]] * (n_gens - 1)
            val += [val[-1]] * (n_gens - 1)
        series.append(
            {"key": setup, "label": LABELS[setup], "test": test, "val": val, "memory": mem}
        )
    return {
        "gens": list(range(0, n_gens + 1)),
        "loss": _loss(data),
        "l0": l0,
        "threshold": data["baseline"]["threshold"] if data["baseline"] else None,
        "series": series,
    }


def _loss(data: dict[str, Any]) -> str:
    """Display name of the run's loss; runs before the field existed are churn (log loss)."""
    return data["config"].get("loss", "log_loss").replace("_", " ")


def headline(data: dict[str, Any]) -> str:
    rows = {
        k: (s["result"] or {}).get("fitness")
        if s["result"]
        else (s["generations"][-1]["test_gain"] if s["generations"] else None)
        for k, s in data["setups"].items()
    }
    sb, nm = rows.get("shared_brain"), rows.get("no_memory")
    if sb is None or nm is None:
        return (
            "Run in progress or incomplete: numbers below are for the generations finished so far."
        )
    diff = (sb - nm) * 100
    verdict = "ahead of" if diff > 0 else "behind" if diff < 0 else "level with"
    return (
        f"Shared Brain improved test {_loss(data)} by {sb * 100:.2f}% over the starting model, "
        f"{abs(diff):.2f} points {verdict} No memory ({nm * 100:.2f}%) on the same budget."
    )


def summary_table(data: dict[str, Any]) -> str:
    rows = []
    for setup in SETUPS:
        s = data["setups"].get(setup)
        if not s:
            continue
        gens = s["generations"]
        res = s["result"] or {}
        fitness = res.get("fitness") if res else (gens[-1]["test_gain"] if gens else None)
        val_gain = res.get("val_gain") if res else (gens[-1]["val_gain"] if gens else None)
        best_v = gens[-1]["best_version"] if gens else 0
        new_bests = sum(g["new_best"] for g in gens)
        rows.append(
            f"<tr class='row-{setup}'><th scope='row'><span class='swatch'></span>"
            f"{LABELS[setup]}</th>"
            f"<td>{_pct(fitness)}</td><td>{_pct(val_gain)}</td>"
            f"<td>v{best_v} <span class='dim'>({new_bests} new)</span></td>"
            f"<td>{len(gens)}</td><td>{s['runs']}</td><td>{s['errors']}</td>"
            f"<td>{s['tokens']['prompt'] / 1e6:.2f}M <span class='dim'>"
            f"({s['tokens']['cached'] / max(s['tokens']['prompt'], 1):.0%} cached)</span></td>"
            f"<td>${s['cost']:.2f}</td></tr>"
        )
    return "".join(rows)


def evolution(data: dict[str, Any], setup: str) -> str:
    s = data["setups"].get(setup)
    if not s:
        return "<p class='dim'>Not run.</p>"
    snaps = {sn["gen"]: sn for sn in s["snapshots"]}
    out = []
    for g in s["generations"]:
        gen = g["gen"]
        sn = snaps.get(gen, {})
        best = (
            f"<span class='pill good'>new best v{g['best_version']}</span> {_e(g['best_note'])}"
            if g["new_best"]
            else "<span class='pill'>no new best</span>"
        )
        researchers = "".join(
            f"<tr><td>{_e(r['focus'])}</td><td>{r['rank']}</td><td>"
            + (
                f"{r['best']['val_loss']:.5f} <span class='dim'>{_e(r['best']['run_id'])}</span>"
                if r["best"]
                else "no successful run"
            )
            + f"</td><td>{_e((r['best'] or {}).get('note', ''))}</td></tr>"
            for r in g["researcher_best"]
        )
        events = "".join(
            f"<li><b>{_e(ev['action'].replace('_', ' '))}</b> "
            f"<code>{_e(ev['entry']['id'])}</code> {_e(ev['entry']['kind'])}: "
            f"{_e(ev['entry']['claim'][:220])}"
            + (f" <span class='dim'>({_e(ev['reason'])})</span>" if ev["reason"] else "")
            + "</li>"
            for ev in sn.get("events", [])
        )
        mem = (
            f"<details><summary>Memory after generation {gen} "
            f"({sn.get('tokens', 0)} tokens)</summary><pre>{_e(sn.get('text', ''))}</pre></details>"
            if sn.get("text")
            else ""
        )
        curator = (
            f"<p class='curator'><b>Curator:</b> {_e(sn['curator_summary'][:1200])}</p>"
            if sn.get("curator_summary")
            else ""
        )
        out.append(
            f"<article class='gen'><header><h4>Generation {gen}</h4><div>{best}</div>"
            f"<div class='nums'>val {g['end_best_val']:.5f} &middot; test gain "
            f"{_pct(g['test_gain'])} &middot; {g['runs']} runs, {g['errors'] + g['timeouts']} "
            f"failed &middot; read {g['memory_tokens']} tokens &middot; {g['minutes']} min</div>"
            f"</header><div class='tablewrap'><table class='small'><thead><tr><th>Focus</th>"
            f"<th>Rank</th><th>Best val</th><th>Change</th></tr></thead><tbody>{researchers}"
            f"</tbody></table></div>"
            + (f"<h5>Memory changes</h5><ul class='events'>{events}</ul>" if events else "")
            + curator
            + mem
            + "</article>"
        )
    return "".join(out) or "<p class='dim'>No finished generations yet.</p>"


STYLE = (
    """
/* Layout: one reading column; charts in a two-up grid that stacks on phones. */
:root {
  --bg: #f6f7f4; --surface: #ffffff; --ink: #18211e; --muted: #5d6a65; --rule: #d9dfdb;
  --accent: #0b6e63;
  --one_shot: #9aa39f; --no_memory: #5b6f9c; --raw_log: #b5832b; --shared_brain: #0b6e63;
  --good: #1f7a45; --good-bg: #e3f1e8;
  --display: "Newsreader", Georgia, serif;
  --body: "Public Sans", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111614; --surface: #19201d; --ink: #e4ebe7; --muted: #94a39d; --rule: #2c3632;
  --accent: #4cc2b2;
  --one_shot: #7f8884; --no_memory: #8ea3d6; --raw_log: #d9a54a; --shared_brain: #4cc2b2;
  --good: #7fd39c; --good-bg: #1d3326; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111614; --surface: #19201d; --ink: #e4ebe7; --muted: #94a39d; --rule: #2c3632;
  --accent: #4cc2b2;
  --one_shot: #7f8884; --no_memory: #8ea3d6; --raw_log: #d9a54a; --shared_brain: #4cc2b2;
  --good: #7fd39c; --good-bg: #1d3326; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
main { max-width: 68rem; margin: 0 auto; padding-inline: 16px; padding-block: 2rem 4rem;
  display: grid; gap: 2.25rem; }
h1, h2, h3 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 2.2rem; line-height: 1.1; }
h2 { font-size: 1.45rem; }
h4 { margin: 0; font-size: 1rem; }
h5 { margin: .75rem 0 .25rem; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); }
p { margin: 0; max-width: 65ch; }
.eyebrow { font: 600 .75rem var(--mono); letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); }
.lede { font-size: 1.1rem; }
.dim { color: var(--muted); }
section { display: grid; gap: .9rem; min-width: 0; }
.meta { display: flex; flex-wrap: wrap; gap: .4rem 1.25rem; font: .8rem var(--mono);
  color: var(--muted); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .45rem .7rem .45rem 0; border-bottom: 1px solid var(--rule);
  vertical-align: top; }
thead th { font: 600 .72rem var(--body); text-transform: uppercase; letter-spacing: .05em;
  color: var(--muted); }
td { font-family: var(--mono); font-size: .85rem; }
table.small td { font-size: .78rem; }
.swatch { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px;
  margin-right: .5rem; vertical-align: baseline; }
"""
    + "".join(f".row-{k} .swatch {{ background: var(--{k}); }}\n" for k in LABELS)
    + """
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
  gap: 1.5rem; }
figure { margin: 0; display: grid; gap: .5rem; min-width: 0; }
figcaption { font-size: .85rem; color: var(--muted); }
.canvasbox { position: relative; height: 18rem; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 6px; padding: .75rem; }
.gen { border-top: 1px solid var(--rule); padding-block: 1rem; display: grid; gap: .5rem;
  min-width: 0; }
.gen header { display: grid; gap: .25rem; }
.nums { font: .78rem var(--mono); color: var(--muted); }
.pill { display: inline-block; font: 600 .72rem var(--body); padding: .1rem .5rem;
  border-radius: 999px; border: 1px solid var(--rule); color: var(--muted); }
.pill.good { background: var(--good-bg); color: var(--good); border-color: transparent; }
.events { margin: 0; padding-left: 1.1rem; font-size: .85rem; display: grid; gap: .2rem; }
.curator { font-size: .88rem; color: var(--muted); }
details summary { cursor: pointer; font-size: .85rem; color: var(--accent); }
details summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
pre { white-space: pre-wrap; font: .78rem/1.5 var(--mono); background: var(--surface);
  border: 1px solid var(--rule); border-radius: 6px; padding: .75rem; overflow-x: auto; }
code { font-family: var(--mono); font-size: .85em; }
.tabs { display: flex; flex-wrap: wrap; gap: .5rem; }
.tabs button { font: 600 .82rem var(--body); padding: .35rem .8rem; border-radius: 999px;
  border: 1px solid var(--rule); background: var(--surface); color: var(--ink); cursor: pointer; }
.tabs button[aria-selected="true"] { border-color: var(--accent); color: var(--accent); }
.tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""
)

SCRIPT = """
const css = getComputedStyle(document.documentElement);
const v = (n) => css.getPropertyValue(n).trim();
const D = JSON.parse(document.getElementById('data').textContent);
Chart.defaults.font.family = v('--body');
Chart.defaults.color = v('--muted');
Chart.defaults.borderColor = v('--rule');
const pct = (x) => (x == null ? 'n/a' : (x * 100).toFixed(2) + '%');
function line(id, key, opts) {
  const sets = D.series.filter((s) => !opts.only || opts.only.includes(s.key)).map((s) => ({
    label: s.label, data: s[key], borderColor: v('--' + s.key), backgroundColor: v('--' + s.key),
    borderWidth: s.key === 'shared_brain' ? 3 : 2, pointRadius: 3, tension: 0,
    borderDash: s.key === 'one_shot' ? [5, 4] : [], stepped: opts.stepped || false,
    spanGaps: true,
  }));
  if (opts.extra) sets.push(...opts.extra);
  new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: opts.labels || D.gens, datasets: sets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } },
        tooltip: { callbacks: { label: (c) => c.dataset.label + ': ' + opts.fmt(c.parsed.y) } } },
      scales: {
        x: { title: { display: true, text: 'Generation' }, grid: { display: false } },
        y: { title: { display: true, text: opts.y }, ticks: { callback: opts.fmt } },
      },
    },
  });
}
line('test', 'test', { y: 'Test gain vs starting model', fmt: pct });
line('val', 'val', { y: 'Best val ' + D.loss, fmt: (x) => x == null ? '' : x.toFixed(4),
  stepped: true,
  extra: D.l0 ? [{ label: 'Start minus noise threshold', data: D.gens.map(() => D.l0 - D.threshold),
    borderColor: v('--rule'), borderDash: [2, 3], pointRadius: 0, borderWidth: 1 }] : [] });
line('mem', 'memory', { y: 'Tokens researchers read', only: ['raw_log', 'shared_brain'],
  labels: D.gens.slice(1), fmt: (x) => x == null ? '' : Math.round(x).toLocaleString() });
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tabs button').forEach((o) => o.setAttribute('aria-selected', o === b));
  document.querySelectorAll('.evo').forEach((p) => { p.hidden = p.id !== 'evo-' + b.dataset.setup; });
}));
"""


def render(data: dict[str, Any]) -> str:
    cfg, base = data["config"], data["baseline"]
    r = cfg["research"]
    loss, name = _loss(data), cfg.get("dataset", "churn")
    tabs = "".join(
        f"<button type='button' role='tab' data-setup='{k}' "
        f"aria-selected='{'true' if k == 'shared_brain' else 'false'}'>{LABELS[k]}</button>"
        for k in SETUPS
        if k in data["setups"]
    )
    panels = "".join(
        f"<div class='evo' id='evo-{k}' {'' if k == 'shared_brain' else 'hidden'}>"
        f"{evolution(data, k)}</div>"
        for k in SETUPS
        if k in data["setups"]
    )
    base_line = (
        f"<span>start val {loss} L0 {base['l0']:.5f}</span>"
        f"<span>noise threshold {base['threshold']:.5f}</span>"
        f"<span>start test {loss} {sum(base['test_losses']) / 3:.5f}</span>"
        if base
        else ""
    )
    page = f"""<title>Shared Brain {_e(name.title())} Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,600&family=Public+Sans:wght@400;600&display=swap">
<style>{STYLE}</style>
<main>
<header style="display:grid;gap:.6rem">
  <div class="eyebrow">Shared Brain &middot; {_e(name)} &middot; run {_e(cfg["started"])}</div>
  <h1>Does curated memory make research agents better?</h1>
  <p class="lede">{_e(headline(data))}</p>
  <div class="meta"><span>{r["generations"]} generations</span>
  <span>{r["researchers"]} researchers each</span>
  <span>28 runs per generation</span><span>researcher {_e(cfg["researcher_model"])}</span>
  <span>curator {_e(cfg["curator_model"])}</span>{base_line}</div>
</header>
<section>
  <h2>Results by setup</h2>
  <p class="dim">Test gain is the drop in test {loss} of the final best train.py versus the
  starting one, averaged over seeds 1, 2 and 3. Agents never see the test split.</p>
  <div class="tablewrap"><table>
  <thead><tr><th>Setup</th><th>Test gain</th><th>Val gain</th><th>Best</th><th>Gens</th>
  <th>Runs</th><th>Failed</th><th>Prompt tokens</th><th>Cost</th></tr></thead>
  <tbody>{summary_table(data)}</tbody></table></div>
</section>
<section>
  <h2>How it evolved</h2>
  <div class="charts">
    <figure><div class="canvasbox"><canvas id="test" aria-label="Test gain per generation"></canvas></div>
      <figcaption>Test gain of each generation's best train.py. Higher is better.</figcaption></figure>
    <figure><div class="canvasbox"><canvas id="val" aria-label="Best val loss per generation"></canvas></div>
      <figcaption>Best val {loss} so far. A step down is a new best that beat the old one by
      more than the noise threshold.</figcaption></figure>
    <figure><div class="canvasbox"><canvas id="mem" aria-label="Memory size per generation"></canvas></div>
      <figcaption>What researchers read at the start of each generation. Curated memory is
      capped at 2,000 tokens; the raw log keeps growing.</figcaption></figure>
  </div>
</section>
<section>
  <h2>Generation by generation</h2>
  <div class="tabs" role="tablist">{tabs}</div>
  {panels}
</section>
</main>
<script id="data" type="application/json">{json.dumps(chart_data(data)).replace("</", "<\\/")}</script>
<script src="{CHART_JS}"></script>
<script>{SCRIPT}</script>
"""
    return page


def write_report(root: Path) -> Path:
    path = root / "report.html"
    path.write_bytes(render(load(root)).encode("ascii", "xmlcharrefreplace"))
    return path


if __name__ == "__main__":
    print(write_report(Path(sys.argv[1])))
