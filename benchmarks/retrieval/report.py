"""Turn a comparison-*.json results file into a self-contained HTML dashboard.

Usage (from project root):
    python -m benchmarks.retrieval.report [path/to/comparison-*.json]

With no path, it uses the most recent comparison-*.json in results/. Writes an
.html next to the JSON and prints its path.
"""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path

from . import config

# Palette (validated via the dataviz skill's checker, light + dark).
HYBRID = "var(--hybrid)"
BASELINE = "var(--baseline)"
TIE = "var(--tie)"


def _latest_results() -> Path | None:
    files = sorted(config.RESULTS_DIR.glob("comparison-*.json"))
    return files[-1] if files else None


def _esc(text) -> str:
    return html.escape(str(text))


# --- SVG chart builders ----------------------------------------------------

def _diverging_hbar(rows, vmax=2.0, height_per=30, width=680, label_w=220):
    """rows: [(label, value)]  value in [-vmax, vmax]; +blue(hybrid) / -orange(baseline)."""
    plot_w = width - label_w - 60
    mid = label_w + plot_w / 2
    h = height_per * len(rows) + 30
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    # zero axis
    parts.append(f'<line x1="{mid}" y1="10" x2="{mid}" y2="{h-20}" stroke="var(--axis)" stroke-width="1"/>')
    for i, (label, value) in enumerate(rows):
        y = 20 + i * height_per
        bar_len = (abs(value) / vmax) * (plot_w / 2)
        color = HYBRID if value >= 0 else BASELINE
        x = mid if value >= 0 else mid - bar_len
        parts.append(f'<text x="{label_w-8}" y="{y+9}" text-anchor="end" font-size="12" fill="var(--text-2)">{_esc(label)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(bar_len,1):.1f}" height="14" rx="3" fill="{color}"/>')
        lx = mid + bar_len + 6 if value >= 0 else mid - bar_len - 6
        anchor = "start" if value >= 0 else "end"
        parts.append(f'<text x="{lx:.1f}" y="{y+11}" text-anchor="{anchor}" font-size="11" fill="var(--muted)">{value:+.2f}</text>')
    parts.append(f'<text x="{label_w}" y="{h-4}" font-size="10" fill="var(--muted)">← baseline better</text>')
    parts.append(f'<text x="{width-50}" y="{h-4}" text-anchor="end" font-size="10" fill="var(--muted)">hybrid better →</text>')
    parts.append("</svg>")
    return "".join(parts)


def _stacked_hbar(rows, height_per=30, width=680, label_w=220):
    """rows: [(label, (win, tie, loss))] fractions summing ~1. win=hybrid, loss=baseline."""
    plot_w = width - label_w - 20
    h = height_per * len(rows) + 10
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for i, (label, (win, tie, loss)) in enumerate(rows):
        y = 20 + i * height_per
        parts.append(f'<text x="{label_w-8}" y="{y+11}" text-anchor="end" font-size="12" fill="var(--text-2)">{_esc(label)}</text>')
        x = label_w
        for frac, color in ((win, HYBRID), (tie, TIE), (loss, BASELINE)):
            seg = frac * plot_w
            if seg > 0.5:
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{seg:.1f}" height="16" fill="{color}"/>')
                if seg > 26:
                    parts.append(f'<text x="{x+seg/2:.1f}" y="{y+12}" text-anchor="middle" font-size="10" fill="#fff">{round(frac*100)}%</text>')
            x += seg
    parts.append("</svg>")
    return "".join(parts)


def _grouped_hbar(rows, vmax, height_per=38, width=680, label_w=220):
    """rows: [(label, hybrid_val, baseline_val)] — two bars per row."""
    plot_w = width - label_w - 60
    h = height_per * len(rows) + 10
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for i, (label, hv, bv) in enumerate(rows):
        y = 15 + i * height_per
        parts.append(f'<text x="{label_w-8}" y="{y+16}" text-anchor="end" font-size="12" fill="var(--text-2)">{_esc(label)}</text>')
        for j, (val, color) in enumerate(((hv, HYBRID), (bv, BASELINE))):
            by = y + j * 14
            bl = (val / vmax) * plot_w if vmax else 0
            parts.append(f'<rect x="{label_w}" y="{by}" width="{max(bl,1):.1f}" height="11" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{label_w+bl+6:.1f}" y="{by+9}" font-size="10" fill="var(--muted)">{val:.1f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _legend(items):
    chips = "".join(
        f'<span class="chip"><span class="sw" style="background:{c}"></span>{_esc(n)}</span>'
        for n, c in items
    )
    return f'<div class="legend">{chips}</div>'


# --- KPI + table -----------------------------------------------------------

def _kpi(value, label, tone=""):
    return f'<div class="kpi {tone}"><div class="kpi-v">{_esc(value)}</div><div class="kpi-l">{_esc(label)}</div></div>'


def _question_rows(per_question):
    rows = []
    for q in sorted(per_question, key=lambda r: r["id"]):
        score = q.get("judge_score")
        if score is None:
            tone, badge = "", "—"
        elif score > 0:
            tone, badge = "pos", f"{score:+.1f}"
        elif score < 0:
            tone, badge = "neg", f"{score:+.1f}"
        else:
            tone, badge = "zero", "0.0"
        reasons = q.get("judge_reasons", [])
        reason_html = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
        verdicts = " / ".join(q.get("judge_verdicts", []))
        models = ", ".join(dict.fromkeys(q.get("judge_models", [])))
        detail = ""
        if reasons:
            detail = (f'<details><summary>verdicts: {_esc(verdicts)}'
                      f'<span class="muted"> · {_esc(models)}</span></summary>'
                      f'<ul>{reason_html}</ul></details>')
        rows.append(
            f'<tr><td>{q["id"]}</td><td>{_esc(q["category"])}</td>'
            f'<td class="q">{_esc(q["question"])}{detail}</td>'
            f'<td class="score {tone}">{badge}</td>'
            f'<td>{q.get("graph_contribution","")}</td>'
            f'<td>{q.get("jaccard",0):.2f}</td></tr>'
        )
    return "".join(rows)


def build_html(data: dict) -> str:
    r = data["report"]
    o = r["overall"]
    cats = r["by_category"]
    judged = "mean_judge_score" in o

    cat_labels = {k: f'{k} · {v["name"]}' for k, v in cats.items()}
    order = list(cats.keys())

    # KPI row
    kpis = []
    if judged:
        kpis.append(_kpi(f'{o["hybrid_win_rate"]*100:.0f}%', "hybrid win", "pos"))
        kpis.append(_kpi(f'{o["tie_rate"]*100:.0f}%', "tie", "zero"))
        kpis.append(_kpi(f'{o["baseline_win_rate"]*100:.0f}%', "baseline win", "neg"))
        kpis.append(_kpi(f'{o["mean_judge_score"]:+.2f}', "mean score (+hybrid)"))
        kpis.append(_kpi(f'{o["judge_consistency"]*100:.0f}%', "judge consistency"))
    kpis.append(_kpi(f'{o["mean_jaccard"]:.2f}', "mean overlap (Jaccard)"))
    kpis.append(_kpi(f'{o["mean_graph_contribution"]:.1f}', "graph-added /10"))
    lat = r["latency"]
    kpis.append(_kpi(f'{lat["hybrid_mean_s"]*1000:.0f}/{lat["baseline_mean_s"]*1000:.0f}ms', "latency hy/bl"))

    charts = []
    if judged:
        div_rows = [(cat_labels[k], cats[k]["mean_judge_score"]) for k in order]
        charts.append(("Mean judge score by category (+ favors hybrid)",
                       _diverging_hbar(div_rows) + _legend([("hybrid", "var(--hybrid)"), ("baseline", "var(--baseline)")])))
        wtl_rows = [(cat_labels[k], (cats[k]["hybrid_win_rate"], cats[k]["tie_rate"], cats[k]["baseline_win_rate"])) for k in order]
        charts.append(("Win / tie / loss by category",
                       _stacked_hbar(wtl_rows) + _legend([("hybrid win", "var(--hybrid)"), ("tie", "var(--tie)"), ("baseline win", "var(--baseline)")])))

    div_rows2 = [(cat_labels[k], cats[k]["mean_hybrid_source_diversity"], cats[k]["mean_baseline_source_diversity"]) for k in order]
    charts.append(("Source-document diversity per top-10 (hybrid vs baseline)",
                   _grouped_hbar(div_rows2, vmax=6) + _legend([("hybrid", "var(--hybrid)"), ("baseline", "var(--baseline)")])))

    graph_rows = [(cat_labels[k], cats[k]["mean_graph_contribution"], cats[k]["mean_jaccard"] * 10) for k in order]
    charts.append(("Graph contribution (/10) and result overlap (Jaccard ×10) by category",
                   _grouped_hbar(graph_rows, vmax=10) + _legend([("graph-added chunks", "var(--hybrid)"), ("Jaccard ×10", "var(--baseline)")])))

    charts_html = "".join(
        f'<section class="card"><h2>{_esc(title)}</h2>{body}</section>' for title, body in charts
    )

    models = r.get("models", {})
    meta = (f'hybrid: {_esc(models.get("hybrid_embed"))} + graph  ·  '
            f'baseline: {_esc(models.get("baseline_embed"))}  ·  '
            f'judge: {_esc(", ".join(models.get("judge_pool") or []) or "none")}  ·  '
            f'top_k {r.get("top_k")}  ·  {o["n"]} questions  ·  {_esc(r.get("timestamp",""))}')

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retrieval Comparison</title>
<style>
:root {{
  --surface:#fcfcfb; --page:#f4f4f1; --text:#0b0b0b; --text-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --hybrid:#2a78d6; --baseline:#eb6834; --tie:#c3c2b7;
  --pos:#2a78d6; --neg:#eb6834;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --surface:#1a1a19; --page:#0d0d0d; --text:#fff; --text-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --hybrid:#3987e5; --baseline:#d95926; --tie:#4a4a47; --pos:#3987e5; --neg:#d95926;
}}}}
:root[data-theme="dark"] {{
  --surface:#1a1a19; --page:#0d0d0d; --text:#fff; --text-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --hybrid:#3987e5; --baseline:#d95926; --tie:#4a4a47; --pos:#3987e5; --neg:#d95926;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--text);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.4; padding:24px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.meta {{ color:var(--muted); font-size:12px; margin-bottom:20px; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:22px; }}
.kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:12px 16px; min-width:110px; }}
.kpi-v {{ font-size:22px; font-weight:600; }}
.kpi-l {{ font-size:11px; color:var(--muted); margin-top:2px; }}
.kpi.pos .kpi-v {{ color:var(--pos); }} .kpi.neg .kpi-v {{ color:var(--neg); }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:16px 18px; margin-bottom:18px; }}
.card h2 {{ font-size:14px; margin:0 0 12px; font-weight:600; }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; }}
.chip {{ font-size:11px; color:var(--text-2); display:flex; align-items:center; gap:5px; }}
.sw {{ width:11px; height:11px; border-radius:2px; display:inline-block; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
.table-wrap {{ max-height:520px; overflow:auto; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); vertical-align:top; }}
th {{ position:sticky; top:0; background:var(--surface); color:var(--muted); font-weight:600; }}
td.q {{ max-width:520px; }}
td.score {{ font-weight:600; font-variant-numeric:tabular-nums; }}
td.score.pos {{ color:var(--pos); }} td.score.neg {{ color:var(--neg); }} td.score.zero {{ color:var(--muted); }}
details summary {{ cursor:pointer; color:var(--text-2); margin-top:4px; font-size:11px; }}
details ul {{ margin:6px 0 0; padding-left:16px; color:var(--text-2); }}
.muted {{ color:var(--muted); }}
</style></head><body>
<h1>Retrieval Comparison — hybrid (graph + bge-small) vs baseline (bge-large)</h1>
<div class="meta">{meta}</div>
<div class="kpis">{''.join(kpis)}</div>
{charts_html}
<section class="card"><h2>Per-question detail ({o['n']}) — expand a row for judge verdicts &amp; reasons</h2>
<div class="table-wrap"><table>
<thead><tr><th>#</th><th>cat</th><th>question</th><th>score</th><th>graph+</th><th>jac</th></tr></thead>
<tbody>{_question_rows(data['per_question'])}</tbody>
</table></div></section>
</body></html>"""


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_results()
    if not path or not path.exists():
        print("No results JSON found. Pass a path or run the benchmark first.")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    out = path.with_suffix(".html")
    out.write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
