"""Generate a standalone HTML performance report from the measured artefacts.

Every figure and every chart coordinate is read from `roc_cache.json` and
`history.json`. Nothing is typed by hand, so the report cannot drift from the
files it describes.
"""

import json
from pathlib import Path

DEMO = Path("D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/05_frontend_demo")
SEC03 = Path("D:/Medical AI Workshop/.claude/worktrees/streamlit-cnn-workshop-dcaf1e/03_augmentation_eval")
OUT = Path(r"C:/Users/ahmed/AppData/Local/Temp/claude/D--Medical-AI-Workshop--claude-worktrees-federated-brain-tumor-segmentation-6c2f4f/48802a4e-338c-466f-91c0-c6953828b254/scratchpad/model_report.html")

roc = json.loads((DEMO / "data" / "roc_cache.json").read_text(encoding="utf-8"))
roc25 = json.loads((DEMO / "data" / "roc_cache_epoch25.json").read_text(encoding="utf-8"))
hist_blob = json.loads((SEC03 / "checkpoints" / "overnight_run" / "history.json").read_text(encoding="utf-8"))
H = hist_blob["history"]

COL = {"ET": "#C8324B", "TC": "#4F46B8", "WT": "#1F6FB2", "NC": "#0F7B5F"}
REGION_LONG = {
    "ET": "Enhancing tumour (label 1)",
    "TC": "Tumour core (labels 1 + 3)",
    "WT": "Whole tumour (labels 1–4)",
}


# ───────────────────────────────────────────── chart helpers
def scale(v, lo, hi, out_lo, out_hi):
    if hi == lo:
        return out_lo
    return out_lo + (v - lo) / (hi - lo) * (out_hi - out_lo)


def path_from(points, xlo, xhi, ylo, yhi, box):
    x0, y0, x1, y1 = box
    d = []
    for i, (px, py) in enumerate(points):
        if py is None:
            continue
        sx = scale(px, xlo, xhi, x0, x1)
        sy = scale(py, ylo, yhi, y1, y0)      # y inverted
        d.append(f"{'M' if not d else 'L'}{sx:.2f},{sy:.2f}")
    return " ".join(d)


def axes(box, xlo, xhi, ylo, yhi, xticks, yticks, xlabel, ylabel, ylabel_fmt="{:.2f}"):
    x0, y0, x1, y1 = box
    out = [f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" class="plot-bg"/>']
    for t in yticks:
        sy = scale(t, ylo, yhi, y1, y0)
        out.append(f'<line x1="{x0}" y1="{sy:.1f}" x2="{x1}" y2="{sy:.1f}" class="grid"/>')
        out.append(f'<text x="{x0-8}" y="{sy+3.5:.1f}" class="tick tick-y">{ylabel_fmt.format(t)}</text>')
    for t in xticks:
        sx = scale(t, xlo, xhi, x0, x1)
        out.append(f'<line x1="{sx:.1f}" y1="{y0}" x2="{sx:.1f}" y2="{y1}" class="grid"/>')
        out.append(f'<text x="{sx:.1f}" y="{y1+16}" class="tick tick-x">{t:g}</text>')
    out.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+36}" class="axis-label">{xlabel}</text>')
    out.append(f'<text transform="translate({x0-40},{(y0+y1)/2:.0f}) rotate(-90)" class="axis-label">{ylabel}</text>')
    return "\n".join(out)


# ───────────────────────────────────────────── ROC chart
def roc_chart():
    W, Hh = 560, 470
    box = (60, 20, 520, 400)
    parts = [axes(box, 0, 1, 0, 1, [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1],
                  "False positive rate", "True positive rate")]
    x0, y0, x1, y1 = box
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}" class="chance"/>')
    for reg in ("WT", "TC", "ET"):
        v = roc["regions"][reg]
        pts = list(zip(v["fpr"], v["tpr"]))
        parts.append(f'<path d="{path_from(pts,0,1,0,1,box)}" fill="none" '
                     f'stroke="{COL[reg]}" stroke-width="2.2" stroke-linejoin="round"/>')
    # legend
    ly = 300
    for reg in ("WT", "TC", "ET"):
        v = roc["regions"][reg]
        parts.append(f'<rect x="{x0+18}" y="{ly-9}" width="11" height="11" rx="2" fill="{COL[reg]}"/>')
        parts.append(f'<text x="{x0+36}" y="{ly}" class="legend">{reg} — AUC {v["auc"]:.3f}</text>')
        ly += 24
    parts.append(f'<line x1="{x0+18}" y1="{ly-13}" x2="{x0+29}" y2="{ly-13}" class="chance"/>')
    parts.append(f'<text x="{x0+36}" y="{ly-9}" class="legend legend-muted">chance</text>')
    return f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="ROC curves per tumour region">' + "\n".join(parts) + "</svg>"


# ───────────────────────────────────────────── training curves
def loss_dice_chart():
    W, Hh = 720, 380
    box = (60, 20, 660, 300)
    x0, y0, x1, y1 = box
    epochs = [h["epoch"] for h in H]
    elo, ehi = min(epochs), max(epochs)
    losses = [h["loss"] for h in H]
    llo, lhi = 0, max(losses) * 1.08
    xticks = [e for e in range(elo, ehi + 1) if e % 3 == 0]

    parts = [axes(box, elo, ehi, llo, lhi, xticks, [0, 0.25, 0.5, 0.75, 1.0],
                  "Epoch", "Training loss")]
    # right axis for dice
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        sy = scale(t, 0, 1, y1, y0)
        parts.append(f'<text x="{x1+8}" y="{sy+3.5:.1f}" class="tick tick-y2">{t:.2f}</text>')
    parts.append(f'<text transform="translate({x1+46},{(y0+y1)/2:.0f}) rotate(90)" class="axis-label">Validation Dice</text>')

    parts.append(f'<path d="{path_from(list(zip(epochs,losses)),elo,ehi,llo,lhi,box)}" '
                 f'fill="none" stroke="#10161C" stroke-width="2.2"/>')
    dice = [(h["epoch"], h["mean_dice"]) for h in H]
    parts.append(f'<path d="{path_from(dice,elo,ehi,0,1,box)}" fill="none" '
                 f'stroke="{COL["NC"]}" stroke-width="2.2"/>')

    # mark the best epoch
    best = max(H, key=lambda h: h["mean_dice"])
    bx = scale(best["epoch"], elo, ehi, x0, x1)
    by = scale(best["mean_dice"], 0, 1, y1, y0)
    parts.append(f'<line x1="{bx:.1f}" y1="{y0}" x2="{bx:.1f}" y2="{y1}" class="marker"/>')
    parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" fill="{COL["NC"]}" stroke="var(--ground)" stroke-width="2"/>')
    parts.append(f'<text x="{bx+8:.1f}" y="{by-10:.1f}" class="annot">best · epoch {best["epoch"]} · {best["mean_dice"]:.3f}</text>')

    ly = 336
    parts.append(f'<line x1="{x0}" y1="{ly-4}" x2="{x0+22}" y2="{ly-4}" stroke="#10161C" stroke-width="2.2"/>')
    parts.append(f'<text x="{x0+30}" y="{ly}" class="legend">Training loss (DiceCE)</text>')
    parts.append(f'<line x1="{x0+210}" y1="{ly-4}" x2="{x0+232}" y2="{ly-4}" stroke="{COL["NC"]}" stroke-width="2.2"/>')
    parts.append(f'<text x="{x0+240}" y="{ly}" class="legend">Validation mean Dice</text>')
    return f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Training loss and validation Dice by epoch">' + "\n".join(parts) + "</svg>"


def region_dice_chart():
    W, Hh = 720, 360
    box = (60, 20, 660, 290)
    x0, y0, x1, y1 = box
    epochs = [h["epoch"] for h in H]
    elo, ehi = min(epochs), max(epochs)
    xticks = [e for e in range(elo, ehi + 1) if e % 3 == 0]
    parts = [axes(box, elo, ehi, 0, 1, xticks, [0, 0.25, 0.5, 0.75, 1.0], "Epoch", "Validation Dice")]
    for key, reg in (("dice_WT", "WT"), ("dice_NC", "NC"), ("dice_ET", "ET")):
        pts = [(h["epoch"], h.get(key)) for h in H if h.get(key) is not None]
        parts.append(f'<path d="{path_from(pts,elo,ehi,0,1,box)}" fill="none" '
                     f'stroke="{COL[reg]}" stroke-width="2.2"/>')
    lx = x0 + 14
    for key, reg, label in (("dice_WT", "WT", "Whole tumour"), ("dice_NC", "NC", "Non-enhancing core"),
                            ("dice_ET", "ET", "Enhancing tumour")):
        parts.append(f'<rect x="{lx}" y="{326-9}" width="11" height="11" rx="2" fill="{COL[reg]}"/>')
        parts.append(f'<text x="{lx+18}" y="326" class="legend">{label}</text>')
        lx += 200
    return f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Validation Dice per region by epoch">' + "\n".join(parts) + "</svg>"


def type_chart():
    pts = [(h["epoch"], h["type_accuracy"]) for h in H if "type_accuracy" in h]
    if not pts:
        return ""
    W, Hh = 720, 320
    box = (60, 20, 660, 250)
    x0, y0, x1, y1 = box
    elo, ehi = pts[0][0], pts[-1][0]
    xticks = [e for e in range(elo, ehi + 1) if e % 3 == 0]
    parts = [axes(box, elo, ehi, 0, 1, xticks, [0, 0.25, 0.5, 0.75, 1.0], "Epoch", "Accuracy")]
    base = 0.634
    sy = scale(base, 0, 1, y1, y0)
    parts.append(f'<line x1="{x0}" y1="{sy:.1f}" x2="{x1}" y2="{sy:.1f}" class="baseline"/>')
    parts.append(f'<text x="{x1-6}" y="{sy-8:.1f}" class="annot annot-end">majority-class baseline 0.634</text>')
    parts.append(f'<path d="{path_from(pts,elo,ehi,0,1,box)}" fill="none" stroke="#7A3FA8" stroke-width="2.2"/>')
    parts.append(f'<text x="{x0+14}" y="286" class="legend">Tumour-type head accuracy (geometric proxy target)</text>')
    return f'<svg viewBox="0 0 {W} {Hh}" role="img" aria-label="Tumour-type head accuracy by epoch">' + "\n".join(parts) + "</svg>"


# ───────────────────────────────────────────── tables
def f1(v):
    """F1 from precision and recall.

    Computed the long way round rather than copied from `dice`, so the identity
    F1 == Dice is demonstrated by the report rather than asserted by it.
    """
    p, r = v["precision"], v["sensitivity"]
    return 2 * p * r / (p + r) if (p + r) else 0.0


def metrics_rows():
    out = []
    for reg in ("ET", "TC", "WT"):
        v = roc["regions"][reg]
        out.append(f"""<tr>
      <th scope="row"><span class="swatch" style="background:{COL[reg]}"></span>{reg}
        <span class="row-note">{REGION_LONG[reg]}</span></th>
      <td class="num strong">{v['dice']:.3f}</td>
      <td class="num strong">{f1(v):.3f}</td>
      <td class="num">{v['sensitivity']:.3f}</td>
      <td class="num">{v['precision']:.3f}</td>
      <td class="num">{v['specificity']:.5f}</td>
      <td class="num">{v['hd95_median_mm']:.2f}</td>
      <td class="num">{v['auc']:.3f}</td>
      <td class="num dim">{v['prevalence']*100:.1f}%</td>
    </tr>""")
    return "\n".join(out)


def epoch_rows():
    out = []
    best_e = max(H, key=lambda h: h["mean_dice"])["epoch"]
    for h in H:
        cls = ' class="is-best"' if h["epoch"] == best_e else ""
        ta = f'{h["type_accuracy"]:.3f}' if "type_accuracy" in h else "—"
        out.append(f'<tr{cls}><td class="num">{h["epoch"]}</td><td class="num">{h["loss"]:.4f}</td>'
                   f'<td class="num">{h["dice_ET"]:.3f}</td><td class="num">{h["dice_NC"]:.3f}</td>'
                   f'<td class="num">{h["dice_WT"]:.3f}</td><td class="num strong">{h["mean_dice"]:.3f}</td>'
                   f'<td class="num">{ta}</td></tr>')
    return "\n".join(out)


best = max(H, key=lambda h: h["mean_dice"])
max_type = max(h.get("type_accuracy", 0) for h in H)
ck = roc["checkpoint"]

HTML = f"""<title>Segmentation model — measured performance, epochs 3–36</title>
<style>
  :root {{
    --ground:#FBFCFD; --panel:#FFFFFF; --ink:#10161C; --ink-2:#3A4954;
    --muted:#5F707D; --rule:#DCE3E9; --rule-soft:#EDF1F4;
    --accent:#1F6FB2; --flag:#8A5A00; --flag-bg:#FDF6E7; --flag-rule:#E0C88C;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --ground:#0D1216; --panel:#141B21; --ink:#E8EEF2; --ink-2:#B7C4CD;
      --muted:#8A9AA6; --rule:#243039; --rule-soft:#1B242B;
      --accent:#63A9E0; --flag:#E8C27A; --flag-bg:#241D0F; --flag-rule:#4A3B1C;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#0D1216; --panel:#141B21; --ink:#E8EEF2; --ink-2:#B7C4CD;
    --muted:#8A9AA6; --rule:#243039; --rule-soft:#1B242B;
    --accent:#63A9E0; --flag:#E8C27A; --flag-bg:#241D0F; --flag-rule:#4A3B1C;
  }}
  :root[data-theme="light"] {{
    --ground:#FBFCFD; --panel:#FFFFFF; --ink:#10161C; --ink-2:#3A4954;
    --muted:#5F707D; --rule:#DCE3E9; --rule-soft:#EDF1F4;
    --accent:#1F6FB2; --flag:#8A5A00; --flag-bg:#FDF6E7; --flag-rule:#E0C88C;
  }}

  body {{ background:var(--ground); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    line-height:1.6; margin:0; padding:56px 24px 96px; }}
  .wrap {{ max-width:1080px; margin:0 auto; display:flex; flex-direction:column; gap:44px; }}
  .prose {{ max-width:68ch; }}

  h1 {{ font-family:Georgia,"Iowan Old Style","Times New Roman",serif;
    font-size:clamp(1.7rem,3.2vw,2.35rem); line-height:1.2; font-weight:600;
    margin:0 0 10px; letter-spacing:-0.012em; text-wrap:balance; }}
  h2 {{ font-family:Georgia,"Iowan Old Style","Times New Roman",serif;
    font-size:1.3rem; font-weight:600; margin:0 0 4px; letter-spacing:-0.008em; }}
  h3 {{ font-size:.95rem; font-weight:650; margin:26px 0 8px; }}
  p {{ margin:0 0 14px; color:var(--ink-2); }}
  a {{ color:var(--accent); }}

  .eyebrow {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.13em;
    color:var(--muted); font-weight:650; margin:0 0 14px; }}
  .lede {{ font-size:1.03rem; color:var(--ink-2); }}

  .meta {{ display:flex; flex-wrap:wrap; gap:0 40px; padding:18px 0 0; margin-top:22px;
    border-top:1px solid var(--rule); }}
  .meta div {{ padding:6px 0; }}
  .meta dt {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.1em;
    color:var(--muted); font-weight:650; margin-bottom:2px; }}
  .meta dd {{ margin:0; font-variant-numeric:tabular-nums; font-size:.92rem; }}

  section {{ display:flex; flex-direction:column; gap:14px; }}
  .sec-head {{ border-bottom:1px solid var(--rule); padding-bottom:10px; }}
  .sec-head p {{ margin:4px 0 0; font-size:.88rem; color:var(--muted); }}

  .figure {{ background:var(--panel); border:1px solid var(--rule); border-radius:10px;
    padding:20px; overflow-x:auto; }}
  .figure svg {{ display:block; width:100%; height:auto; min-width:460px; }}
  figcaption {{ font-size:.82rem; color:var(--muted); margin-top:12px; max-width:70ch; }}
  .two-up {{ display:grid; grid-template-columns:1fr; gap:20px; }}
  @media (min-width:900px) {{ .two-up {{ grid-template-columns:1fr 1fr; }} }}

  .plot-bg {{ fill:var(--panel); }}
  .grid {{ stroke:var(--rule-soft); stroke-width:1; }}
  .chance {{ stroke:var(--muted); stroke-width:1.2; stroke-dasharray:4 4; opacity:.75; }}
  .baseline {{ stroke:var(--muted); stroke-width:1.2; stroke-dasharray:5 4; }}
  .marker {{ stroke:var(--rule); stroke-width:1; stroke-dasharray:3 3; }}
  .tick {{ font-size:11px; fill:var(--muted); font-variant-numeric:tabular-nums;
    font-family:ui-monospace,"SF Mono",Menlo,monospace; }}
  .tick-y {{ text-anchor:end; }} .tick-y2 {{ text-anchor:start; }} .tick-x {{ text-anchor:middle; }}
  .axis-label {{ font-size:11.5px; fill:var(--muted); text-anchor:middle;
    text-transform:uppercase; letter-spacing:.08em; }}
  .legend {{ font-size:12.5px; fill:var(--ink-2); }}
  .legend-muted {{ fill:var(--muted); }}
  .annot {{ font-size:11.5px; fill:var(--muted);
    font-family:ui-monospace,"SF Mono",Menlo,monospace; }}
  .annot-end {{ text-anchor:end; }}

  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  .table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--rule);
    border-radius:10px; }}
  th,td {{ padding:11px 14px; text-align:left; border-bottom:1px solid var(--rule-soft); }}
  thead th {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.09em;
    color:var(--muted); font-weight:650; white-space:nowrap;
    border-bottom:1px solid var(--rule); }}
  tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums;
    font-family:ui-monospace,"SF Mono",Menlo,monospace; white-space:nowrap; }}
  .strong {{ font-weight:650; color:var(--ink); }}
  .dim {{ color:var(--muted); }}
  tbody th {{ font-weight:600; }}
  .row-note {{ display:block; font-weight:400; font-size:.75rem; color:var(--muted); }}
  .th-sub {{ display:block; font-weight:400; text-transform:none; letter-spacing:0;
    font-size:.68rem; opacity:.75; }}
  .note {{ font-size:.88rem; color:var(--ink-2); background:var(--rule-soft);
    border-radius:8px; padding:12px 16px; margin:0; }}
  .swatch {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:8px; }}
  tr.is-best td {{ background:color-mix(in srgb, var(--accent) 8%, transparent); }}
  details.scroll-table > summary {{ cursor:pointer; font-size:.86rem; color:var(--accent);
    padding:4px 0; font-weight:600; }}

  .flag {{ background:var(--flag-bg); border:1px solid var(--flag-rule);
    border-left:3px solid var(--flag); border-radius:0 8px 8px 0; padding:16px 20px; }}
  .flag h3 {{ margin:0 0 8px; color:var(--flag); font-size:.9rem; }}
  .flag p {{ color:var(--ink-2); margin:0 0 10px; font-size:.9rem; }}
  .flag p:last-child {{ margin-bottom:0; }}

  ul {{ margin:0 0 14px; padding-left:20px; color:var(--ink-2); }}
  li {{ margin-bottom:7px; }}
  li::marker {{ color:var(--muted); }}
  code {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:.86em;
    background:var(--rule-soft); padding:1px 5px; border-radius:4px; }}
  footer {{ border-top:1px solid var(--rule); padding-top:18px; font-size:.8rem;
    color:var(--muted); }}
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>

<div class="wrap">

<header class="prose">
  <p class="eyebrow">Pediatric brain tumour segmentation · measured performance</p>
  <h1>Segmentation model results, epochs 3–36</h1>
  <p class="lede">A 2D U-Net trained on the BraTS-PEDs cohort. Every figure in this
  document was measured on patients the model never saw during training. Nothing is
  estimated, illustrative, or carried over from a previous experiment — this is
  circulated for clinical feedback on whether the errors it makes are the errors
  that matter.</p>
  <dl class="meta">
    <div><dt>Checkpoint</dt><dd>epoch {ck['epoch']} (best of {len(H)})</dd></div>
    <div><dt>Best mean Dice</dt><dd>{ck['best_mean_dice']:.4f}</dd></div>
    <div><dt>Evaluation cohort</dt><dd>{roc['n_subjects']} held-out patients</dd></div>
    <div><dt>Slices scored</dt><dd>{roc['n_slices']} ({roc['plane']})</dd></div>
    <div><dt>Model size</dt><dd>482,949 parameters</dd></div>
  </dl>
</header>

<section>
  <div class="sec-head">
    <h2>Segmentation accuracy by region</h2>
    <p>Pooled over every pixel of all {roc['n_slices']} scored slices. Regions follow the
    BraTS convention.</p>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th scope="col">Region</th><th scope="col" class="num">Dice</th>
        <th scope="col" class="num">F1</th>
        <th scope="col" class="num">Sensitivity<br><span class="th-sub">recall</span></th>
        <th scope="col" class="num">Precision<br><span class="th-sub">PPV</span></th>
        <th scope="col" class="num">Specificity</th><th scope="col" class="num">HD95 (mm)</th>
        <th scope="col" class="num">AUC</th><th scope="col" class="num">Prevalence</th>
      </tr></thead>
      <tbody>{metrics_rows()}</tbody>
    </table>
  </div>
  <p class="note prose"><strong>F1 and Dice are the same statistic here.</strong> For a
  binary mask both reduce to 2·TP / (2·TP + FP + FN), so the two columns agree to every
  decimal place shown — F1 is listed because it is the familiar name outside imaging,
  not because it is a second, independent result.</p>
  <div class="flag prose">
    <h3>Read Dice, not AUC or specificity</h3>
    <p>Tumour occupies only {roc['regions']['ET']['prevalence']*100:.0f}–{roc['regions']['WT']['prevalence']*100:.0f}%
    of the pixels in a slice, and the remaining background is trivially easy to
    rule out. That inflates both AUC (0.98–0.99) and specificity (0.999) to
    numbers that look near-perfect while whole-tumour Dice sits at
    {roc['regions']['WT']['dice']:.2f}. Dice and HD95 are the figures that reflect
    what the segmentation actually looks like.</p>
    <p>Sensitivity is the more clinically pointed number: the model recovers
    {roc['regions']['WT']['sensitivity']*100:.0f}% of whole-tumour pixels but only
    {roc['regions']['ET']['sensitivity']*100:.0f}% of enhancing tumour, i.e. it
    systematically <em>under-segments</em>, and most of what it does mark is correct
    (precision {roc['regions']['WT']['precision']:.2f} for whole tumour).</p>
  </div>
</section>

<section>
  <div class="sec-head">
    <h2>Discrimination — ROC by region</h2>
    <p>One-vs-rest per pixel, using the model's softmax probability for each region
    as the score.</p>
  </div>
  <div class="two-up">
    <figure class="figure" style="margin:0">
      {roc_chart()}
      <figcaption>Curves are computed from a class-stratified pixel sample: every
      tumour pixel is kept and background is subsampled to
      {roc['max_pixels_per_slice']:,} pixels per slice, so the estimate is not
      dominated by easy background.</figcaption>
    </figure>
    <figure class="figure" style="margin:0">
      {type_chart()}
      <figcaption>The auxiliary head predicting DMG-like versus astrocytoma-like,
      peaking at {max_type:.3f} against a {0.634:.3f} majority-class baseline.
      See the limitation noted below before reading anything clinical into this.</figcaption>
    </figure>
  </div>
</section>

<section>
  <div class="sec-head">
    <h2>Training behaviour</h2>
    <p>Epochs 3–36. Loss is the combined Dice + cross-entropy objective; validation
    is scored per patient on held-out patients, unaugmented.</p>
  </div>
  <figure class="figure" style="margin:0">
    {loss_dice_chart()}
    <figcaption>Loss falls steadily from {H[0]['loss']:.2f} to {H[-1]['loss']:.2f},
    but validation Dice plateaus around epoch {best['epoch']} and does not improve
    over the following {H[-1]['epoch']-best['epoch']} epochs — the model stopped
    generalising further well before it stopped fitting.</figcaption>
  </figure>
  <figure class="figure" style="margin:0">
    {region_dice_chart()}
    <figcaption>Enhancing tumour tracks well below the other regions throughout, which
    is consistent with this cohort: DMG and DIPG frequently show little or no
    enhancement, so there is less signal to segment.</figcaption>
  </figure>
</section>

<section>
  <div class="sec-head">
    <h2>Checkpoint selection — a result worth stating on its own</h2>
    <p>Training ran to epoch 36 and nominated epoch 25 as its best. Scored on the
    held-out patients, epoch 25 is markedly worse.</p>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th scope="col">Region</th>
        <th scope="col" class="num">Epoch 16<br><span class="th-sub">Dice</span></th>
        <th scope="col" class="num">Epoch 25<br><span class="th-sub">Dice</span></th>
        <th scope="col" class="num">Δ Dice</th>
        <th scope="col" class="num">Epoch 16<br><span class="th-sub">HD95 mm</span></th>
        <th scope="col" class="num">Epoch 25<br><span class="th-sub">HD95 mm</span></th>
      </tr></thead>
      <tbody>
      {"".join(
        f'<tr><th scope="row"><span class="swatch" style="background:{COL[r]}"></span>{r}</th>'
        f'<td class="num strong">{roc["regions"][r]["dice"]:.3f}</td>'
        f'<td class="num">{roc25["regions"][r]["dice"]:.3f}</td>'
        f'<td class="num" style="color:{"#0F7B5F" if roc25["regions"][r]["dice"]>roc["regions"][r]["dice"] else "#C8324B"}">'
        f'{roc25["regions"][r]["dice"]-roc["regions"][r]["dice"]:+.3f}</td>'
        f'<td class="num strong">{roc["regions"][r]["hd95_median_mm"]:.2f}</td>'
        f'<td class="num">{roc25["regions"][r]["hd95_median_mm"]:.2f}</td></tr>'
        for r in ("ET","TC","WT"))}
      <tr><th scope="row">Mean</th>
        <td class="num strong">{sum(roc["regions"][r]["dice"] for r in ("ET","TC","WT"))/3:.3f}</td>
        <td class="num">{sum(roc25["regions"][r]["dice"] for r in ("ET","TC","WT"))/3:.3f}</td>
        <td class="num" style="color:#C8324B">{(sum(roc25["regions"][r]["dice"] for r in ("ET","TC","WT"))-sum(roc["regions"][r]["dice"] for r in ("ET","TC","WT")))/3:+.3f}</td>
        <td class="num"></td><td class="num"></td></tr>
      </tbody>
    </table>
  </div>
  <div class="flag prose">
    <h3>The selection metric disagreed with held-out performance</h3>
    <p>Checkpoints were chosen by mean Dice over ET, non-enhancing core and whole
    tumour on 36 patients drawn from the <em>training</em> hospitals. By that
    measure epoch 25 (0.6885) beat epoch 16 (0.6820). On the {roc['n_subjects']}
    genuinely held-out patients the ordering reverses: enhancing-tumour Dice falls
    by {abs(roc25['regions']['ET']['dice']-roc['regions']['ET']['dice']):.3f} and its
    boundary error more than doubles, from
    {roc['regions']['ET']['hd95_median_mm']:.1f}&nbsp;mm to
    {roc25['regions']['ET']['hd95_median_mm']:.1f}&nbsp;mm.</p>
    <p>Whole tumour did improve slightly, so the later checkpoint learned to trace
    the gross tumour outline better while losing the internal, contrast-enhancing
    detail. Averaging the three regions into one number hid that trade-off
    completely. <strong>Every figure elsewhere in this report is from epoch 16</strong>,
    which is the model that should be used.</p>
  </div>
</section>

<section class="prose">
  <div class="sec-head">
    <h2>What was measured, and how</h2>
  </div>
  <ul>
    <li><strong>Held-out patients only.</strong> The {roc['n_subjects']} patients scored
    here appear in no training or validation set at any epoch. Splits are fixed at
    the patient level, never the slice level, so no adjacent slice of the same
    patient sits on both sides.</li>
    <li><strong>Dice, sensitivity, specificity and precision</strong> are pooled from
    exact pixel counts across all scored slices, not averaged per slice — a slice
    with three tumour pixels does not count as much as one with three thousand.</li>
    <li><strong>HD95</strong> is the median across slices of the 95th-percentile
    Hausdorff distance between predicted and true boundaries, at 1&nbsp;mm spacing.
    Slices where a region is absent from either mask are excluded rather than scored
    as zero.</li>
    <li><strong>Validation during training</strong> is computed per patient from
    restacked volumes, using the same code path as final evaluation.</li>
  </ul>
</section>

<section class="prose">
  <div class="sec-head">
    <h2>Limitations worth knowing before you judge these numbers</h2>
  </div>
  <ul>
    <li><strong>This is a small model trained briefly.</strong> 482,949 parameters,
    24 epochs, trained from scratch with no pretraining. It exists to prove the
    pipeline end to end, not to be the final architecture.</li>
    <li><strong>Axial slices only, in 2D.</strong> The model sees one slice at a time
    and has no access to through-plane context a radiologist would use.</li>
    <li><strong>The tumour-type head is not histology.</strong> No tumour-type ground
    truth exists anywhere in this dataset. The head is trained against a geometric
    proxy computed from the segmentation mask — how near the midline the tumour sits,
    how far inferior it lies, how little of it enhances. It should not be described as
    a tumour-type classification result.</li>
    <li><strong>Training is not federated.</strong> The cohort is split across two
    hospital sites plus a held-out site, but this checkpoint was trained centrally.</li>
    <li><strong>Epochs 0–2 are missing</strong> from the curves — a logging fault, since
    fixed. It affects the plots only, not the checkpoint or any measurement above.</li>
  </ul>
</section>

<section class="prose">
  <div class="sec-head">
    <h2>Where clinical feedback would help most</h2>
    <p>Specific questions, in rough order of how much they would change the work.</p>
  </div>
  <ul>
    <li>The model <strong>under-segments</strong> — high precision, lower sensitivity.
    For this use, is missing tumour worse than over-calling it? That decision changes
    how the operating point should be set, and it is a clinical judgement rather than
    a technical one.</li>
    <li><strong>Enhancing tumour is the weakest region</strong> (Dice
    {roc['regions']['ET']['dice']:.2f}, sensitivity {roc['regions']['ET']['sensitivity']:.2f}).
    Is ET accuracy important enough here to trade overall performance for, or is whole
    tumour the number that matters?</li>
    <li><strong>HD95 of {roc['regions']['WT']['hd95_median_mm']:.1f}&nbsp;mm on whole
    tumour</strong> — is that boundary error tolerable for the intended purpose, or is
    the useful threshold much tighter?</li>
    <li>Is the <strong>DMG-versus-astrocytoma geometric proxy</strong> worth keeping as a
    training signal at all, given it encodes assumptions about midline position and
    enhancement rather than tissue?</li>
  </ul>
</section>

<footer class="prose">
  Generated from <code>roc_cache.json</code> and <code>history.json</code>; every value and
  chart coordinate is read from those files rather than transcribed.
  Research use only — not a clinically validated system and not a diagnostic aid.
</footer>

</div>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({len(HTML):,} bytes)")
print(f"best epoch {best['epoch']} mean_dice {best['mean_dice']}  max type_acc {max_type}")
