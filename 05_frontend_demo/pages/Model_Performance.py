"""Measured model performance: training curves, ROC / AUC, per-region Dice.

Sources, all real:
  * training curves  -> checkpoints/<run>/history.json (written per epoch)
  * ROC / AUC        -> data/roc_cache.json, computed by utils.build_metrics_cache
                        on the held-out patients the model never trained on
  * checkpoint facts -> the .pt file itself

If a source is missing the page says so. It never falls back to a stored number.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from components.theme import apply_custom_theme
from utils import loaders
from utils.metrics import REGION_COLOURS

st.set_page_config(page_title="Model Performance | NeuroFed AI", page_icon="📈", layout="wide")
apply_custom_theme()

st.markdown("""
<style>
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:8px; }
.page-sub { font-size:0.92rem; }
.panel-title { font-size:0.95rem; font-weight:700; margin-bottom:16px; }
.metric-row { display:flex; gap:14px; flex-wrap:wrap; }
.metric-box { flex:1; min-width:150px; border:1px solid var(--panel-border); border-radius:12px;
              padding:16px 18px; background:#fff; }
.metric-label { font-size:0.72rem; color:var(--text-muted); text-transform:uppercase;
                letter-spacing:0.05em; margin-bottom:6px; }
.metric-value { font-size:1.6rem; font-weight:800; letter-spacing:-0.02em; }
.metric-note  { font-size:0.7rem; color:var(--text-faint); margin-top:4px; }
</style>
<div class="page-hero">
    <div class="page-title">Model Performance</div>
    <div class="page-sub">Every figure below is measured from the trained checkpoint. Nothing is illustrative.</div>
</div>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
    font=dict(color="#0F172A", size=12),
    xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    margin=dict(l=50, r=20, t=30, b=45), height=380,
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2E8F0", borderwidth=1),
)

status = loaders.load_checkpoint_status()
if not status["available"]:
    st.error(f"No checkpoint found.\n\n```\n{status['detail']}\n```")
    st.stop()

# ─────────────────────────────── headline numbers
roc_path = Path(__file__).resolve().parent.parent / "data" / "roc_cache.json"
roc = json.loads(roc_path.read_text(encoding="utf-8")) if roc_path.exists() else None

best = status["best_mean_dice"]
cols = st.columns(4)
cols[0].markdown(f"""<div class="metric-box"><div class="metric-label">Epochs trained</div>
    <div class="metric-value">{status['epochs_completed']}</div>
    <div class="metric-note">checkpoint: best epoch</div></div>""", unsafe_allow_html=True)
cols[1].markdown(f"""<div class="metric-box"><div class="metric-label">Best mean Dice</div>
    <div class="metric-value" style="color:#059669;">{best:.3f}</div>
    <div class="metric-note">held-out validation patients</div></div>""", unsafe_allow_html=True)
if roc:
    wt = roc["regions"].get("WT")
    et = roc["regions"].get("ET")
    cols[2].markdown(f"""<div class="metric-box"><div class="metric-label">AUC · whole tumour</div>
        <div class="metric-value" style="color:#0284C7;">{wt['auc']:.3f}</div>
        <div class="metric-note">{roc['n_subjects']} held-out patients</div></div>""", unsafe_allow_html=True)
    cols[3].markdown(f"""<div class="metric-box"><div class="metric-label">AUC · enhancing</div>
        <div class="metric-value" style="color:#EF4444;">{et['auc']:.3f}</div>
        <div class="metric-note">hardest region in this cohort</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")

# ─────────────────────────────── training curves
with left:
    st.markdown('<div class="panel"><div class="panel-title">📉 Training loss & validation Dice</div>',
                unsafe_allow_html=True)
    history = loaders.load_training_history()
    if not history:
        st.info("No `history.json` for this run yet — train (or recover a log with "
                "`tools/history_from_log.py`) to populate this curve.")
    else:
        epochs = [h["epoch"] for h in history]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=[h["loss"] for h in history], name="train loss",
                                 line=dict(color="#0F172A", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=[h.get("mean_dice") for h in history], name="val mean Dice",
                                 line=dict(color="#059669", width=2), yaxis="y2"))
        layout = dict(PLOT_LAYOUT)
        layout["yaxis"] = dict(PLOT_LAYOUT["yaxis"], title="loss")
        layout["yaxis2"] = dict(title="Dice", overlaying="y", side="right",
                                range=[0, 1], gridcolor="rgba(0,0,0,0)")
        layout["xaxis"] = dict(PLOT_LAYOUT["xaxis"], title="epoch")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(history)} epochs recorded (epoch {epochs[0]}–{epochs[-1]}). "
                   "Loss is the DiceCE training loss; Dice is measured per held-out patient.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">🎯 Validation Dice by region</div>',
                unsafe_allow_html=True)
    if history:
        fig = go.Figure()
        for region, colour in (("ET", "#EF4444"), ("NC", "#10B981"), ("WT", "#0284C7")):
            key = f"dice_{region}"
            if any(key in h for h in history):
                fig.add_trace(go.Scatter(x=[h["epoch"] for h in history if key in h],
                                         y=[h[key] for h in history if key in h],
                                         name=region, line=dict(color=colour, width=2)))
        layout = dict(PLOT_LAYOUT)
        layout["yaxis"] = dict(PLOT_LAYOUT["yaxis"], title="Dice", range=[0, 1])
        layout["xaxis"] = dict(PLOT_LAYOUT["xaxis"], title="epoch")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ET trails the other regions throughout — expected in pediatric gliomas, "
                   "where DMG/DIPG enhances little or not at all.")
    else:
        st.info("No history available.")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────── ROC
with right:
    st.markdown('<div class="panel"><div class="panel-title">📈 ROC — held-out patients</div>',
                unsafe_allow_html=True)
    if not roc:
        st.info("No ROC cache yet. Build it with:\n\n"
                "```bash\npython -m utils.build_metrics_cache\n```")
    else:
        fig = go.Figure()
        for region, v in roc["regions"].items():
            if v is None:
                continue
            fig.add_trace(go.Scatter(x=v["fpr"], y=v["tpr"], name=f"{region} (AUC {v['auc']:.3f})",
                                     line=dict(color=REGION_COLOURS.get(region, "#0284C7"), width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="chance",
                                 line=dict(color="#94A3B8", width=1, dash="dash")))
        layout = dict(PLOT_LAYOUT)
        layout["xaxis"] = dict(PLOT_LAYOUT["xaxis"], title="false positive rate", range=[0, 1])
        layout["yaxis"] = dict(PLOT_LAYOUT["yaxis"], title="true positive rate", range=[0, 1])
        layout["height"] = 420
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"One-vs-rest per pixel over {roc['n_slices']} tumour-bearing slices from "
            f"{roc['n_subjects']} held-out patients ({roc['plane']} plane)."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if roc:
        st.markdown('<div class="panel"><div class="panel-title">🔬 How to read these AUCs</div>',
                    unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td class='meta-key'>{r}</td>"
            f"<td class='meta-val'><b>{v['auc']:.4f}</b></td>"
            f"<td class='meta-val' style='font-family:monospace;font-size:0.78rem;'>"
            f"{v['prevalence']*100:.1f}% of pixels</td></tr>"
            for r, v in roc["regions"].items() if v
        )
        st.markdown(f"""
        <table class="meta-table" style="width:100%;">
            <tr><td class="meta-key"><b>region</b></td><td class="meta-val"><b>AUC</b></td>
                <td class="meta-val"><b>prevalence</b></td></tr>
            {rows}
        </table>
        <div class="caveat-box">
            <b>AUC is the optimistic metric here; Dice is the honest one.</b> Tumour pixels are
            2–18% of each slice, and most background is trivially easy to rule out, which pushes
            per-pixel AUC high even when boundaries are imperfect. That is why AUC reads ~0.98
            while mean Dice is {best:.2f}. Quote them together, never AUC alone.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="panel">
    <div class="panel-title">🧾 Provenance</div>
    <table class="meta-table" style="width:100%;">
        <tr><td class="meta-key">Checkpoint</td>
            <td class="meta-val" style="font-family:monospace;font-size:0.75rem;">{status['path']}</td></tr>
        <tr><td class="meta-key">Architecture</td>
            <td class="meta-val" style="font-family:monospace;">{status['architecture']} · 2D · from scratch</td></tr>
        <tr><td class="meta-key">Augmentation / mixup</td>
            <td class="meta-val" style="font-family:monospace;">{status['use_augmentation']} / {status['use_mixup']}</td></tr>
        <tr><td class="meta-key">ROC evaluated on</td>
            <td class="meta-val">{roc['evaluated_on'] if roc else 'not computed'}</td></tr>
    </table>
    <div style="font-size:0.7rem;color:var(--text-faint);margin-top:10px;">
        Research use only. A small from-scratch U-Net trained for {status['epochs_completed']} epochs
        is not a clinically validated system.
    </div>
</div>
""", unsafe_allow_html=True)
