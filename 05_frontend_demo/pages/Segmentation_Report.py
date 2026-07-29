"""Segmentation report driven by a real checkpoint on real cached slices.

Every number on this page is measured at request time: the segmentation comes
from the trained model, the subregion percentages are voxel counts from that
prediction, and the Dice figures compare it against the cached ground truth.
Where a value cannot be measured, the page says so instead of showing a
placeholder -- this is a medical model, and an invented metric on screen is
indistinguishable from a real one to anyone reading it.
"""

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import apply_custom_theme
from utils import loaders
from utils.inference import (
    DEFAULT_CHECKPOINT,
    dice_per_region,
    load_model,
    predict_slice,
    region_breakdown,
)

st.set_page_config(page_title="Segmentation Report | NeuroFed AI", page_icon="📋", layout="wide")

# Shared light theme first; the rules below only add layout this page needs.
apply_custom_theme()

st.markdown("""
<style>
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:8px; }
.page-sub { font-size:0.92rem; }

.panel-title { font-size:0.95rem; font-weight:700; margin-bottom:18px; display:flex; align-items:center; gap:8px; }

.meta-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.meta-table tr:last-child { border-bottom:none; }
.meta-table td { padding:11px 8px; }
.meta-key { width:42%; font-size:0.78rem; font-family:'JetBrains Mono',monospace; }
.meta-val { font-weight:500; }

.interp-box { font-size:0.87rem; line-height:1.85; }
.caveat-box { font-size:0.8rem; line-height:1.7; margin-top:14px; }

.seg-row { margin-bottom:14px; }
.seg-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.seg-label { display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#475569; }
.seg-dot   { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.seg-pct   { font-size:0.8rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.seg-bar-bg  { height:6px; border-radius:4px; overflow:hidden; }
.seg-bar     { height:100%; border-radius:4px; }

.conf-inner {
    width:82px; height:82px; border-radius:50%;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.conf-num   { font-size:1.3rem; font-weight:800; letter-spacing:-0.02em; line-height:1; }
.conf-label { font-size:0.6rem; text-transform:uppercase; letter-spacing:0.05em; margin-top:2px; }

.status-bar {
    display:flex; justify-content:space-between; align-items:center;
    border-radius:10px; padding:10px 16px; margin-top:6px;
    font-size:0.72rem; font-family:'JetBrains Mono',monospace;
}
.status-left  { display:flex; align-items:center; gap:8px; }
.live-dot { width:6px; height:6px; border-radius:50%; background:#059669; animation:pulseDot 1.8s infinite; }
@keyframes pulseDot {
    0%  { box-shadow:0 0 0 0 rgba(5,150,105,.5); }
    70% { box-shadow:0 0 0 6px rgba(5,150,105,0); }
    100%{ box-shadow:0 0 0 0 rgba(5,150,105,0); }
}
</style>

<div class="page-hero">
    <div class="page-title">Clinical AI Segmentation Report</div>
    <div class="page-sub">Generated live from the trained BraTS-PEDs 2D segmentation checkpoint.</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  Preconditions -- be explicit when we cannot measure
# ══════════════════════════════════════════════════════
status = loaders.load_checkpoint_status()

if not status["available"]:
    st.error(
        "**No trained checkpoint found.** This page reports measured model output "
        "only, so it has nothing to show until a model is trained.\n\n"
        f"```\n{status['detail']}\n```"
    )
    st.stop()

if not loaders.cache_available():
    st.error(
        f"**Slice cache not found** at `{loaders.cache_dir()}`.\n\n"
        "Set the `NEUROFED_CACHE_2D` environment variable to the directory holding "
        "`<subject>/<plane>/slice_NNN.npz`."
    )
    st.stop()


# ══════════════════════════════════════════════════════
#  Controls
# ══════════════════════════════════════════════════════
subjects = loaders.list_demo_subjects()
c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    subject_id = st.selectbox("Subject", subjects, index=0)
with c2:
    plane = st.selectbox("Plane", ["axial", "coronal"], index=0)
with c3:
    tumor_idx = loaders.tumor_slice_indices(subject_id, plane)
    all_idx = loaders.tumor_slice_indices(subject_id, plane) or []
    if tumor_idx:
        slice_index = st.select_slider(
            f"Slice (showing the {len(tumor_idx)} tumour-bearing slices)",
            options=tumor_idx, value=tumor_idx[len(tumor_idx) // 2],
        )
    else:
        st.warning("No tumour-bearing slices cached for this subject/plane.")
        st.stop()

with st.spinner("Running the model…"):
    model, type_head, cfg, meta = load_model(DEFAULT_CHECKPOINT, str(loaders.cache_dir()))
    result = predict_slice(model, cfg, loaders.cache_dir(), subject_id, plane,
                           slice_index, type_head=type_head)

pred = result["prediction"]
truth = result["ground_truth"]
dice = dice_per_region(pred, truth)
regions = region_breakdown(pred)
tumor_voxels = int((pred > 0).sum())

st.markdown(f"""
<div class="status-bar">
    <div class="status-left"><span class="live-dot"></span>Live inference · 2D U-Net ·
        {meta['epochs_completed']} epoch(s) trained</div>
    <div class="status-right">Generated: {datetime.now().strftime("%Y-%m-%d  %H:%M")}</div>
</div>
<div style="margin-bottom:20px;"></div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([2, 1], gap="medium")

# ══════════════════════════════════════════════════════
#  LEFT — image, metadata, subregions
# ══════════════════════════════════════════════════════
with col_main:
    st.markdown('<div class="panel"><div class="panel-title">🖼️ Prediction vs Ground Truth</div>',
                unsafe_allow_html=True)

    t1c = result["image"][0]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.flipud(t1c), colorscale="Gray", showscale=False))
    overlay = np.where(pred > 0, pred, np.nan)
    fig.add_trace(go.Heatmap(z=np.flipud(overlay), colorscale="Turbo", opacity=0.55,
                             showscale=False, zmin=1, zmax=4))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=6, r=6, t=6, b=6), height=420,
                      yaxis=dict(scaleanchor="x", visible=False), xaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("T1c with the model's predicted segmentation overlaid. "
               "Ground-truth Dice for this slice is reported at right.")
    st.markdown("</div>", unsafe_allow_html=True)

    dice_cell = lambda v: "n/a" if v is None else f"{v:.3f}"
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📄 Scan & Inference Metadata</div>
        <table class="meta-table">
            <tr><td class="meta-key">Subject ID</td>
                <td class="meta-val" style="color:#38BDF8;font-family:'JetBrains Mono',monospace;font-weight:600;">{subject_id}</td></tr>
            <tr><td class="meta-key">Dataset Cohort</td>
                <td class="meta-val">BraTS-PEDs (pediatric)</td></tr>
            <tr><td class="meta-key">MRI Modalities</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">{' · '.join(cfg.modalities)}</td></tr>
            <tr><td class="meta-key">Slice</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">{plane} · index {slice_index} · {pred.shape[0]} × {pred.shape[1]}</td></tr>
            <tr><td class="meta-key">Predicted Tumour Area</td>
                <td class="meta-val" style="color:#00F2FE;font-weight:700;">{tumor_voxels:,} px</td></tr>
            <tr><td class="meta-key">Ground-truth Tumour Area</td>
                <td class="meta-val">{int((truth > 0).sum()):,} px</td></tr>
            <tr><td class="meta-key">Architecture</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">{meta['architecture']} · {meta['spatial_dims']}D</td></tr>
            <tr><td class="meta-key">Augmentation / Mixup</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">{meta['use_augmentation']} / {meta['use_mixup']}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">🧠 Predicted Subregion Breakdown</div>',
                unsafe_allow_html=True)
    if tumor_voxels == 0:
        st.info("The model predicted no tumour on this slice.")
    else:
        for row in regions:
            st.markdown(f"""
            <div class="seg-row">
                <div class="seg-top">
                    <div class="seg-label">
                        <div class="seg-dot" style="background:{row['colour']};box-shadow:0 0 6px {row['colour']}66;"></div>
                        {row['name']}
                    </div>
                    <span class="seg-pct" style="color:{row['colour']};">{row['percent']:.1f}%  ({row['voxels']:,} px)</span>
                </div>
                <div class="seg-bar-bg">
                    <div class="seg-bar" style="width:{min(row['percent'],100)}%;background:linear-gradient(90deg,{row['colour']}cc,{row['colour']});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    dominant = max(regions, key=lambda r: r["voxels"]) if tumor_voxels else None
    type_line = ""
    if "tumor_type" in result:
        tt = result["tumor_type"]
        type_line = (f"<br><br>The auxiliary tumour-type head reports "
                     f"<b>{tt['label'].replace('_', ' ')}</b> at "
                     f"{tt['confidence']*100:.1f}% — this is a <i>geometric imaging proxy</i>, "
                     f"not a histological diagnosis.")

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">🩺 Measured Findings</div>
        <div class="interp-box">
            On {plane} slice {slice_index} of <b>{subject_id}</b>, the model segmented
            <b>{tumor_voxels:,} px</b> of tumour
            {"— predominantly <b style='color:%s;'>%s</b> (%.1f%%)." % (dominant['colour'], dominant['name'], dominant['percent']) if dominant else "."}
            Against the cached ground truth for this slice, Dice is
            <b>ET {dice_cell(dice['ET'])}</b>, <b>TC {dice_cell(dice['TC'])}</b>,
            <b>WT {dice_cell(dice['WT'])}</b>.
            Mean softmax confidence over the predicted labels is
            <b>{result['confidence']*100:.1f}%</b>.{type_line}
        </div>
        <div class="caveat-box">
            <b>Research use only.</b> These are measured outputs of a small from-scratch
            U-Net trained for {meta['epochs_completed']} epoch(s) — not a clinically validated
            system, and not a diagnosis. "n/a" means the region is absent from both the
            prediction and the ground truth, so Dice is undefined rather than perfect.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  RIGHT — confidence, metrics, export
# ══════════════════════════════════════════════════════
with col_side:
    conf_pct = result["confidence"] * 100
    st.markdown(f"""
    <div class="panel" style="text-align:center;">
        <div class="panel-title" style="justify-content:center;">📊 Prediction Confidence</div>
        <div style="width:110px;height:110px;border-radius:50%;
             background:conic-gradient(#34D399 0% {conf_pct:.1f}%, rgba(255,255,255,0.07) {conf_pct:.1f}% 100%);
             display:flex;align-items:center;justify-content:center;margin:0 auto 12px auto;
             box-shadow:0 0 24px rgba(52,211,153,0.25);">
            <div class="conf-inner">
                <div class="conf-num">{conf_pct:.1f}%</div>
                <div class="conf-label">Confidence</div>
            </div>
        </div>
        <div style="font-size:0.75rem;color:#64748B;margin-top:4px;">
            Mean softmax probability of the<br>predicted class, this slice
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">⚡ Dice · This Slice</div>
        <table class="meta-table">
            <tr><td class="meta-key">ET</td><td class="meta-val" style="color:#EF4444;font-family:monospace;">{dice_cell(dice['ET'])}</td></tr>
            <tr><td class="meta-key">TC</td><td class="meta-val" style="color:#818CF8;font-family:monospace;">{dice_cell(dice['TC'])}</td></tr>
            <tr><td class="meta-key">WT</td><td class="meta-val" style="color:#38BDF8;font-family:monospace;">{dice_cell(dice['WT'])}</td></tr>
            <tr><td class="meta-key">Mean</td><td class="meta-val" style="color:#34D399;font-family:monospace;font-weight:700;">{dice_cell(dice['mean'])}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    best = status["best_mean_dice"]
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">🧪 Checkpoint</div>
        <table class="meta-table">
            <tr><td class="meta-key">Epochs trained</td>
                <td class="meta-val" style="font-family:monospace;">{status['epochs_completed']}</td></tr>
            <tr><td class="meta-key">Best val mean Dice</td>
                <td class="meta-val" style="color:#34D399;font-family:monospace;font-weight:700;">
                {f"{best:.4f}" if isinstance(best, (int, float)) else "n/a"}</td></tr>
            <tr><td class="meta-key">Tumour-type head</td>
                <td class="meta-val" style="font-family:monospace;">{"present" if status['has_type_head'] else "absent"}</td></tr>
        </table>
        <div style="font-size:0.7rem;color:#64748B;margin-top:10px;line-height:1.5;">
            Validation Dice is measured per patient on held-out patients during training,
            not on this slice.
        </div>
    </div>
    """, unsafe_allow_html=True)

    csv_rows = "region,dice\n" + "\n".join(
        f"{k},{'' if dice[k] is None else f'{dice[k]:.6f}'}" for k in ("ET", "TC", "WT", "mean")
    ) + "\n\nlabel,name,pixels,percent\n" + "\n".join(
        f"{r['label']},{r['name']},{r['voxels']},{r['percent']:.4f}" for r in regions
    )
    st.download_button("⬇  Download slice metrics (CSV)",
                       data=csv_rows,
                       file_name=f"{subject_id}_{plane}_{slice_index}_metrics.csv",
                       mime="text/csv", use_container_width=True)

    st.markdown("""
    <div style="font-size:0.68rem;color:rgba(100,116,139,0.5);text-align:center;margin-top:10px;font-family:'JetBrains Mono',monospace;">
        RESEARCH USE ONLY · NOT FOR CLINICAL DECISIONS
    </div>
    """, unsafe_allow_html=True)
