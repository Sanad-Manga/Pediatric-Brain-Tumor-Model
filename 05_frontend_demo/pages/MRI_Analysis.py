import numpy as np
import plotly.graph_objects as go
import streamlit as st

import backend as be

st.set_page_config(page_title="MRI Analysis Studio | NeuroFed AI", page_icon="🖥️", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 20% 50%, rgba(56,189,248,0.09), transparent 50%),
                linear-gradient(135deg,#0B1628,#0F172A);
    border:1px solid rgba(56,189,248,0.2); border-radius:20px; padding:40px 44px; margin-bottom:28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#fff 40%,#38BDF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#94A3B8; font-size:0.92rem; }
.panel { background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:24px; }
.panel-title { font-size:0.9rem; font-weight:700; color:#F3F4F6; margin-bottom:16px; display:flex; align-items:center; gap:8px; }
.divider-line { height:1px; background:rgba(255,255,255,0.07); margin:16px 0; }
.subregion-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
.subregion-left { display:flex; align-items:center; gap:10px; }
.subregion-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.subregion-name { font-size:0.8rem; color:#94A3B8; }
.subregion-pct  { font-size:0.8rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.subregion-bar-bg { height:4px; border-radius:4px; background:rgba(255,255,255,0.06); overflow:hidden; margin-top:3px; }
.subregion-bar { height:100%; border-radius:4px; }
.viewer-status {
    display:flex; justify-content:space-between; align-items:center;
    background:rgba(5,10,31,0.7); border:1px solid rgba(56,189,248,0.15);
    border-radius:10px; padding:10px 14px; margin-top:12px;
    font-size:0.75rem; font-family:'JetBrains Mono',monospace;
}
.status-left { color:#38BDF8; }
.status-right { color:#64748B; }
.warn-box {
    background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.35);
    border-radius:12px; padding:14px 18px; margin-top:18px;
    font-size:0.82rem; color:#FDE68A; line-height:1.7;
}
.real-box {
    background:rgba(52,211,153,0.07); border:1px solid rgba(52,211,153,0.3);
    border-radius:10px; padding:10px 14px; margin-top:12px;
    font-size:0.72rem; color:#A7F3D0; font-family:'JetBrains Mono',monospace;
}
</style>

<div class="page-hero">
    <div class="page-title">MRI Analysis Studio</div>
    <div class="page-sub">Multi-modality radiological workspace over the real BraTS-PEDs 96³ cache.</div>
</div>
""", unsafe_allow_html=True)

if not be.cache_available():
    st.error(
        f"Cache directory not found: `{be.CACHE_DIR}`\n\n"
        "Set the `NEUROFED_CACHE` environment variable to the shared 96-cube cache."
    )
    st.stop()


@st.cache_data(show_spinner="Loading volume from cache…")
def _load(subject_id: str):
    v = be.load_volume(subject_id)
    return v.raw, v.seg, v.cohort


@st.cache_data(show_spinner="Running 3D U-Net inference…")
def _infer(subject_id: str):
    v = be.load_volume(subject_id)
    out = be.run_inference(v)
    return out["pred_seg"], out["dice"]


subjects = be.list_subjects()
status = be.model_status()

col_ctrl, col_view = st.columns([1, 2], gap="medium")

with col_ctrl:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📁 Subject Selection</div>', unsafe_allow_html=True)

    cohort = st.selectbox("Cohort", list(subjects.keys()), index=2,
                          help="Hospital A / B are the training clients; Held-out is the test institution")
    subject_id = st.selectbox("Subject", subjects[cohort],
                              help="Real subject IDs, read from the shared manifests")

    raw, seg, cohort_name = _load(subject_id)

    modality = st.selectbox("MRI Modality", list(be.MODALITIES), index=0,
                            format_func=lambda m: be.MODALITY_LABELS[m])
    plane = st.radio("Plane", ["Axial", "Coronal", "Sagittal"], horizontal=True)

    extent = be.plane_extent(seg.shape, plane)
    default_slice = be.busiest_slice(seg, plane)
    slice_idx = st.slider(f"{plane} slice", 0, extent - 1, default_slice,
                          help=f"Defaults to the slice with the most tumor ({default_slice})")

    overlay_mode = st.radio(
        "Overlay",
        ["Ground truth", "Model prediction", "None"],
        help="Ground truth is the real expert annotation. Model prediction runs the real 3D U-Net.",
    )

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🧠 Subregion Statistics</div>', unsafe_allow_html=True)
    st.caption("Measured from this subject's ground-truth mask.")

    for row in be.subregion_stats(seg):
        pct = row["pct_of_tumor"]
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div class="subregion-row">
                <div class="subregion-left">
                    <div class="subregion-dot" style="background:{row['color']};box-shadow:0 0 6px {row['color']}55;"></div>
                    <span class="subregion-name">{row['name']}</span>
                </div>
                <span class="subregion-pct" style="color:{row['color']};">{pct:.1f}%</span>
            </div>
            <div class="subregion-bar-bg">
                <div class="subregion-bar" style="width:{min(pct, 100):.1f}%;background:{row['color']};"></div>
            </div>
            <div style="font-size:0.66rem;color:#64748B;font-family:'JetBrains Mono',monospace;margin-top:3px;">
                {row['voxels']:,} voxels · ~{row['volume_ml']:.1f} mL
            </div>
        </div>
        """, unsafe_allow_html=True)

    total_vox = int((seg > 0).sum())
    st.markdown(
        f'<div class="real-box">Whole tumor: {total_vox:,} voxels · '
        f'~{total_vox * be.VOXEL_MM3 / 1000:.1f} mL (est.)</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_view:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🖥️ Interactive Viewer</div>', unsafe_allow_html=True)

    img_slice = be.slice_of(raw[modality], slice_idx, plane).T
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=img_slice, colorscale="Greys_r", showscale=False, hoverinfo="skip"))

    mask_source = None
    dice_scores = None
    if overlay_mode == "Ground truth":
        mask_source = seg
    elif overlay_mode == "Model prediction":
        mask_source, dice_scores = _infer(subject_id)

    if mask_source is not None:
        mask_slice = be.slice_of(mask_source, slice_idx, plane).T
        # One trace per class so overlay colours map exactly to label IDs.
        for label, colour in be.CLASS_COLORS.items():
            m = np.where(mask_slice == label, 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                z=m, colorscale=[[0, colour], [1, colour]],
                showscale=False, opacity=0.55, hoverinfo="skip",
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=460,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
    )
    st.plotly_chart(fig, use_container_width=True)

    tumor_in_slice = int((be.slice_of(seg, slice_idx, plane) > 0).sum())
    st.markdown(f"""
    <div class="viewer-status">
        <div class="status-left">{subject_id} · {cohort_name} · {plane} {slice_idx}/{extent - 1}</div>
        <div class="status-right">{be.MODALITY_LABELS[modality]} · {tumor_in_slice:,} tumor voxels in slice</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if dice_scores is not None:
        if status.trained:
            st.markdown("#### Per-region Dice vs ground truth")
            c1, c2, c3 = st.columns(3)
            for col, region in zip((c1, c2, c3), ("ET", "NC", "WT")):
                col.metric(region, f"{dice_scores[region]:.3f}")
            st.caption(f"Checkpoint: `{status.checkpoint.name}`")
        else:
            st.markdown(f"""
            <div class="warn-box">
                <b>⚠ This prediction is meaningless — no trained model exists yet.</b><br>
                {status.detail}<br><br>
                Real measured Dice against this subject's ground truth:
                <b>ET {dice_scores['ET']:.3f} · NC {dice_scores['NC']:.3f} · WT {dice_scores['WT']:.3f}</b>
                — essentially zero, which is exactly what randomly initialized weights
                should produce. It is shown rather than hidden so the state is unambiguous.<br><br>
                Switch the overlay to <b>Ground truth</b> for the real expert annotation.
                Once <code>01_model_federated</code> produces a checkpoint, set
                <code>NEUROFED_CHECKPOINT</code> and this panel becomes real numbers automatically.
            </div>
            """, unsafe_allow_html=True)
