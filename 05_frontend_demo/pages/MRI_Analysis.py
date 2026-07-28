import numpy as np
import plotly.graph_objects as go
import streamlit as st

import backend as be

st.set_page_config(page_title="MRI Analysis Studio | NeuroFed AI", page_icon="🖥️", layout="wide")

# NOTE ON LAYOUT: do NOT wrap Streamlit widgets in a raw `<div class="panel">`
# emitted by st.markdown. Streamlit closes that div immediately, so the div
# renders as an empty box and every widget lands outside it — which is exactly
# why the panel headings used to appear *below* empty rounded rectangles.
# Use st.container(border=True) and style the real wrapper instead.
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

/* Style Streamlit's real bordered container so panels look like the mockup while
   still actually containing their widgets. In Streamlit 1.60 `border=True` puts
   the border on the inner stVerticalBlock; :has() with a direct-child selector
   scopes this to blocks holding one of our panel headings, leaving the column
   and layout wrappers untouched. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .panel-head) {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 20px 22px;
}
.panel-head {
    font-size:0.9rem; font-weight:700; color:#F3F4F6; margin:0 0 14px 0;
    display:flex; align-items:center; gap:8px;
}
.subregion-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; }
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
.real-box {
    background:rgba(52,211,153,0.07); border:1px solid rgba(52,211,153,0.3);
    border-radius:10px; padding:10px 14px; margin-top:10px;
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
    out = be.run_inference(be.load_volume(subject_id))
    return out["pred_seg"], out["dice"]


status = be.model_status()
subjects = be.list_subjects()

source = st.radio(
    "Scan source",
    ["Dataset subject", "Upload a scan (second opinion)"],
    horizontal=True,
)

# ══════════════════════════════════════════════════════════════════════════
#  UPLOAD PATH — clinician submits a scan for a second opinion
# ══════════════════════════════════════════════════════════════════════════
if source == "Upload a scan (second opinion)":
    st.markdown('<div class="panel-head">📤 Upload MRI Scan</div>', unsafe_allow_html=True)

    st.error(
        "**Not for clinical use.** This tool has no trained model and no regulatory "
        "clearance. It cannot provide a second opinion on a real patient. Do not upload "
        "identifiable patient data, and do not use any output here to inform care.",
        icon="⛔",
    )

    upload_mode = st.radio(
        "Upload format",
        ["Four NIfTI files (one per modality)", "Single .npz bundle"],
        horizontal=True,
    )

    volume = None
    with st.container(border=True):
        if upload_mode == "Single .npz bundle":
            st.markdown(
                '<div class="panel-head">Cache-format bundle</div>', unsafe_allow_html=True)
            st.caption(
                f"One .npz containing keys {', '.join(be.MODALITIES)} "
                "(and optionally `seg` for ground truth)."
            )
            bundle = st.file_uploader("NPZ bundle", type=["npz"], label_visibility="collapsed")
            if bundle is not None:
                try:
                    volume = be.volume_from_npz_bundle(bundle, label=bundle.name)
                except Exception as exc:
                    st.error(f"Could not read the bundle: {exc}")
        else:
            st.markdown('<div class="panel-head">One file per modality</div>',
                        unsafe_allow_html=True)
            st.caption(
                "All four are required — the network was trained to read four channels, "
                "so a missing one would be fed as zeros and produce a confidently wrong answer."
            )
            cols = st.columns(4)
            files = {}
            for col, m in zip(cols, be.MODALITIES):
                files[m] = col.file_uploader(
                    be.MODALITY_LABELS[m], type=["nii", "gz", "npz"], key=f"up_{m}"
                )
            if all(files[m] is not None for m in be.MODALITIES):
                try:
                    volume = be.volume_from_uploads(files, label="uploaded scan")
                except Exception as exc:
                    st.error(f"Could not read the uploaded files: {exc}")
            else:
                got = sum(files[m] is not None for m in be.MODALITIES)
                st.info(f"{got} of 4 modalities provided.", icon="📄")

    if volume is None:
        st.stop()

    st.success(
        f"Loaded and resampled to {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}. "
        "Intensities z-scored over brain voxels, matching how the model expects its input.",
        icon="✅",
    )

    with st.spinner("Running 3D U-Net inference…"):
        result = be.run_inference(volume)
    pred = result["pred_seg"]

    up_l, up_r = st.columns([1, 2], gap="medium")

    with up_l:
        with st.container(border=True):
            st.markdown('<div class="panel-head">🖼 View</div>', unsafe_allow_html=True)
            up_mod = st.selectbox("Modality", list(be.MODALITIES),
                                  format_func=lambda m: be.MODALITY_LABELS[m], key="up_mod")
            up_plane = st.radio("Plane", ["Axial", "Coronal", "Sagittal"],
                                horizontal=True, key="up_plane")
            up_ext = be.plane_extent(pred.shape, up_plane)
            up_default = be.busiest_slice(pred, up_plane)
            up_slice = st.slider(f"{up_plane} slice", 0, up_ext - 1, up_default, key="up_slice")
            up_overlay = st.toggle("Show predicted mask", value=True)

        with st.container(border=True):
            st.markdown('<div class="panel-head">🧠 Predicted Subregions</div>',
                        unsafe_allow_html=True)
            for row in be.subregion_stats(pred):
                pct = row["pct_of_tumor"]
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div class="subregion-row">
                        <div class="subregion-left">
                            <div class="subregion-dot" style="background:{row['color']};"></div>
                            <span class="subregion-name">{row['name']}</span>
                        </div>
                        <span class="subregion-pct" style="color:{row['color']};">{pct:.1f}%</span>
                    </div>
                    <div class="subregion-bar-bg">
                        <div class="subregion-bar" style="width:{min(pct, 100):.1f}%;background:{row['color']};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with up_r:
        with st.container(border=True):
            st.markdown('<div class="panel-head">🖥️ Predicted Segmentation</div>',
                        unsafe_allow_html=True)
            up_img = be.slice_of(volume.raw[up_mod], up_slice, up_plane).T
            up_fig = go.Figure()
            up_fig.add_trace(go.Heatmap(z=up_img, colorscale="Greys_r",
                                        showscale=False, hoverinfo="skip"))
            if up_overlay:
                up_mask = be.slice_of(pred, up_slice, up_plane).T
                for lbl, colour in be.CLASS_COLORS.items():
                    up_fig.add_trace(go.Heatmap(
                        z=np.where(up_mask == lbl, 1.0, np.nan),
                        colorscale=[[0, colour], [1, colour]],
                        showscale=False, opacity=0.55, hoverinfo="skip"))
            up_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0), height=460,
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"))
            st.plotly_chart(up_fig, use_container_width=True)

    if result["dice"] is None:
        st.info(
            "No Dice or ROC for this scan: an uploaded scan has no expert ground-truth "
            "mask to score against. Include a `seg` key in an .npz bundle if you have one.",
            icon="ℹ️",
        )
    else:
        d = result["dice"]
        st.markdown("#### Per-region Dice vs the supplied ground truth")
        dc = st.columns(3)
        for c, r in zip(dc, ("ET", "NC", "WT")):
            c.metric(r, f"{d[r]:.3f}")

    if not status.trained:
        st.warning(
            "The mask above came from a real forward pass, but the network is UNTRAINED — "
            f"{status.detail} Treat this output as a pipeline demonstration only: it shows "
            "that a clinician's scan can be ingested, preprocessed and segmented end to end, "
            "not that the segmentation is correct.",
            icon="⚠️",
        )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  DATASET PATH
# ══════════════════════════════════════════════════════════════════════════
col_ctrl, col_view = st.columns([1, 2], gap="medium")

with col_ctrl:
    with st.container(border=True):
        st.markdown('<div class="panel-head">📁 Subject Selection</div>', unsafe_allow_html=True)
        cohort = st.selectbox("Cohort", list(subjects.keys()), index=2,
                              help="Hospital A / B are training clients; Held-out is the test institution")
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
            "Overlay", ["Ground truth", "Model prediction", "None"],
            help="Ground truth is the real expert annotation. Model prediction runs the real 3D U-Net.")

    with st.container(border=True):
        st.markdown('<div class="panel-head">🧠 Subregion Statistics</div>', unsafe_allow_html=True)
        st.caption("Measured from this subject's ground-truth mask.")
        for row in be.subregion_stats(seg):
            pct = row["pct_of_tumor"]
            st.markdown(f"""
            <div style="margin-bottom:10px;">
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
            unsafe_allow_html=True)

with col_view:
    with st.container(border=True):
        st.markdown('<div class="panel-head">🖥️ Interactive Viewer</div>', unsafe_allow_html=True)

        img_slice = be.slice_of(raw[modality], slice_idx, plane).T
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=img_slice, colorscale="Greys_r", showscale=False, hoverinfo="skip"))

        mask_source, dice_scores = None, None
        if overlay_mode == "Ground truth":
            mask_source = seg
        elif overlay_mode == "Model prediction":
            mask_source, dice_scores = _infer(subject_id)

        if mask_source is not None:
            mask_slice = be.slice_of(mask_source, slice_idx, plane).T
            for label, colour in be.CLASS_COLORS.items():
                fig.add_trace(go.Heatmap(
                    z=np.where(mask_slice == label, 1.0, np.nan),
                    colorscale=[[0, colour], [1, colour]],
                    showscale=False, opacity=0.55, hoverinfo="skip"))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0), height=460,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"))
        st.plotly_chart(fig, use_container_width=True)

        tumor_in_slice = int((be.slice_of(seg, slice_idx, plane) > 0).sum())
        st.markdown(f"""
        <div class="viewer-status">
            <div class="status-left">{subject_id} · {cohort_name} · {plane} {slice_idx}/{extent - 1}</div>
            <div class="status-right">{be.MODALITY_LABELS[modality]} · {tumor_in_slice:,} tumor voxels in slice</div>
        </div>
        """, unsafe_allow_html=True)

    if dice_scores is not None:
        if status.trained:
            st.markdown("#### Per-region Dice vs ground truth")
            c1, c2, c3 = st.columns(3)
            for col, region in zip((c1, c2, c3), ("ET", "NC", "WT")):
                col.metric(region, f"{dice_scores[region]:.3f}")
            st.caption(f"Checkpoint: `{status.checkpoint.name}`")
        else:
            st.warning(
                f"This prediction is meaningless — no trained model exists yet. {status.detail} "
                f"Measured Dice for this subject: ET {dice_scores['ET']:.3f} · "
                f"NC {dice_scores['NC']:.3f} · WT {dice_scores['WT']:.3f} — essentially zero, "
                "exactly what randomly initialized weights should give. Switch the overlay to "
                "Ground truth for the real expert annotation.",
                icon="⚠️",
            )
