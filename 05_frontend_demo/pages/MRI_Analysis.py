"""MRI Analysis Studio — accepts a patient folder (axial + coronal .npz slices)."""
import io
import re
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MRI Analysis Studio | NeuroPeds AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
* { font-family: 'Inter', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif; }

.page-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
    border: 1px solid rgba(14, 165, 233, 0.15);
    border-radius: 20px;
    padding: 30px 40px; margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.02);
}
.page-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; font-family: 'Outfit', sans-serif;
    color: #0F172A; margin-bottom: 8px;
}
.page-sub { color: #475569; font-size: 0.95rem; line-height: 1.6; }

.section-header {
    font-size: 1.1rem; font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif;
    display: flex; align-items: center; gap: 8px;
    padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;
}
.info-box {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 12px 16px; font-size: 0.85rem; color: #475569; line-height: 1.6; margin-bottom: 16px;
}
.info-box b { color: #0EA5E9; }

.plane-chip {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:0.75rem; font-weight:600; font-family:monospace; margin-right: 6px;
}
.subregion-row { margin-bottom:12px; }
.subregion-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
.subregion-label { display:flex; align-items:center; gap:8px; font-size:0.85rem; color:#475569; font-weight: 500;}
.subregion-dot  { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.subregion-pct  { font-size:0.85rem; font-weight:600; font-family:monospace; }
.subregion-bar-bg { height:8px; border-radius:4px; background:#F1F5F9; overflow:hidden; }
.subregion-bar    { height:100%; border-radius:4px; }

.viewer-status {
    display:flex; justify-content:space-between; align-items:center;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 12px 18px; margin-top: 16px;
    font-size: 0.85rem; font-weight: 500;
}
.status-left  { color: #0F172A; display:flex; align-items:center; gap:8px; font-weight: 600; }
.status-right { color: #64748B; font-family: monospace; font-size: 0.75rem; }
.status-dot {
    width:8px; height:8px; border-radius:50%; background:#10B981;
    animation: pulse-green 2s infinite; display:inline-block;
}

.disclaimer {
    background: #FEF2F2;
    border-left: 4px solid #EF4444; border-radius: 12px; padding: 16px 20px;
    font-size: 0.9rem; color: #991B1B; line-height: 1.5; margin-top: 24px;
    display: flex; align-items: flex-start; gap: 12px;
}
.disclaimer-icon { font-size: 1.4rem; }
.disclaimer-content b { color: #7F1D1D; font-size: 0.95rem; display: block; margin-bottom: 4px; }

/* Customizing Streamlit containers to look like cards */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
</style>

<div class="page-hero">
    <div class="page-title">🖥️ MRI Analysis Studio</div>
    <div class="page-sub">Upload patient scans (axial + coronal .npz slices) for real-time inference and multi-plane segmentation overlays.</div>
</div>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODALITY_CHANNELS = {"T1c (Contrast)": 0, "T1n": 1, "T2-FLAIR": 2, "T2w": 3}
MODALITY_COLORSCALES = {
    "T1c (Contrast)": "Greys", "T1n": "Greys",
    "T2-FLAIR": "YlOrRd",     "T2w": "Blues",
}
LABEL_NAMES    = {1: "Enhancing Tumor (ET)", 2: "Non-enhancing Core (NETC)",
                  3: "Cystic Component (CC)", 4: "Peritumoral Edema (ED)"}
LABEL_COLORS   = {1: "#EF4444", 2: "#10B981", 3: "#3B82F6", 4: "#EAB308"}


def slice_num(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def overlay_colorscale():
    return [
        [0.00,"rgba(0,0,0,0)"],[0.20,"rgba(0,0,0,0)"],
        [0.21,"rgba(239,68,68,0.65)"],[0.40,"rgba(239,68,68,0.65)"],
        [0.41,"rgba(16,185,129,0.60)"],[0.60,"rgba(16,185,129,0.60)"],
        [0.61,"rgba(59,130,246,0.60)"],[0.80,"rgba(59,130,246,0.60)"],
        [0.81,"rgba(234,179,8,0.60)"],[1.00,"rgba(234,179,8,0.60)"],
    ]


def read_npz(f):
    buf = io.BytesIO(f.read())
    data = np.load(buf, allow_pickle=False)
    if "image" not in data.files or "mask" not in data.files:
        return None, None
    return data["image"], data["mask"]


# ─── File upload ──────────────────────────────────────────────────────────────
col_ctrl, col_view = st.columns([1, 2], gap="large")

with col_ctrl:
    st.markdown('<div class="section-header">📂 Data Source</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Upload <b>.npz files</b> containing preprocessed MRI slices.<br>
        Required format: <b>image</b> (4, H, W) float32 and <b>mask</b> (1, H, W) uint8.
    </div>
    """, unsafe_allow_html=True)

    axial_files = st.file_uploader(
        "Upload AXIAL slices",
        type=["npz"],
        accept_multiple_files=True,
        key="axial_uploader",
    )

    coronal_files = st.file_uploader(
        "Upload CORONAL slices",
        type=["npz"],
        accept_multiple_files=True,
        key="coronal_uploader",
    )
    
    st.session_state["axial_files"] = axial_files
    st.session_state["coronal_files"] = coronal_files

    # ── Parse uploaded files into planes ─────────────────────────────────────
    planes = {"axial": {}, "coronal": {}}
    
    if axial_files:
        for f in axial_files:
            idx = slice_num(f.name)
            planes["axial"][idx] = f

    if coronal_files:
        for f in coronal_files:
            idx = slice_num(f.name)
            planes["coronal"][idx] = f

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Viewer Settings</div>', unsafe_allow_html=True)

    available_planes = [p for p, d in planes.items() if d]
    if available_planes:
        plane_sel = st.radio("Select Plane", available_planes, horizontal=True)
        slice_dict = planes[plane_sel]
        sorted_indices = sorted(slice_dict.keys())
        slice_idx = st.select_slider(
            "Navigate Slice", options=sorted_indices,
            value=sorted_indices[len(sorted_indices)//2]
        )
    else:
        plane_sel = "axial"
        slice_idx = None

    modality = st.selectbox(
        "MRI Modality", list(MODALITY_CHANNELS.keys()),
    )
    show_overlay = st.toggle("Show Tumor Segmentation", value=True)

    # ── Subregion stats ───────────────────────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Slice Statistics</div>', unsafe_allow_html=True)

    if slice_idx is not None:
        f = planes[plane_sel][slice_idx]
        f.seek(0)
        img_arr, mask_arr = read_npz(f)
        if img_arr is not None:
            mask_2d = mask_arr[0] if mask_arr.ndim == 3 else mask_arr
            total   = int((mask_2d > 0).sum())
            if total == 0:
                st.info("No tumour labels detected in this slice.")
            else:
                for lid, lname in LABEL_NAMES.items():
                    vox  = int((mask_2d == lid).sum())
                    pct  = 100.0 * vox / total if total > 0 else 0
                    color = LABEL_COLORS[lid]
                    st.markdown(f"""
                    <div class="subregion-row">
                        <div class="subregion-top">
                            <div class="subregion-label">
                                <div class="subregion-dot" style="background:{color};box-shadow:0 0 6px {color}66;"></div>
                                {lname}
                            </div>
                            <span class="subregion-pct" style="color:{color};">{vox:,} px · {pct:.1f}%</span>
                        </div>
                        <div class="subregion-bar-bg">
                            <div class="subregion-bar"
                                 style="width:{min(pct,100)}%;background:{color};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.caption("Awaiting data upload to compute statistics.")

# ─── Viewer ───────────────────────────────────────────────────────────────────
with col_view:
    st.markdown('<div class="section-header">🔍 Interactive Viewer</div>',
                unsafe_allow_html=True)

    ch     = MODALITY_CHANNELS[modality]
    cscale = MODALITY_COLORSCALES[modality]

    if slice_idx is not None and img_arr is not None:
        channel_img = img_arr[ch]
        mask_2d_v   = mask_arr[0] if mask_arr.ndim == 3 else mask_arr
        h, w = channel_img.shape

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=channel_img, colorscale=cscale, showscale=False, opacity=1.0,
            hovertemplate="Intensity: %{z:.3f}<extra></extra>"
        ))
        if show_overlay and int((mask_2d_v > 0).sum()) > 0:
            fig.add_trace(go.Heatmap(
                z=mask_2d_v.astype(float),
                colorscale=overlay_colorscale(),
                zmin=0, zmax=4, showscale=False, opacity=1.0,
                hovertemplate="Label: %{z:.0f}<extra></extra>",
            ))

        n_ax = len(planes.get("axial", {}))
        n_co = len(planes.get("coronal", {}))
        status_right = f"Real data · {h}×{w} · {modality} · slice {slice_idx}"
        source_label = (f"📂 Patient loaded — "
                        f"<span class='plane-chip' style='background:#EFF6FF;color:#1D4ED8;'>"
                        f"{n_ax} axial</span>"
                        f"<span class='plane-chip' style='background:#F0FDF4;color:#166534;'>"
                        f"{n_co} coronal</span>")
    else:
        # Demo fallback
        MKEYS = list(MODALITY_CHANNELS.keys())
        rng   = np.random.default_rng(77 + MKEYS.index(modality)*1000)
        csf_bright = modality in ("T2-FLAIR","T2w")
        cb = 0.5 if modality=="T1c (Contrast)" else (0.18 if modality=="T2-FLAIR" else 0.0)
        base = rng.random((96,96)) * 0.07
        cx,cy = 48,52
        for i in range(96):
            for j in range(96):
                d=((i-cx)**2+(j-cy)**2)**0.5
                if ((i-48)**2+(j-48)**2)**0.5<44:
                    base[i,j]+=0.35+0.1*np.sin(i/5)*np.cos(j/5) if csf_bright else 0.45+0.12*np.cos(i/4)*np.sin(j/4)
                if ((i-44)**2+(j-50)**2)**0.5<7:
                    base[i,j]=0.88 if csf_bright else 0.04
                if d<18: base[i,j]+=(0.3+cb)*max(0,1-d/18)
                if d<10: base[i,j]+=0.18+cb*0.5
        base=np.clip(base,0,1)
        fig=go.Figure()
        fig.add_trace(go.Heatmap(z=base,colorscale=cscale,showscale=False,opacity=1.0))
        if show_overlay:
            ov=np.zeros((96,96))
            for i in range(96):
                for j in range(96):
                    d=((i-cx)**2+(j-cy)**2)**0.5
                    if d<8: ov[i,j]=1
                    elif d<13: ov[i,j]=3
                    elif d<18: ov[i,j]=4
            fig.add_trace(go.Heatmap(z=ov,colorscale=overlay_colorscale(),zmin=0,zmax=4,showscale=False,opacity=1.0))
        h,w=96,96
        status_right=f"Synthetic · 96×96 · {modality}"
        source_label="🔬 Synthetic Demo (Upload patient .npz slices for real data)"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0), height=520,
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,scaleanchor="x"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if slice_idx is not None and img_arr is not None:
        tumor_detected = int((mask_2d_v > 0).sum()) > 0
        if tumor_detected:
            dominant_lid = max(LABEL_NAMES.keys(), key=lambda lid: int((mask_2d_v == lid).sum()))
            dom_name = LABEL_NAMES[dominant_lid]
    else:
        tumor_detected = True 
        dom_name = "Cystic Component (CC)"

    if tumor_detected:
        tumor_status = f"🚨 Tumor Detected — Predominantly <b>{dom_name}</b>"
        tumor_color = "#991B1B"
        tumor_bg = "#FEF2F2"
        tumor_border = "#FECACA"
    else:
        tumor_status = "✅ No Tumor Detected in this slice"
        tumor_color = "#065F46"
        tumor_bg = "#ECFDF5"
        tumor_border = "#A7F3D0"

    st.markdown(f"""
    <div class="viewer-status">
        <div class="status-left"><span class="status-dot"></span>{source_label}</div>
        <div class="status-right">{status_right}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="padding:16px 20px; background:{tumor_bg}; border:1px solid {tumor_border}; border-radius:12px; color:{tumor_color}; font-size:1.05rem; font-weight: 500; display:flex; align-items:center; gap:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); margin-top:20px;">
        {tumor_status}
    </div>
    """, unsafe_allow_html=True)

# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-content">
        <b>Research Use Only</b>
        NeuroPeds AI is a research-grade decision support platform. It does not replace independent clinical diagnosis or professional medical judgment.
    </div>
</div>
""", unsafe_allow_html=True)