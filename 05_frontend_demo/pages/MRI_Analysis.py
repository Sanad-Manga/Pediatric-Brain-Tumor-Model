import streamlit as st
import plotly.graph_objects as go
import numpy as np

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
.upload-zone {
    border:2px dashed rgba(56,189,248,0.25); border-radius:12px;
    padding:24px; text-align:center; margin-bottom:16px;
    background:rgba(56,189,248,0.03);
    transition: border-color .2s ease;
}
.upload-zone:hover { border-color:rgba(56,189,248,0.5); }
.upload-hint { font-size:0.75rem; color:#64748B; margin-top:6px; }
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
.status-dot { display:inline-block; width:6px; height:6px; border-radius:50%;
    background:#34D399; margin-right:6px; animation:pulseDot 1.8s infinite; }
@keyframes pulseDot {
    0%  { box-shadow:0 0 0 0 rgba(52,211,153,.6); }
    70% { box-shadow:0 0 0 6px rgba(52,211,153,0); }
    100%{ box-shadow:0 0 0 0 rgba(52,211,153,0); }
}
</style>

<div class="page-hero">
    <div class="page-title">MRI Analysis Studio</div>
    <div class="page-sub">Multi-modality radiological workspace with 3D U-Net segmentation overlays and real-time inference.</div>
</div>
""", unsafe_allow_html=True)

col_ctrl, col_view = st.columns([1, 2], gap="medium")

with col_ctrl:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📁 Volume Input</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload NIfTI Volume (.nii / .nii.gz)", type=["nii", "gz"],
                                 label_visibility="collapsed")
    st.markdown('<div class="upload-hint">Supports .nii and .nii.gz · Max 500 MB</div>', unsafe_allow_html=True)

    modality = st.selectbox("MRI Modality", ["T1", "T1c (Contrast)", "T2", "FLAIR"],
                             help="Select the MRI sequence to visualize")
    slice_idx = st.slider("Axial Slice", 0, 155, 77,
                           help="Navigate through axial slices of the 3D volume")
    overlay = st.toggle("Show Segmentation Overlay", value=True)

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🧠 Subregion Statistics</div>', unsafe_allow_html=True)

    regions = [
        ("Enhancing Tumor (ET)", 24.2, "#EF4444"),
        ("Peritumoral Edema (ED)", 48.6, "#EAB308"),
        ("Cystic Component (CC)", 18.1, "#3B82F6"),
        ("Non-enhancing (NET)", 9.1, "#10B981"),
    ]
    for name, pct, color in regions:
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div class="subregion-row">
                <div class="subregion-left">
                    <div class="subregion-dot" style="background:{color};box-shadow:0 0 6px {color}55;"></div>
                    <span class="subregion-name">{name}</span>
                </div>
                <span class="subregion-pct" style="color:{color};">{pct}%</span>
            </div>
            <div class="subregion-bar-bg">
                <div class="subregion-bar" style="width:{pct*2}%;background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_view:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🖥️ Interactive Viewer & Segmentation Overlay</div>', unsafe_allow_html=True)

    np.random.seed(slice_idx)
    base = np.random.rand(96, 96) * 0.6
    # simulate tumor blobs
    cx, cy = 48, 52
    for i in range(96):
        for j in range(96):
            d = ((i-cx)**2 + (j-cy)**2)**0.5
            if d < 18: base[i,j] += 0.4 * max(0, 1 - d/18)
            if d < 10: base[i,j] += 0.3

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=base, colorscale='Greys', showscale=False, opacity=1.0))
    if overlay:
        overlay_z = np.zeros((96, 96))
        for i in range(96):
            for j in range(96):
                d = ((i-cx)**2 + (j-cy)**2)**0.5
                if d < 8:  overlay_z[i,j] = 3
                elif d < 13: overlay_z[i,j] = 1
                elif d < 18: overlay_z[i,j] = 2
        fig.add_trace(go.Heatmap(z=overlay_z, colorscale=[
            [0,'rgba(0,0,0,0)'], [0.01,'rgba(239,68,68,0)'],
            [0.33,'rgba(59,130,246,0.45)'], [0.66,'rgba(234,179,8,0.45)'],
            [1.0,'rgba(239,68,68,0.55)']
        ], showscale=False, opacity=0.75))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=420,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor='x')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="viewer-status">
        <div class="status-left"><span class="status-dot"></span>Inference Ready · FP16 · Slice {slice_idx}/155</div>
        <div class="status-right">Federated 3D U-Net + CORAL · {modality}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)