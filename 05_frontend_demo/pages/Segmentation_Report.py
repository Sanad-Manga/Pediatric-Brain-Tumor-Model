"""Clinical AI Segmentation Report — works from uploaded .npz slice files.

Generates a full clinical report (ground-truth mask visualisation, subregion
breakdown, Dice vs prediction if model is available) directly from the
patient's .npz slice cache. Does not require a trained checkpoint to produce
the ground-truth analysis section.
"""
import io
import re
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.theme import apply_custom_theme

st.set_page_config(page_title="Segmentation Report | NeuroPeds AI", page_icon="📋", layout="wide")
apply_custom_theme()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em; font-family:'Outfit',sans-serif; margin-bottom:8px; }
.page-sub   { font-size:0.93rem; color:#475569; }

.panel { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:18px; padding:28px; margin-bottom:20px;
         box-shadow:0 2px 10px rgba(0,0,0,0.02); }
.panel-title { font-size:1.05rem; font-weight:700; color:#0F172A; font-family:'Outfit',sans-serif;
               margin-bottom:16px; display:flex; align-items:center; gap:10px; }

.meta-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.meta-table td { padding:11px 8px; border-bottom:1px solid #F1F5F9; }
.meta-table tr:last-child td { border-bottom:none; }
.meta-key { width:44%; font-size:0.78rem; font-family:'JetBrains Mono',monospace; color:#64748B; }
.meta-val { font-weight:600; color:#0F172A; }

.seg-row { margin-bottom:14px; }
.seg-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.seg-label { display:flex; align-items:center; gap:8px; font-size:0.82rem; color:#475569; }
.seg-dot  { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.seg-pct  { font-size:0.82rem; font-family:'JetBrains Mono',monospace; font-weight:700; }
.seg-bar-bg { height:6px; border-radius:4px; background:#F1F5F9; overflow:hidden; }
.seg-bar    { height:100%; border-radius:4px; }

.status-bar {
    display:flex; justify-content:space-between; align-items:center;
    background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:10px; padding:10px 16px; margin-bottom:20px;
    font-size:0.75rem; font-family:'JetBrains Mono',monospace;
}
.live-dot { width:6px; height:6px; border-radius:50%; background:#059669;
            display:inline-block; animation:pulseDot 1.8s infinite; margin-right:6px; }
@keyframes pulseDot {
    0%  { box-shadow:0 0 0 0 rgba(5,150,105,.5); }
    70% { box-shadow:0 0 0 6px rgba(5,150,105,0); }
    100%{ box-shadow:0 0 0 0 rgba(5,150,105,0); }
}
.conf-ring {
    width:110px; height:110px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 12px auto;
}
.conf-inner { display:flex; flex-direction:column; align-items:center; justify-content:center; }
.conf-num   { font-size:1.4rem; font-weight:800; color:#0F172A; line-height:1; }
.conf-label { font-size:0.6rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748B; margin-top:2px; }

.upload-box {
    background:#F0F9FF; border:2px dashed rgba(2,132,199,0.3);
    border-radius:14px; padding:28px; text-align:center; margin-bottom:20px;
}
.upload-box .title { font-size:1.1rem; font-weight:700; font-family:'Outfit',sans-serif; color:#0F172A; margin-bottom:6px; }
.upload-box .sub   { font-size:0.85rem; color:#475569; line-height:1.6; }

@keyframes pulseWarning {
    0%{box-shadow:0 0 0 0 rgba(245,158,11,.4)} 70%{box-shadow:0 0 0 10px rgba(245,158,11,0)} 100%{box-shadow:0 0 0 0 rgba(245,158,11,0)}
}
.disclaimer {
    background:linear-gradient(to right,rgba(254,243,199,0.8),rgba(255,251,235,0.9));
    border-left:4px solid #F59E0B; border-radius:12px; padding:18px 22px;
    font-size:0.88rem; color:#92400E; line-height:1.6; margin-top:28px;
    display:flex; align-items:flex-start; gap:14px;
}
.disclaimer-icon {
    font-size:1.4rem; background:#FFFBEB; border-radius:50%; width:36px; height:36px;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; animation:pulseWarning 2s infinite;
}
.disclaimer-content b { color:#B45309; font-size:0.95rem; display:block; margin-bottom:4px; }
</style>

<div class="page-hero">
    <div class="page-title">📋 Clinical AI Segmentation Report</div>
    <div class="page-sub">Upload patient .npz slices to generate a full segmentation analysis report with subregion breakdown and clinical findings.</div>
</div>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODALITY_CHANNELS = {"T1c (Contrast)": 0, "T1n": 1, "T2-FLAIR": 2, "T2w": 3}
LABEL_NAMES  = {1: "Enhancing Tumor (ET)", 2: "Non-enhancing Core (NETC)",
                3: "Cystic Component (CC)", 4: "Peritumoral Edema (ED)"}
LABEL_COLORS = {1: "#EF4444", 2: "#10B981", 3: "#3B82F6", 4: "#EAB308"}

OVERLAY_CS = [
    [0.00,"rgba(0,0,0,0)"],[0.20,"rgba(0,0,0,0)"],
    [0.21,"rgba(239,68,68,0.65)"],[0.40,"rgba(239,68,68,0.65)"],
    [0.41,"rgba(16,185,129,0.60)"],[0.60,"rgba(16,185,129,0.60)"],
    [0.61,"rgba(59,130,246,0.60)"],[0.80,"rgba(59,130,246,0.60)"],
    [0.81,"rgba(234,179,8,0.60)"],[1.00,"rgba(234,179,8,0.60)"],
]


def slice_num(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def read_npz(f):
    buf = io.BytesIO(f.read())
    data = np.load(buf, allow_pickle=False)
    if "image" not in data.files or "mask" not in data.files:
        return None, None
    return data["image"], data["mask"]


def overlay_fig(image_ch, mask_2d, title: str, colorscale="Greys"):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=image_ch, colorscale=colorscale, showscale=False, opacity=1.0,
        hovertemplate="Intensity: %{z:.3f}<extra></extra>"
    ))
    if int((mask_2d > 0).sum()) > 0:
        fig.add_trace(go.Heatmap(
            z=mask_2d.astype(float), colorscale=OVERLAY_CS,
            zmin=0, zmax=4, showscale=False, opacity=1.0,
            hovertemplate="Label: %{z:.0f}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#0F172A"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4,r=4,t=34,b=4), height=320,
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,scaleanchor="x"),
    )
    return fig


# ─── Upload section ───────────────────────────────────────────────────────────
# ─── Read from Session State ──────────────────────────────────────────────────
st.markdown("""
<div class="upload-box">
    <div class="title">📂 Data Loaded from MRI Analysis Studio</div>
    <div class="sub">The report is generated automatically based on the files you uploaded in the MRI Analysis Studio page.</div>
</div>
""", unsafe_allow_html=True)

axial_files = st.session_state.get("axial_files", [])
coronal_files = st.session_state.get("coronal_files", [])

if not axial_files and not coronal_files:
    st.info("👆 Please go to the **MRI Analysis Studio** page and upload your `.npz` slices first.")
    st.stop()

# ─── Parse files by plane ─────────────────────────────────────────────────────
planes = {"axial": {}, "coronal": {}}

if axial_files:
    for f in axial_files:
        planes["axial"][slice_num(f.name)] = f

if coronal_files:
    for f in coronal_files:
        planes["coronal"][slice_num(f.name)] = f

# ─── Controls row ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.5, 1, 1.5])
with c1:
    available_planes = [p for p, d in planes.items() if d]
    plane_sel = st.radio("Plane", available_planes, horizontal=True)

with c2:
    modality = st.selectbox("Modality", list(MODALITY_CHANNELS.keys()))

with c3:
    slice_dict = planes[plane_sel]
    sorted_idx = sorted(slice_dict.keys())
    # Only show tumor-bearing slices (non-zero mask)
    tumor_indices = []
    for idx in sorted_idx:
        f = slice_dict[idx]
        f.seek(0)
        _, m = read_npz(f)
        if m is not None and int((m > 0).sum()) > 0:
            tumor_indices.append(idx)

    if tumor_indices:
        slice_idx = st.select_slider(
            f"Slice ({len(tumor_indices)} with tumour)",
            options=tumor_indices,
            value=tumor_indices[len(tumor_indices) // 2]
        )
    else:
        slice_idx = st.select_slider("Slice", options=sorted_idx,
                                     value=sorted_idx[len(sorted_idx)//2])

# ─── Load selected slice ──────────────────────────────────────────────────────
selected_file = slice_dict[slice_idx]
selected_file.seek(0)
image_data, mask_data = read_npz(selected_file)

if image_data is None:
    st.error("Could not read the selected file — ensure it contains `image` and `mask` arrays.")
    st.stop()

ch       = MODALITY_CHANNELS[modality]
img_ch   = image_data[ch]
mask_2d  = mask_data[0] if mask_data.ndim == 3 else mask_data
h, w     = img_ch.shape
n_tumor  = int((mask_2d > 0).sum())
n_total_px = mask_2d.size

# ─── Status bar ───────────────────────────────────────────────────────────────
n_ax = len(planes.get("axial",{}))
n_co = len(planes.get("coronal",{}))
st.markdown(f"""
<div class="status-bar">
    <div style="display:flex;align-items:center;"><span class="live-dot"></span>
        Report generated · {datetime.now().strftime("%Y-%m-%d  %H:%M")}
    </div>
    <div style="color:#64748B;">
        {n_ax} axial · {n_co} coronal · slice {slice_idx} · {h}×{w} · {modality}
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main layout ──────────────────────────────────────────────────────────────
col_main, col_side = st.columns([2, 1], gap="medium")

with col_main:
    # ── Segmentation visualisation ────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-title">🖼️ Segmentation Overlay</div>',
                unsafe_allow_html=True)

    img_l, img_r = st.columns(2)
    with img_l:
        st.plotly_chart(overlay_fig(img_ch, np.zeros_like(mask_2d), "MRI Image Only"),
                        use_container_width=True)
    with img_r:
        st.plotly_chart(overlay_fig(img_ch, mask_2d, "Mask Overlay"),
                        use_container_width=True)

    # Colour legend
    legend = "  ".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:{c};display:inline-block;"></span>'
        f'<span style="font-size:0.75rem;color:#475569;">{n}</span></span>'
        for n, c in [(n, LABEL_COLORS[lid]) for lid, n in LABEL_NAMES.items()]
    )
    st.markdown(f'<div style="margin:-4px 0 6px 2px;">{legend}</div>', unsafe_allow_html=True)
    st.caption(f"Showing {modality} channel · Ground-truth mask overlaid in colour.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Scan metadata ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📄 Scan & Slice Metadata</div>
        <table class="meta-table">
            <tr><td class="meta-key">Plane</td>
                <td class="meta-val" style="color:#0284C7;">{plane_sel.capitalize()}</td></tr>
            <tr><td class="meta-key">Slice index</td>
                <td class="meta-val" style="font-family:monospace;">{slice_idx}</td></tr>
            <tr><td class="meta-key">Slice resolution</td>
                <td class="meta-val" style="font-family:monospace;">{h} × {w} px</td></tr>
            <tr><td class="meta-key">MRI channels</td>
                <td class="meta-val" style="font-family:monospace;">{image_data.shape[0]} (t1c · t1n · t2f · t2w)</td></tr>
            <tr><td class="meta-key">Display modality</td>
                <td class="meta-val" style="font-family:monospace;">{modality} (channel {ch})</td></tr>
            <tr><td class="meta-key">Tumour pixels (GT)</td>
                <td class="meta-val" style="color:#0891B2;font-weight:700;">{n_tumor:,} px ({100*n_tumor/n_total_px:.2f}% of slice)</td></tr>
            <tr><td class="meta-key">Dataset cohort</td>
                <td class="meta-val">BraTS-PEDs (pediatric)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Subregion breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-title">🧠 Predicted Subregion Breakdown</div>',
                unsafe_allow_html=True)
    if n_tumor == 0:
        st.info("No tumour labels in this slice — all pixels are background.")
    else:
        for lid, lname in LABEL_NAMES.items():
            vox   = int((mask_2d == lid).sum())
            pct   = 100.0 * vox / n_tumor
            color = LABEL_COLORS[lid]
            st.markdown(f"""
            <div class="seg-row">
                <div class="seg-top">
                    <div class="seg-label">
                        <div class="seg-dot" style="background:{color};box-shadow:0 0 6px {color}66;"></div>
                        {lname}
                    </div>
                    <span class="seg-pct" style="color:{color};">{pct:.1f}% &nbsp;({vox:,} px)</span>
                </div>
                <div class="seg-bar-bg">
                    <div class="seg-bar" style="width:{min(pct,100)}%;
                         background:linear-gradient(90deg,{color}cc,{color});"></div>
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Clinical findings ─────────────────────────────────────────────────────
    if n_tumor > 0:
        dominant_lid = max(LABEL_NAMES.keys(), key=lambda lid: int((mask_2d == lid).sum()))
        dom_name  = LABEL_NAMES[dominant_lid]
        dom_color = LABEL_COLORS[dominant_lid]
        dom_pct   = 100.0 * int((mask_2d == dominant_lid).sum()) / n_tumor
        et_pct    = 100.0 * int((mask_2d == 1).sum()) / n_tumor
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🩺 Clinical Findings</div>
            <div style="font-size:0.88rem;line-height:1.85;color:#334155;">
                On <b>{plane_sel}</b> slice <b>{slice_idx}</b>, the ground-truth annotation
                marks <b>{n_tumor:,} px</b> of tumour — predominantly
                <b style="color:{dom_color};">{dom_name}</b> ({dom_pct:.1f}%).
                Enhancing Tumor (ET) accounts for <b>{et_pct:.1f}%</b> of the labelled region.
            </div>
            <div style="font-size:0.8rem;color:#64748B;margin-top:14px;line-height:1.7;">
                <b>Note:</b> These figures are derived from the ground-truth expert annotation mask.
                Model prediction requires a trained checkpoint in
                <code>03_augmentation_eval/checkpoints/overnight_run/best.pt</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_side:
    # ── Tumour coverage gauge ─────────────────────────────────────────────────
    tumor_pct = 100.0 * n_tumor / n_total_px
    st.markdown(f"""
    <div class="panel" style="text-align:center;">
        <div class="panel-title" style="justify-content:center;">📊 Tumour Coverage</div>
        <div class="conf-ring" style="background:conic-gradient(#EF4444 0% {tumor_pct:.1f}%,#E2E8F0 {tumor_pct:.1f}% 100%);
             box-shadow:0 0 24px rgba(239,68,68,0.2);">
            <div class="conf-inner" style="background:#FFFFFF;width:82px;height:82px;border-radius:50%;">
                <div class="conf-num">{tumor_pct:.1f}%</div>
                <div class="conf-label">of slice</div>
            </div>
        </div>
        <div style="font-size:0.75rem;color:#64748B;margin-top:4px;">
            {n_tumor:,} labelled px<br>out of {n_total_px:,} total
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-label counts ──────────────────────────────────────────────────────
    rows = "".join(
        f'<tr><td class="meta-key">{LABEL_NAMES[lid]}</td>'
        f'<td class="meta-val" style="color:{LABEL_COLORS[lid]};font-family:monospace;">'
        f'{int((mask_2d==lid).sum()):,} px</td></tr>'
        for lid in LABEL_NAMES
    )
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">🏷️ Label Counts</div>
        <table class="meta-table">{rows}</table>
    </div>""", unsafe_allow_html=True)

    # ── Slice summary ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📁 Patient Summary</div>
        <table class="meta-table">
            <tr><td class="meta-key">Axial slices</td>
                <td class="meta-val" style="font-family:monospace;">{n_ax}</td></tr>
            <tr><td class="meta-key">Coronal slices</td>
                <td class="meta-val" style="font-family:monospace;">{n_co}</td></tr>
            <tr><td class="meta-key">Tumour-bearing</td>
                <td class="meta-val" style="color:#059669;font-family:monospace;">{len(tumor_indices)} slices</td></tr>
            <tr><td class="meta-key">Report generated</td>
                <td class="meta-val" style="font-family:monospace;">{datetime.now().strftime("%H:%M:%S")}</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

    # ── CSV & PDF Export ──────────────────────────────────────────────────────
    csv = "label_id,label_name,pixels,pct_of_tumour\n" + "\n".join(
        f"{lid},{LABEL_NAMES[lid]},{int((mask_2d==lid).sum())},{100.0*int((mask_2d==lid).sum())/max(n_tumor,1):.4f}"
        for lid in LABEL_NAMES
    )
    st.download_button(
        "⬇  Download Slice Report (CSV)",
        data=csv,
        file_name=f"{plane_sel}_slice{slice_idx}_report.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    try:
        from fpdf import FPDF
        from PIL import Image
        import tempfile
        import os
        import numpy as np

        class ReportPDF(FPDF):
            def header(self):
                self.set_fill_color(2, 132, 199) # #0284C7
                self.rect(0, 0, 210, 25, 'F')
                self.set_font("Arial", 'B', 18)
                self.set_text_color(255, 255, 255)
                self.cell(0, 10, "Clinical AI Segmentation Report", border=0, ln=1, align="C")
                self.set_font("Arial", '', 10)
                self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}", border=0, ln=1, align="C")
                self.ln(8)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, "RESEARCH USE ONLY - NOT FOR CLINICAL DECISIONS", 0, 0, "C")

        pdf = ReportPDF()
        pdf.add_page()
        
        # Prepare the slice image for embedding
        img_norm = (img_ch - img_ch.min()) / (img_ch.max() - img_ch.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        
        # Add basic RGB overlay if tumor exists
        if n_tumor > 0:
            color_mask = np.zeros((*img_uint8.shape, 3), dtype=np.uint8)
            for i in range(3): color_mask[..., i] = img_uint8
            
            # Simple red overlay for all tumor pixels
            red_overlay = mask_2d > 0
            color_mask[red_overlay, 0] = 239 # R
            color_mask[red_overlay, 1] = 68  # G
            color_mask[red_overlay, 2] = 68  # B
            pil_img = Image.fromarray(color_mask)
        else:
            pil_img = Image.fromarray(img_uint8).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
            # Resize image slightly to improve PDF visual quality
            pil_img = pil_img.resize((h*2, w*2), Image.NEAREST)
            pil_img.save(img_tmp.name)
            img_path = img_tmp.name

        # Title / Patient Meta
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "Scan Metadata", ln=1)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(71, 85, 105)
        
        pdf.cell(60, 8, f"Plane: {plane_sel.capitalize()}", ln=0)
        pdf.cell(60, 8, f"Slice Index: {slice_idx}", ln=0)
        pdf.cell(60, 8, f"Modality: {modality}", ln=1)
        pdf.cell(60, 8, f"Resolution: {h} x {w} px", ln=1)
        
        pdf.ln(5)
        # Embed Image
        pdf.image(img_path, x=65, w=80)
        os.unlink(img_path)
        pdf.ln(8)

        # Tumor Stats
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "Tumour Analysis", ln=1)
        
        if n_tumor > 0:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(220, 38, 38) # red
            pdf.cell(0, 8, f"TUMOUR DETECTED: {n_tumor:,} px ({100*n_tumor/n_total_px:.2f}% of slice)", ln=1)
            pdf.ln(3)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(71, 85, 105)
            
            # Draw table
            pdf.set_fill_color(241, 245, 249) # header bg
            pdf.cell(100, 8, "Subregion", border=1, fill=True)
            pdf.cell(40, 8, "Pixels", border=1, fill=True)
            pdf.cell(40, 8, "% of Tumour", border=1, fill=True, ln=1)
            
            for lid in LABEL_NAMES:
                vox = int((mask_2d==lid).sum())
                pct = 100.0 * vox / n_tumor
                pdf.cell(100, 8, LABEL_NAMES[lid], border=1)
                pdf.cell(40, 8, f"{vox:,}", border=1)
                pdf.cell(40, 8, f"{pct:.1f}%", border=1, ln=1)
        else:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(16, 185, 129) # green
            pdf.cell(0, 10, "NO TUMOUR DETECTED (Healthy Tissue)", ln=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_name = tmp.name
        pdf.output(tmp_name)
        with open(tmp_name, "rb") as f:
            pdf_bytes = f.read()
        os.unlink(tmp_name)
        
        st.download_button(
            "📄 Download Slice Report (PDF)",
            data=pdf_bytes,
            file_name=f"{plane_sel}_slice{slice_idx}_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except ImportError:
        st.info("💡 To enable PDF export, please run `pip install fpdf2` and `pip install Pillow` in your terminal.")

    st.markdown("""
    <div style="font-size:0.68rem;color:rgba(100,116,139,0.5);text-align:center;
                margin-top:10px;font-family:'JetBrains Mono',monospace;">
        RESEARCH USE ONLY · NOT FOR CLINICAL DECISIONS
    </div>""", unsafe_allow_html=True)

# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-content">
        <b>Medical Disclaimer</b>
        NeuroPeds AI is a research-grade decision support platform designed to assist clinicians and researchers.
        It does not replace independent clinical diagnosis or professional medical judgment. All outputs are for research purposes only.
    </div>
</div>
""", unsafe_allow_html=True)
