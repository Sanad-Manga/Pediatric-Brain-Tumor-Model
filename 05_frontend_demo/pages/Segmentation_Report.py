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

st.set_page_config(page_title="Segmentation Report | NeuroPeds AI", page_icon="📋", layout="wide")

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
.page-sub { font-size: 0.95rem; color: #475569; line-height: 1.6;}

.panel { 
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; 
    padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); 
}
.panel-title { 
    font-size: 1.1rem; font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif;
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px; 
    border-bottom: 1px solid #E2E8F0; padding-bottom: 12px;
}

.meta-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.meta-table td { padding: 10px 8px; border-bottom: 1px solid #F1F5F9; }
.meta-table tr:last-child td { border-bottom: none; }
.meta-key { width: 45%; font-weight: 500; color: #64748B; }
.meta-val { font-weight: 600; color: #0F172A; font-family: monospace; font-size:0.9rem;}

.seg-row { margin-bottom: 12px; }
.seg-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.seg-label { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #475569; font-weight: 500;}
.seg-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.seg-pct  { font-size: 0.85rem; font-family: monospace; font-weight: 600; }
.seg-bar-bg { height: 8px; border-radius: 4px; background: #F1F5F9; overflow: hidden; }
.seg-bar    { height: 100%; border-radius: 4px; }

.status-bar {
    display: flex; justify-content: space-between; align-items: center;
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 12px 18px; margin-bottom: 24px;
    font-size: 0.85rem; font-weight: 500;
}
.live-dot { 
    width: 8px; height: 8px; border-radius: 50%; background: #10B981;
    display: inline-block; animation: pulse-green 2s infinite; margin-right: 8px; 
}
@keyframes pulse-green {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.conf-ring {
    width: 120px; height: 120px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px auto;
}
.conf-inner { 
    display: flex; flex-direction: column; align-items: center; justify-content: center; 
    background: #FFFFFF; width: 90px; height: 90px; border-radius: 50%;
}
.conf-num   { font-size: 1.5rem; font-weight: 800; color: #0F172A; line-height: 1; font-family: 'Outfit', sans-serif;}
.conf-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-top: 4px; font-weight:600;}

.upload-box {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 24px;
}
.upload-box .title { font-size: 1.1rem; font-weight: 700; font-family: 'Outfit', sans-serif; color: #0F172A; margin-bottom: 8px; }
.upload-box .sub   { font-size: 0.9rem; color: #475569; }

.disclaimer {
    background: #FEF2F2;
    border-left: 4px solid #EF4444; border-radius: 12px; padding: 16px 20px;
    font-size: 0.9rem; color: #991B1B; line-height: 1.5; margin-top: 24px;
    display: flex; align-items: flex-start; gap: 12px;
}
.disclaimer-icon { font-size: 1.4rem; }
.disclaimer-content b { color: #7F1D1D; font-size: 0.95rem; display: block; margin-bottom: 4px; }
</style>

<div class="page-hero">
    <div class="page-title">📋 Clinical Segmentation Report</div>
    <div class="page-sub">Generate a comprehensive clinical analysis report, including ground-truth visualization and subregion statistics, directly from loaded MRI slices.</div>
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
        title=dict(text=title, font=dict(size=14, color="#0F172A", family="Outfit"), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4,r=4,t=34,b=4), height=340,
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False,scaleanchor="x"),
    )
    return fig


# ─── Upload section ───────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-box">
    <div class="title">📂 Data Loaded from MRI Analysis Studio</div>
    <div class="sub">This report relies on the `.npz` slices uploaded during your active session.</div>
</div>
""", unsafe_allow_html=True)

axial_files = st.session_state.get("axial_files", [])
coronal_files = st.session_state.get("coronal_files", [])

if not axial_files and not coronal_files:
    st.info("👆 Please navigate to the **MRI Analysis Studio** and upload patient slices to generate a report.")
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
    plane_sel = st.radio("Select Plane", available_planes, horizontal=True)

with c2:
    modality = st.selectbox("Display Modality", list(MODALITY_CHANNELS.keys()))

with c3:
    slice_dict = planes[plane_sel]
    sorted_idx = sorted(slice_dict.keys())
    
    tumor_indices = []
    for idx in sorted_idx:
        f = slice_dict[idx]
        f.seek(0)
        _, m = read_npz(f)
        if m is not None and int((m > 0).sum()) > 0:
            tumor_indices.append(idx)

    if tumor_indices:
        slice_idx = st.select_slider(
            f"Navigate Slice ({len(tumor_indices)} with tumour)",
            options=tumor_indices,
            value=tumor_indices[len(tumor_indices) // 2]
        )
    else:
        slice_idx = st.select_slider("Navigate Slice", options=sorted_idx,
                                     value=sorted_idx[len(sorted_idx)//2])

# ─── Load selected slice ──────────────────────────────────────────────────────
selected_file = slice_dict[slice_idx]
selected_file.seek(0)
image_data, mask_data = read_npz(selected_file)

if image_data is None:
    st.error("Invalid file format. Ensure the selected file contains `image` and `mask` arrays.")
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
        <span style="color:#0F172A;">Report generated · {datetime.now().strftime("%Y-%m-%d  %H:%M")}</span>
    </div>
    <div style="color:#64748B; font-family:monospace; font-size:0.75rem;">
        {n_ax} axial · {n_co} coronal · slice {slice_idx} · {h}×{w} · {modality}
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Main layout ──────────────────────────────────────────────────────────────
col_main, col_side = st.columns([2.2, 1], gap="large")

with col_main:
    # ── Segmentation visualisation ────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-title">🖼️ Segmentation Overlay</div>',
                unsafe_allow_html=True)

    img_l, img_r = st.columns(2)
    with img_l:
        st.plotly_chart(overlay_fig(img_ch, np.zeros_like(mask_2d), "MRI Source Image"),
                        use_container_width=True, config={'displayModeBar': False})
    with img_r:
        st.plotly_chart(overlay_fig(img_ch, mask_2d, "Ground Truth Mask"),
                        use_container_width=True, config={'displayModeBar': False})

    # Colour legend
    legend = "  ".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};display:inline-block;"></span>'
        f'<span style="font-size:0.8rem;color:#475569;font-weight:500;">{n}</span></span>'
        for n, c in [(n, LABEL_COLORS[lid]) for lid, n in LABEL_NAMES.items()]
    )
    st.markdown(f'<div style="text-align:center; padding-top:10px;">{legend}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Scan metadata ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📄 Core Metadata</div>
        <table class="meta-table">
            <tr><td class="meta-key">Plane</td>
                <td class="meta-val" style="color:#0EA5E9;">{plane_sel.capitalize()}</td></tr>
            <tr><td class="meta-key">Slice Index</td>
                <td class="meta-val">{slice_idx}</td></tr>
            <tr><td class="meta-key">Resolution</td>
                <td class="meta-val">{h} × {w} px</td></tr>
            <tr><td class="meta-key">MRI Channels</td>
                <td class="meta-val">{image_data.shape[0]} (t1c · t1n · t2f · t2w)</td></tr>
            <tr><td class="meta-key">Tumour Area (GT)</td>
                <td class="meta-val" style="color:#059669;">{n_tumor:,} px ({100*n_tumor/n_total_px:.2f}%)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Clinical findings ─────────────────────────────────────────────────────
    if n_tumor > 0:
        dominant_lid = max(LABEL_NAMES.keys(), key=lambda lid: int((mask_2d == lid).sum()))
        dom_name  = LABEL_NAMES[dominant_lid]
        dom_color = LABEL_COLORS[dominant_lid]
        dom_pct   = 100.0 * int((mask_2d == dominant_lid).sum()) / n_tumor
        et_pct    = 100.0 * int((mask_2d == 1).sum()) / n_tumor
        st.markdown(f"""
        <div class="panel" style="background:#F8FAFC; border-color:#CBD5E1;">
            <div class="panel-title">🩺 Clinical Observations</div>
            <div style="font-size:0.9rem;line-height:1.7;color:#334155;">
                In the analyzed <b>{plane_sel}</b> slice <b>{slice_idx}</b>, the annotated region 
                comprises <b>{n_tumor:,} pixels</b> of pathological tissue. The predominant subregion is 
                <b style="color:{dom_color};">{dom_name}</b>, representing ({dom_pct:.1f}%) of the total tumor mass. 
                Active Enhancing Tumor (ET) accounts for <b>{et_pct:.1f}%</b>.
            </div>
            <div style="font-size:0.8rem;color:#64748B;margin-top:14px; padding-top:10px; border-top:1px solid #E2E8F0;">
                <b>Note:</b> Metrics reflect the provided ground-truth mask. For model inference, 
                ensure a valid checkpoint exists in the `checkpoints` directory.
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_side:
    # ── Tumour coverage gauge ─────────────────────────────────────────────────
    tumor_pct = 100.0 * n_tumor / n_total_px
    ring_color = "#EF4444" if tumor_pct > 10 else ("#F59E0B" if tumor_pct > 2 else "#10B981")
    
    st.markdown(f"""
    <div class="panel" style="text-align:center;">
        <div class="panel-title" style="justify-content:center; border:none; padding-bottom:0;">📊 Tumour Coverage</div>
        <div class="conf-ring" style="background:conic-gradient({ring_color} 0% {tumor_pct:.1f}%,#F1F5F9 {tumor_pct:.1f}% 100%);">
            <div class="conf-inner">
                <div class="conf-num">{tumor_pct:.1f}%</div>
                <div class="conf-label">of slice area</div>
            </div>
        </div>
        <div style="font-size:0.8rem;color:#64748B; font-weight:500;">
            {n_tumor:,} labelled px<br>out of {n_total_px:,} total
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Subregion breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-title">🧠 Composition</div>', unsafe_allow_html=True)
    if n_tumor == 0:
        st.info("Healthy tissue. No tumor labels present.")
    else:
        for lid, lname in LABEL_NAMES.items():
            vox   = int((mask_2d == lid).sum())
            pct   = 100.0 * vox / n_tumor
            color = LABEL_COLORS[lid]
            st.markdown(f"""
            <div class="seg-row">
                <div class="seg-top">
                    <div class="seg-label">
                        <div class="seg-dot" style="background:{color};box-shadow:0 0 4px {color}88;"></div>
                        {lname.split('(')[1].replace(')','')}
                    </div>
                    <span class="seg-pct" style="color:{color};">{pct:.1f}%</span>
                </div>
                <div class="seg-bar-bg">
                    <div class="seg-bar" style="width:{min(pct,100)}%;background:{color};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Slice summary ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📁 Session Summary</div>
        <table class="meta-table">
            <tr><td class="meta-key">Loaded Axial</td>
                <td class="meta-val">{n_ax}</td></tr>
            <tr><td class="meta-key">Loaded Coronal</td>
                <td class="meta-val">{n_co}</td></tr>
            <tr><td class="meta-key">Pathological Slices</td>
                <td class="meta-val" style="color:#0EA5E9;">{len(tumor_indices)}</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

    # ── CSV & PDF Export ──────────────────────────────────────────────────────
    csv = "label_id,label_name,pixels,pct_of_tumour\n" + "\n".join(
        f"{lid},{LABEL_NAMES[lid]},{int((mask_2d==lid).sum())},{100.0*int((mask_2d==lid).sum())/max(n_tumor,1):.4f}"
        for lid in LABEL_NAMES
    )
    st.download_button(
        "⬇  Export to CSV",
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

        class ReportPDF(FPDF):
            def header(self):
                self.set_fill_color(14, 165, 233) 
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
        
        img_norm = (img_ch - img_ch.min()) / (img_ch.max() - img_ch.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        
        if n_tumor > 0:
            color_mask = np.zeros((*img_uint8.shape, 3), dtype=np.uint8)
            for i in range(3): color_mask[..., i] = img_uint8
            red_overlay = mask_2d > 0
            color_mask[red_overlay, 0] = 239 
            color_mask[red_overlay, 1] = 68  
            color_mask[red_overlay, 2] = 68  
            pil_img = Image.fromarray(color_mask)
        else:
            pil_img = Image.fromarray(img_uint8).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
            pil_img = pil_img.resize((h*2, w*2), Image.NEAREST)
            pil_img.save(img_tmp.name)
            img_path = img_tmp.name

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
        pdf.image(img_path, x=65, w=80)
        os.unlink(img_path)
        pdf.ln(8)

        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "Tumour Analysis", ln=1)
        
        if n_tumor > 0:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(220, 38, 38)
            pdf.cell(0, 8, f"TUMOUR DETECTED: {n_tumor:,} px ({100*n_tumor/n_total_px:.2f}% of slice)", ln=1)
            pdf.ln(3)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(71, 85, 105)
            
            pdf.set_fill_color(241, 245, 249)
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
            pdf.set_text_color(16, 185, 129)
            pdf.cell(0, 10, "NO TUMOUR DETECTED (Healthy Tissue)", ln=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_name = tmp.name
        pdf.output(tmp_name)
        with open(tmp_name, "rb") as f:
            pdf_bytes = f.read()
        os.unlink(tmp_name)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "📄 Export to PDF",
            data=pdf_bytes,
            file_name=f"{plane_sel}_slice{slice_idx}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    except ImportError:
        st.info("💡 To enable PDF export, please run `pip install fpdf2 Pillow`.")

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