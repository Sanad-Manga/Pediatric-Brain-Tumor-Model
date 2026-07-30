import streamlit as st
import pandas as pd
import time
from datetime import datetime

from components.theme import apply_custom_theme
from utils.loaders import cache_available, list_demo_subjects, load_metrics_cache

apply_custom_theme()

# ──────────────────────────────────────────────────────────
#  PAGE-LEVEL CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
    border: 1px solid rgba(2, 132, 199, 0.15);
    border-radius: 20px;
    padding: 48px 44px;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}
.hero-label {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px;
    background: rgba(2, 132, 199, 0.08);
    border: 1px solid rgba(2, 132, 199, 0.2);
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #0284C7;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #059669;
    animation: pulseDot 2s infinite;
}
.hero-title {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: #0F172A;
    margin-bottom: 16px;
    font-family: 'Outfit', sans-serif;
}
.hero-sub {
    color: #475569;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 650px;
    line-height: 1.6;
    margin-bottom: 30px;
}
.hero-tags { display: flex; flex-wrap: wrap; gap: 10px; }
.hero-tag {
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    color: #334155;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

/* ── STAT CARDS ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 40px; }
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(2, 132, 199, 0.08);
    border-color: rgba(2, 132, 199, 0.3);
}
.stat-icon { font-size: 1.8rem; margin-bottom: 12px; }
.stat-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 8px; }
.stat-value { font-size: 2rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em; color: #0F172A; margin-bottom: 8px; }
.stat-delta { font-size: 0.8rem; font-weight: 500; color: #059669; }

/* ── SECTION HEADINGS ── */
.section-heading {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 24px;
}
.section-heading-icon {
    width: 38px; height: 38px;
    background: rgba(2, 132, 199, 0.08);
    border: 1px solid rgba(2, 132, 199, 0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.section-heading-text { font-size: 1.2rem; font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif;}
.section-heading-sub  { font-size: 0.85rem; color: #64748B; margin-top: 2px; }

/* ── MODULE CARDS ── */
.mod-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 40px; }
.mod-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 26px;
    transition: all 0.2s ease;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.mod-card:hover { transform: translateY(-3px); border-color: #0284C7; box-shadow:0 10px 25px rgba(2, 132, 199, 0.1); }
.mod-card-top { display:flex; align-items:center; gap:14px; margin-bottom:16px; }
.mod-card-icon {
    width:46px; height:46px; border-radius:12px;
    display:flex; align-items:center; justify-content:center; font-size:1.4rem;
    flex-shrink:0;
    background: rgba(2, 132, 199, 0.08); border: 1px solid rgba(2, 132, 199, 0.2);
}
.mod-card-name { font-size:1.05rem; font-weight:700; color:#0F172A; font-family: 'Outfit', sans-serif;}
.mod-card-nav  { font-size:0.75rem; color:#64748B; margin-top:2px; font-weight: 500;}
.mod-card-desc { font-size:0.9rem; color:#475569; line-height:1.6; margin-bottom:16px; }

/* ── DATA TABLE ── */
.data-section { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:40px; }
.data-panel {
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:16px; padding:24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}

/* ── FOOTER ── */
.dash-footer {
    text-align:center;
    padding:24px;
    font-size:0.8rem;
    color:#94A3B8;
    border-top:1px solid #E2E8F0;
    letter-spacing:0.02em;
    font-weight: 500;
}

@media (max-width: 900px) {
    .stat-grid, .mod-grid { grid-template-columns:1fr; }
    .data-section { grid-template-columns:1fr; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  HERO
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">
        <span class="hero-live-dot"></span>
        Clinical Dashboard
    </div>
    <div class="hero-title">Pediatric Brain Tumor<br>Clinical Decision Support</div>
    <div class="hero-sub">
        AI-assisted segmentation and explainable reports for pediatric oncology, designed for streamlined radiologist workflows. Analyze multi-modal MRI scans with high precision.
    </div>
    <div class="hero-tags">
        <span class="hero-tag">⚕️ Clinical Review</span>
        <span class="hero-tag">🔒 HIPAA Compliant</span>
        <span class="hero-tag">🧠 Multi-modal MRI</span>
        <span class="hero-tag">📁 DICOM & NIfTI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  STAT CARDS
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">✅</div>
        <div class="stat-label">System Status</div>
        <div class="stat-value">Online</div>
        <div class="stat-delta">All clinical systems ready</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🗂</div>
        <div class="stat-label">Analyzed Scans</div>
        <div class="stat-value">257</div>
        <div class="stat-delta">T1 · T1c · T2 · FLAIR</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⏱</div>
        <div class="stat-label">Avg Processing Time</div>
        <div class="stat-value">~45s</div>
        <div class="stat-delta" style="color: #0284C7">Per full 2D volume</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-label">Baseline Accuracy</div>
        <div class="stat-value">70%</div>
        <div class="stat-delta" style="color: #0284C7">Dice Score (Whole Tumor)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  PLATFORM MODULES
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading">
    <div class="section-heading-icon">🧩</div>
    <div>
        <div class="section-heading-text">Clinical Modules</div>
        <div class="section-heading-sub">Tools for analysis, reporting, and explainability</div>
    </div>
</div>
""", unsafe_allow_html=True)

def mod_card(icon, name, nav, desc):
    return f"""
    <div class="mod-card" style="height:100%;">
        <div class="mod-card-top">
            <div class="mod-card-icon">{icon}</div>
            <div>
                <div class="mod-card-name">{name}</div>
                <div class="mod-card-nav">{nav}</div>
            </div>
        </div>
        <div class="mod-card-desc">{desc}</div>
    </div>"""

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown(mod_card("🧠", "MRI Analysis Studio", "Clinical & Analysis →", 
        "Upload multi-modal MRI volumes. Inspect slices in an interactive 2D viewer with segmentation overlays rendered per-class."), unsafe_allow_html=True)
with col2:
    st.markdown(mod_card("📄", "Segmentation Report", "Clinical & Analysis →", 
        "AI-generated patient summary with tumor statistics, region-level interpretation, and one-click PDF export for clinical records."), unsafe_allow_html=True)
with col3:
    st.markdown(mod_card("🩺", "Clinical Explainability", "Clinical & Analysis →", 
        "Grad-CAM and attention maps reveal exactly which voxels drove each prediction — grounding AI decisions in anatomy radiologists recognize."), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  DATASET + RECENT CASES  (2-column)
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading">
    <div class="section-heading-icon">📚</div>
    <div>
        <div class="section-heading-text">Dataset & Recent Cases</div>
        <div class="section-heading-sub">Recent pediatric patient cases and reference dataset details</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown("""
    <div class="data-panel">
        <div style="font-size:0.8rem; font-weight: 600; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-bottom:18px;">Reference Dataset Summary</div>
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
            <tr style="border-bottom:1px solid #E2E8F0;">
                <td style="padding:12px 0; color:#475569; width:45%;">Dataset</td>
                <td style="padding:12px 0; color:#0F172A; font-weight:600;">BraTS-PEDs 2024</td>
            </tr>
            <tr style="border-bottom:1px solid #E2E8F0;">
                <td style="padding:12px 0; color:#475569;">Subjects Analyzed</td>
                <td style="padding:12px 0; color:#0284C7; font-weight:600;">257</td>
            </tr>
            <tr style="border-bottom:1px solid #E2E8F0;">
                <td style="padding:12px 0; color:#475569;">Supported Modalities</td>
                <td style="padding:12px 0; color:#0F172A;">T1, T1c, T2, FLAIR</td>
            </tr>
            <tr>
                <td style="padding:12px 0; color:#475569;">Primary Task</td>
                <td style="padding:12px 0; color:#0F172A;">Pediatric Tumor Segmentation</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    if not cache_available():
        st.write("") # Removed cache not found info
    else:
        _subjects = list_demo_subjects(limit=5)
        cases_df = pd.DataFrame({
            "Patient ID": _subjects,
            "Modalities": ["T1c, T1, T2, FLAIR"] * len(_subjects),
            "Status": ["Ready for review"] * len(_subjects),
        })
        st.dataframe(cases_df, use_container_width=True, hide_index=True)
        st.caption("Open **Segmentation Report** to run analysis on any of these cases and export clinical reports.")

# ──────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-footer">
    NEUROPEDS AI &nbsp;·&nbsp; PEDIATRIC ONCOLOGY DECISION SUPPORT &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)