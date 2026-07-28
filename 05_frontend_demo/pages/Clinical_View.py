import streamlit as st

st.set_page_config(page_title="Clinical Explainability | NeuroFed AI", page_icon="🩺", layout="wide")

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
.panel { background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:28px; margin-bottom:20px; }
.panel-title { font-size:1rem; font-weight:700; color:#F3F4F6; margin-bottom:18px; }
.sub-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.sub-card {
    background:rgba(10,15,35,0.65); border-radius:14px; padding:20px;
    transition: transform .2s ease, box-shadow .2s ease;
}
.sub-card:hover { transform:translateY(-2px); }
.sub-card.et  { border-left:4px solid #EF4444; }
.sub-card.ed  { border-left:4px solid #EAB308; }
.sub-card.cc  { border-left:4px solid #3B82F6; }
.sub-card.net { border-left:4px solid #10B981; }
.sub-card:hover.et  { box-shadow:0 8px 24px rgba(239,68,68,0.12); }
.sub-card:hover.ed  { box-shadow:0 8px 24px rgba(234,179,8,0.12); }
.sub-card:hover.cc  { box-shadow:0 8px 24px rgba(59,130,246,0.12); }
.sub-card:hover.net { box-shadow:0 8px 24px rgba(16,185,129,0.12); }
.sub-num  { font-size:0.68rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:6px; }
.et .sub-num  { color:#EF4444; } .ed .sub-num  { color:#EAB308; }
.cc .sub-num  { color:#3B82F6; } .net .sub-num { color:#10B981; }
.sub-name { font-size:0.9rem; font-weight:700; color:#F3F4F6; margin-bottom:8px; }
.sub-desc { font-size:0.8rem; color:#94A3B8; line-height:1.6; }
.sub-badge { display:inline-block; margin-top:10px; padding:3px 10px; border-radius:6px; font-size:0.7rem; font-family:monospace; font-weight:600; }
.et  .sub-badge { background:rgba(239,68,68,0.12); color:#EF4444; }
.ed  .sub-badge { background:rgba(234,179,8,0.12); color:#EAB308; }
.cc  .sub-badge { background:rgba(59,130,246,0.12); color:#3B82F6; }
.net .sub-badge { background:rgba(16,185,129,0.12); color:#10B981; }
.xai-row { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.xai-card {
    background:rgba(10,15,35,0.65); border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:20px; text-align:center;
    transition: border-color .2s ease, transform .2s ease;
}
.xai-card:hover { border-color:rgba(56,189,248,0.3); transform:translateY(-2px); }
.xai-icon { font-size:1.8rem; margin-bottom:10px; }
.xai-name { font-size:0.85rem; font-weight:700; color:#F3F4F6; margin-bottom:6px; }
.xai-desc { font-size:0.75rem; color:#64748B; line-height:1.5; }
.modality-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.modality-table th { text-align:left; padding:10px 12px; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:0.06em; color:#64748B; border-bottom:1px solid rgba(255,255,255,0.07); }
.modality-table td { padding:12px; border-bottom:1px solid rgba(255,255,255,0.05); color:#CBD5E1; }
.modality-table tr:last-child td { border-bottom:none; }
.disclaimer { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
    border-radius:12px; padding:16px 20px; font-size:0.83rem; color:#FCA5A5; line-height:1.6; }
</style>

<div class="page-hero">
    <div class="page-title">Clinical Explainability</div>
    <div class="page-sub">Translating deep learning feature maps into actionable pediatric neuro-oncology insights.</div>
</div>

<div class="panel">
    <div class="panel-title">🏷️ BraTS-PEDs Tumor Subregions</div>
    <div class="sub-grid">
        <div class="sub-card et">
            <div class="sub-num">Region 01</div>
            <div class="sub-name">Enhancing Tumor (ET)</div>
            <div class="sub-desc">Active tumor regions with blood-brain barrier breakdown, highlighted clearly on T1-contrast sequences. Typically indicates aggressive proliferating tissue.</div>
            <span class="sub-badge">T1c · Contrast</span>
        </div>
        <div class="sub-card ed">
            <div class="sub-num">Region 02</div>
            <div class="sub-name">Peritumoral Edema (ED)</div>
            <div class="sub-desc">Surrounding tissue swelling and vasogenic edema identified by high signal intensity on FLAIR scans. Extends beyond the tumor margin.</div>
            <span class="sub-badge">FLAIR · High Signal</span>
        </div>
        <div class="sub-card cc">
            <div class="sub-num">Region 03</div>
            <div class="sub-name">Cystic Component (CC)</div>
            <div class="sub-desc">Fluid-filled necrotic or cystic regions within the tumor core structure. Appears as hypointense on T1 and hyperintense on T2 sequences.</div>
            <span class="sub-badge">T2 · Hypointense</span>
        </div>
        <div class="sub-card net">
            <div class="sub-num">Region 04</div>
            <div class="sub-name">Non-enhancing Tumor (NET)</div>
            <div class="sub-desc">Solid tumor core regions without active contrast enhancement. Represents infiltrative tumor tissue not yet compromising the blood-brain barrier.</div>
            <span class="sub-badge">T1 · Non-contrast</span>
        </div>
    </div>
</div>

<div class="panel">
    <div class="panel-title">🔍 Explainability Methods</div>
    <div class="xai-row">
        <div class="xai-card">
            <div class="xai-icon">🌡️</div>
            <div class="xai-name">Grad-CAM</div>
            <div class="xai-desc">Gradient-weighted class activation maps highlight which 3D voxel regions most influenced the segmentation decision for each subregion class.</div>
        </div>
        <div class="xai-card">
            <div class="xai-icon">👁️</div>
            <div class="xai-name">Attention Rollout</div>
            <div class="xai-desc">Traces attention flow through transformer encoder layers, producing spatial importance maps aligned to anatomical structures.</div>
        </div>
        <div class="xai-card">
            <div class="xai-icon">📊</div>
            <div class="xai-name">Uncertainty Maps</div>
            <div class="xai-desc">Monte Carlo dropout inference generates epistemic uncertainty estimates — flagging low-confidence boundary predictions for clinical review.</div>
        </div>
    </div>
</div>

<div class="panel">
    <div class="panel-title">📡 MRI Modality Reference</div>
    <table class="modality-table">
        <thead>
            <tr><th>Modality</th><th>Key Visualization</th><th>Primary Subregion</th><th>Signal</th></tr>
        </thead>
        <tbody>
            <tr><td><b style="color:#38BDF8;">T1</b></td><td>Anatomical baseline, CSF dark</td><td>Non-Enhancing Tumor (NET)</td><td>Gray matter / White matter contrast</td></tr>
            <tr><td><b style="color:#818CF8;">T1c</b></td><td>Contrast-enhancing lesion</td><td>Enhancing Tumor (ET)</td><td>Bright on active regions</td></tr>
            <tr><td><b style="color:#34D399;">T2</b></td><td>Fluid & edema bright</td><td>Cystic Component (CC)</td><td>Hyperintense fluid</td></tr>
            <tr><td><b style="color:#FBBF24;">FLAIR</b></td><td>CSF suppressed, edema bright</td><td>Peritumoral Edema (ED)</td><td>Perilesional high signal</td></tr>
        </tbody>
    </table>
</div>

<div class="disclaimer">
    <b>⚠️ Medical Disclaimer:</b> NeuroFed AI is a research-grade decision support platform designed to assist clinicians and researchers.
    It does not replace independent clinical diagnosis or professional medical judgment.
</div>
""", unsafe_allow_html=True)