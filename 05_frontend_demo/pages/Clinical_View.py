import streamlit as st

st.set_page_config(page_title="Tumor Subregion Guide | NeuroPeds AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
* { font-family: 'Inter', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif; }

.page-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
    border: 1px solid rgba(14, 165, 233, 0.15);
    border-radius: 20px; padding: 30px 40px; margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.02);
}
.page-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; font-family: 'Outfit', sans-serif;
    color: #0F172A; margin-bottom: 8px;
}
.page-sub { color: #475569; font-size: 0.95rem; line-height: 1.6;}

.panel { 
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; 
    padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); 
}
.panel-title { 
    font-size: 1.1rem; font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif;
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px; 
}

.sub-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.sub-card {
    background: #F8FAFC; border-radius: 14px; padding: 20px; border: 1px solid transparent;
    transition: all .2s ease;
}
.sub-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.04); }
.sub-card.et  { border-left: 4px solid #EF4444; }
.sub-card.ed  { border-left: 4px solid #F59E0B; }
.sub-card.cc  { border-left: 4px solid #3B82F6; }
.sub-card.net { border-left: 4px solid #10B981; }

.sub-num  { font-size: 0.7rem; font-family: monospace; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
.et .sub-num  { color: #EF4444; } .ed .sub-num  { color: #F59E0B; }
.cc .sub-num  { color: #3B82F6; } .net .sub-num { color: #10B981; }

.sub-name { font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 8px; }
.sub-desc { font-size: 0.85rem; color: #475569; line-height: 1.6; }
.sub-badge { 
    display: inline-block; margin-top: 12px; padding: 4px 10px; border-radius: 6px; 
    font-size: 0.75rem; font-family: monospace; font-weight: 600; 
}
.et  .sub-badge { background: #FEF2F2; color: #DC2626; }
.ed  .sub-badge { background: #FFFBEB; color: #D97706; }
.cc  .sub-badge { background: #EFF6FF; color: #2563EB; }
.net .sub-badge { background: #ECFDF5; color: #059669; }

.xai-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 20px;
    transition: all .2s ease;
}
.xai-icon { font-size: 1.8rem; margin-bottom: 10px; }
.xai-name { font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 6px; }
.xai-desc { font-size: 0.85rem; color: #475569; line-height: 1.6; }

.modality-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.modality-table th { 
    text-align: left; padding: 12px; font-size: 0.75rem; text-transform: uppercase;
    color: #64748B; border-bottom: 1px solid #E2E8F0; background: #F8FAFC;
}
.modality-table td { padding: 14px 12px; border-bottom: 1px solid #E2E8F0; color: #334155; }
.modality-table tr:last-child td { border-bottom: none; }

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
    <div class="page-title">🩺 Tumor Subregion Guide</div>
    <div class="page-sub">Reference documentation for the BraTS-PEDs tumor subregions and the multi-modal MRI sequences processed by the AI engine.</div>
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
    <div class="panel-title">🔍 Explainability Module</div>
    <div class="xai-card">
        <div class="xai-icon">ℹ️</div>
        <div class="xai-name">Not Implemented in Current Build</div>
        <div class="xai-desc">Grad-CAM, attention rollout, and uncertainty maps are currently out-of-scope for this phase of the app. The deployed model is an ensemble without attention layers. Explaining its predictions visually is planned for future work. Currently, clinical interpretation relies on the predicted segmentation overlays and per-region performance metrics (Dice/HD95).</div>
    </div>
</div>

<div class="panel">
    <div class="panel-title">📡 MRI Modality Reference</div>
    <table class="modality-table">
        <thead>
            <tr><th>Modality</th><th>Key Visualization</th><th>Primary Subregion Target</th><th>Signal Profile</th></tr>
        </thead>
        <tbody>
            <tr><td><b style="color:#0EA5E9;">T1</b></td><td>Anatomical baseline, CSF dark</td><td>Non-Enhancing Tumor (NET)</td><td>Gray matter / White matter contrast</td></tr>
            <tr><td><b style="color:#4F46E5;">T1c</b></td><td>Contrast-enhancing lesion</td><td>Enhancing Tumor (ET)</td><td>Bright on active regions</td></tr>
            <tr><td><b style="color:#10B981;">T2</b></td><td>Fluid & edema bright</td><td>Cystic Component (CC)</td><td>Hyperintense fluid</td></tr>
            <tr><td><b style="color:#F59E0B;">FLAIR</b></td><td>CSF suppressed, edema bright</td><td>Peritumoral Edema (ED)</td><td>Perilesional high signal</td></tr>
        </tbody>
    </table>
</div>

<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-content">
        <b>Research Use Only</b>
        NeuroPeds AI is a research-grade decision support platform designed to assist clinicians and researchers. It does not replace independent clinical diagnosis or professional medical judgment. All outputs are strictly for research purposes.
    </div>
</div>
""", unsafe_allow_html=True)