import streamlit as st

st.set_page_config(page_title="About NeuroPeds AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 10% 50%, rgba(2, 132, 199, 0.08), transparent 50%),
                linear-gradient(135deg, #FFFFFF, #F0F9FF);
    border: 1px solid rgba(2, 132, 199, 0.15);
    border-radius: 20px; padding: 48px 44px; margin-bottom: 32px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.02);
    animation: fadeInUp 0.6s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.page-title { 
    font-size:2.2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg, #0F172A 30%, #0284C7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:10px; 
    font-family: 'Outfit', sans-serif;
}
.page-sub { color:#475569; font-size:1.05rem; line-height:1.6; max-width:700px;}

.panel {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:18px; padding:32px; margin-bottom:24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    animation: fadeInUp 0.8s ease-out;
}
.panel-title { 
    font-size:1.2rem; font-weight:700; color:#0F172A; margin-bottom:16px; 
    display:flex; align-items:center; gap:12px; font-family: 'Outfit', sans-serif;
}
.panel-body  { color:#334155; font-size:0.95rem; line-height:1.8; }
.panel-body b { color:#0F172A; font-weight:600;}

.tech-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-top:24px; }
.tech-card {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:16px; padding:24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    cursor: default;
}
.tech-card:hover { 
    border-color:#0284C7; 
    transform:translateY(-5px) scale(1.02); 
    box-shadow: 0 15px 30px rgba(2, 132, 199, 0.1); 
}
.tech-card-icon { 
    font-size:1.8rem; margin-bottom:14px; 
    display: inline-block;
    padding: 10px;
    background: rgba(2, 132, 199, 0.08);
    border-radius: 12px;
}
.tech-card-name { font-size:1.05rem; font-weight:700; color:#0F172A; margin-bottom:8px; font-family: 'Outfit', sans-serif;}
.tech-card-desc { font-size:0.85rem; color:#475569; line-height:1.6; }

.timeline { position:relative; padding-left:28px; margin-top: 20px;}
.timeline::before { content:''; position:absolute; left:9px; top:0; bottom:0; width:2px;
    background:linear-gradient(180deg,#0284C7,rgba(2, 132, 199, 0.1)); border-radius:2px; }
.tl-item { position:relative; margin-bottom:28px; transition: transform 0.2s ease;}
.tl-item:hover { transform: translateX(4px); }
.tl-dot { position:absolute; left:-24px; top:4px; width:12px; height:12px; border-radius:50%;
    background:#0284C7; border:2px solid #FFFFFF; box-shadow:0 0 0 3px rgba(2, 132, 199, 0.2); }
.tl-title { font-size:1rem; font-weight:700; color:#0F172A; margin-bottom:6px; font-family: 'Outfit', sans-serif;}
.tl-desc  { font-size:0.9rem; color:#475569; line-height:1.6; }

@keyframes pulseWarning {
    0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
    100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}
.disclaimer {
    background: linear-gradient(to right, rgba(254, 243, 199, 0.8), rgba(255, 251, 235, 0.9));
    border-left: 4px solid #F59E0B;
    border-radius: 12px; padding: 20px 24px;
    font-size: 0.95rem; color: #92400E; line-height: 1.6; 
    margin-top: 32px; margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    display: flex; align-items: flex-start; gap: 16px;
    animation: fadeInUp 1s ease-out;
}
.disclaimer-icon {
    font-size: 1.5rem;
    background: #FFFBEB;
    border-radius: 50%;
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    animation: pulseWarning 2s infinite;
}
.disclaimer-content b {
    color: #B45309;
    font-size: 1rem;
    display: block;
    margin-bottom: 4px;
}
</style>

<div class="page-hero">
    <div class="page-title">About NeuroPeds AI</div>
    <div class="page-sub">Advancing decentralized pediatric neuro-oncology through privacy-first federated AI and clinical decision support.</div>
</div>

<div class="panel">
    <div class="panel-title">🎯 Research Objectives & Motivation</div>
    <div class="panel-body">
        Pediatric brain tumors present unique morphological variations and diffuse margins compared to adult neuro-oncology cases.
        Training robust deep learning models requires large multi-institutional datasets; however, patient privacy regulations and
        hospital data silos strictly limit raw data sharing.
        <br><br>
        <b>NeuroPeds AI</b> solves this by combining <b>Federated Learning (FedAvg)</b> with <b>CORAL Domain Adaptation</b>,
        allowing hospitals to collaboratively train state-of-the-art 3D U-Net segmentation models without ever transferring
        sensitive patient MRI scans across institutional boundaries.
    </div>
</div>

<div class="panel">
    <div class="panel-title">🧱 Core Technology Stack</div>
    <div class="tech-grid">
        <div class="tech-card">
            <div class="tech-card-icon">🌐</div>
            <div class="tech-card-name">Federated Learning</div>
            <div class="tech-card-desc">FedAvg protocol across 5 hospital nodes with secure gradient encryption and no raw data transfer.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">🧠</div>
            <div class="tech-card-name">3D U-Net Segmentation</div>
            <div class="tech-card-desc">Volumetric encoder-decoder with residual connections trained on 96³ voxel FP16 inputs.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">🧬</div>
            <div class="tech-card-name">CORAL Adaptation</div>
            <div class="tech-card-desc">Second-order covariance alignment closing scanner-domain gaps across acquisition protocols.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">🔍</div>
            <div class="tech-card-name">Explainable AI</div>
            <div class="tech-card-desc">Grad-CAM and attention rollout maps grounding predictions in anatomy radiologists recognize.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">⚡</div>
            <div class="tech-card-name">FP16 Mixed Precision</div>
            <div class="tech-card-desc">2× throughput over FP32 with loss-scaled training for numerical stability.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">📄</div>
            <div class="tech-card-name">Clinical PDF Reports</div>
            <div class="tech-card-desc">Automated publication-ready outputs with per-subregion statistics and model confidence.</div>
        </div>
    </div>
</div>

<div class="panel">
    <div class="panel-title">📅 Research Timeline</div>
    <div class="timeline">
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 1 — Dataset Curation</div>
            <div class="tl-desc">BraTS-PEDs 2024 acquisition: 257 pediatric subjects across 4 MRI modalities with expert annotations.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 2 — Centralized Baseline</div>
            <div class="tl-desc">3D U-Net trained centrally to establish Dice / IoU / HD95 benchmark targets.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 3 — Federated Training</div>
            <div class="tl-desc">FedAvg deployment across 5 simulated hospital nodes; convergence validated over 50 rounds.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 4 — Domain Adaptation</div>
            <div class="tl-desc">CORAL alignment layer added; domain gap reduced from 0.684 to 0.042 on held-out site.</div>
        </div>
        <div class="tl-item" style="margin-bottom:0"><div class="tl-dot"></div>
            <div class="tl-title">Phase 5 — Clinical Integration</div>
            <div class="tl-desc">XAI overlays, PDF export, and this Streamlit platform — ready for research demonstration.</div>
        </div>
    </div>
</div>

<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-content">
        <b>Medical Disclaimer</b>
        NeuroPeds AI is a research-grade decision support platform designed to assist clinicians and researchers.
        It does not replace independent clinical diagnosis or professional medical judgment. All outputs are for research purposes only.
    </div>
</div>
""", unsafe_allow_html=True)