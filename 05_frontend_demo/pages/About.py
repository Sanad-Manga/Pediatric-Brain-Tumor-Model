import streamlit as st

st.set_page_config(page_title="About Research | NeuroFed AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 20% 50%, rgba(56,189,248,0.09), transparent 50%),
                linear-gradient(135deg,#F8FAFC,#E0F2FE);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px; padding: 40px 44px; margin-bottom: 28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#0F172A 30%,#0284C7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#64748B; font-size:0.92rem; }
.panel {
    background:#FFFFFF; border:1px solid #E2E8F0;
    border-radius:16px; padding:28px; margin-bottom:20px;
}
.panel-title { font-size:1rem; font-weight:700; color:#0F172A; margin-bottom:14px; display:flex; align-items:center; gap:10px; }
.panel-body  { color:#334155; font-size:0.88rem; line-height:1.8; }
.panel-body b { color:#0F172A; }
.tech-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:20px; }
.tech-card {
    background:rgba(10,15,35,0.6); border:1px solid #E2E8F0;
    border-radius:12px; padding:18px;
    transition: border-color .2s ease, transform .2s ease;
}
.tech-card:hover { border-color:rgba(56,189,248,0.3); transform:translateY(-2px); }
.tech-card-icon { font-size:1.4rem; margin-bottom:10px; }
.tech-card-name { font-size:0.85rem; font-weight:700; color:#0F172A; margin-bottom:4px; }
.tech-card-desc { font-size:0.75rem; color:#64748B; line-height:1.5; }
.timeline { position:relative; padding-left:24px; }
.timeline::before { content:''; position:absolute; left:7px; top:0; bottom:0; width:2px;
    background:linear-gradient(180deg,#0284C7,rgba(129,140,248,0.3)); border-radius:2px; }
.tl-item { position:relative; margin-bottom:22px; }
.tl-dot { position:absolute; left:-21px; top:4px; width:10px; height:10px; border-radius:50%;
    background:#0284C7; border:2px solid #0F172A; box-shadow:0 0 8px rgba(56,189,248,0.5); }
.tl-title { font-size:0.85rem; font-weight:700; color:#0F172A; margin-bottom:3px; }
.tl-desc  { font-size:0.78rem; color:#64748B; line-height:1.5; }
.disclaimer {
    background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
    border-radius:12px; padding:16px 20px;
    font-size:0.83rem; color:#FCA5A5; line-height:1.6; margin-top:8px;
}
</style>

<div class="page-hero">
    <div class="page-title">About BraTS-PEDs Research</div>
    <div class="page-sub">Advancing decentralized pediatric neuro-oncology through privacy-first federated AI.</div>
</div>

<div class="panel">
    <div class="panel-title">🎯 Research Objectives & Motivation</div>
    <div class="panel-body">
        Pediatric brain tumors present unique morphological variations and diffuse margins compared to adult neuro-oncology cases.
        Training robust deep learning models requires large multi-institutional datasets; however, patient privacy regulations and
        hospital data silos strictly limit raw data sharing.
        <br><br>
        <b>NeuroFed AI</b> solves this by combining <b>Federated Learning (FedAvg)</b> with <b>CORAL Domain Adaptation</b>,
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
    <b>⚠️ Medical Disclaimer:</b> NeuroFed AI is a research-grade decision support platform designed to assist clinicians and researchers.
    It does not replace independent clinical diagnosis or professional medical judgment. All outputs are for research purposes only.
</div>
""", unsafe_allow_html=True)