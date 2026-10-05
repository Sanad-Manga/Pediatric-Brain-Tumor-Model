import streamlit as st

st.set_page_config(page_title="About NeuroPeds AI", page_icon="📚", layout="wide")

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
.page-sub { font-size: 0.95rem; color: #475569; line-height: 1.6; max-width: 800px;}

.panel { 
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; 
    padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); 
}
.panel-title { 
    font-size: 1.15rem; font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif;
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px; 
}
.panel-body  { color: #475569; font-size: 0.95rem; line-height: 1.7; }
.panel-body b { color: #0F172A; font-weight: 600;}

.tech-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; margin-top: 24px; }
.tech-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 24px;
    transition: all 0.2s ease;
}
.tech-card:hover { border-color: #0EA5E9; transform: translateY(-4px); box-shadow: 0 10px 25px rgba(14, 165, 233, 0.1); }
.tech-card-icon { font-size: 1.8rem; margin-bottom: 12px; }
.tech-card-name { font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 8px; font-family: 'Outfit', sans-serif;}
.tech-card-desc { font-size: 0.85rem; color: #64748B; line-height: 1.6; }

.timeline { position: relative; padding-left: 28px; margin-top: 24px;}
.timeline::before { 
    content: ''; position: absolute; left: 10px; top: 0; bottom: 0; width: 2px;
    background: #E2E8F0; border-radius: 2px; 
}
.tl-item { position: relative; margin-bottom: 30px; }
.tl-dot { 
    position: absolute; left: -24.5px; top: 4px; width: 13px; height: 13px; border-radius: 50%;
    background: #0EA5E9; box-shadow: 0 0 0 4px #F0F9FF; 
}
.tl-title { font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 6px; font-family: 'Outfit', sans-serif;}
.tl-desc  { font-size: 0.9rem; color: #475569; line-height: 1.6; }

.disclaimer {
    background: #FEF2F2;
    border-left: 4px solid #EF4444; border-radius: 12px; padding: 16px 20px;
    font-size: 0.9rem; color: #991B1B; line-height: 1.5; margin-top: 32px;
    display: flex; align-items: flex-start; gap: 12px;
}
.disclaimer-icon { font-size: 1.4rem; }
.disclaimer-content b { color: #7F1D1D; font-size: 0.95rem; display: block; margin-bottom: 4px; }
</style>

<div class="page-hero">
    <div class="page-title">📚 About NeuroPeds AI</div>
    <div class="page-sub">A clinical decision support research project focused on pediatric brain tumor segmentation using the BraTS-PEDs 2024 dataset.</div>
</div>

<div class="panel">
    <div class="panel-title">🎯 Research Objectives & Motivation</div>
    <div class="panel-body">
        Pediatric brain tumors present unique morphological variations and diffuse margins compared to adult neuro-oncology cases. 
        Training robust deep learning models requires large multi-institutional datasets; however, patient privacy regulations and 
        hospital data silos strictly limit raw data sharing.
        <br><br>
        <b>NeuroPeds AI</b> is a workshop project that explores three ideas for that problem: data augmentation, federated 
        learning (FedAvg), and CORAL domain adaptation. The model we present is <b>R*</b>, which combines a
        <b>centrally trained 2D U-Net ensemble</b> (augmentation on; federation and domain adaptation off) with a family of
        four 3D models; its results are precomputed on test scans (Dashboard and Model Comparison pages). The live MRI
        Analysis page runs the 2D ensemble alone. The federated and domain-adaptation code lives in separate research
        modules of the repository and did not produce any deployed checkpoint.
    </div>
</div>

<div class="panel">
    <div class="panel-title">🧱 Core Technology Stack</div>
    <div class="tech-grid">
        <div class="tech-card">
            <div class="tech-card-icon">🌐</div>
            <div class="tech-card-name">Federated Learning (Research)</div>
            <div class="tech-card-desc">FedAvg is implemented in module `01_model_federated`. The deployed checkpoint was trained centrally, not using this module.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">🧠</div>
            <div class="tech-card-name">2D U-Net Segmentation</div>
            <div class="tech-card-desc">Slice-wise encoder-decoder reading four MRI channels (T1c, T1n, T2f, T2w). The deployed model averages two architectures (widths 16 and 64).</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">🧬</div>
            <div class="tech-card-name">CORAL Adaptation (Research)</div>
            <div class="tech-card-desc">Covariance alignment is implemented in `01_model_federated`, with a feature-space visualization in `02_domain_adaptation`.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">🔍</div>
            <div class="tech-card-name">Explainability</div>
            <div class="tech-card-desc">Not implemented in this demo. Grad-CAM, attention, and uncertainty maps are planned for future iterations.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">⚡</div>
            <div class="tech-card-name">FP16 Mixed Precision</div>
            <div class="tech-card-desc">Mixed-precision (AMP) training with dynamic loss scaling optimized for CUDA environments.</div>
        </div>
        <div class="tech-card">
            <div class="tech-card-icon">📄</div>
            <div class="tech-card-name">Clinical PDF Reports</div>
            <div class="tech-card-desc">Per-subregion statistics generation and PDF export, clearly marked for research use only.</div>
        </div>
    </div>
</div>

<div class="panel">
    <div class="panel-title">📅 Research Timeline</div>
    <div class="timeline">
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 1 — Dataset Curation</div>
            <div class="tl-desc">BraTS-PEDs 2024: 227 subjects curated (53 + 92 training, 82 held-out), 4 MRI modalities with expert annotations.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 2 — Centralized Baseline</div>
            <div class="tl-desc">2D U-Net trained centrally with robust augmentation. Scored per patient using Dice and HD95 on the held-out dataset.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 3 — Federated Training</div>
            <div class="tl-desc">FedAvg implemented as an isolated research module (`01_model_federated`). No results from it are reported in this demo.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 4 — Domain Adaptation</div>
            <div class="tl-desc">CORAL alignment and feature-space visualizations implemented for future cross-institutional studies.</div>
        </div>
        <div class="tl-item"><div class="tl-dot"></div>
            <div class="tl-title">Phase 5 — Research Demo</div>
            <div class="tl-desc">Deployment of this Streamlit app featuring segmentation overlays and PDF export, running the 2D ensemble (0.754 held-out mean Dice).</div>
        </div>
        <div class="tl-item" style="margin-bottom:0"><div class="tl-dot"></div>
            <div class="tl-title">Phase 6 — R* (Current)</div>
            <div class="tl-desc">R* combines the 2D ensemble with a family of four 3D models: 0.805 mean Dice on 30 never-seen test patients (2D alone 0.711, 3D alone 0.747), 0.810 on the 81 held-out patients. Compared side by side on the Model Comparison page.</div>
        </div>
    </div>
</div>

<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-content">
        <b>Medical Disclaimer</b>
        NeuroPeds AI is a research-grade decision support platform designed to assist clinicians and researchers. It does not replace independent clinical diagnosis or professional medical judgment. All outputs are for research purposes only.
    </div>
</div>
""", unsafe_allow_html=True)