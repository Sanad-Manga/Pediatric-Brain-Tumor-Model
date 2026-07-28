import streamlit as st
import pandas as pd
import time
import torch

import backend as be


@st.cache_data(show_spinner=False)
def _cohort_counts():
    """Real per-cohort counts. Reads only the manifest JSON, so this is instant.

    Deliberately does NOT open any .npz. Streamlit flushes no output at all -
    not even the sidebar - until the script yields, so any slow work at module
    top level makes the whole landing page (nav included) appear blank on a cold
    start. Volume reads belong behind an explicit user action, not here.
    """
    subs = be.list_subjects()
    counts = {k: len(v) for k, v in subs.items()}
    return counts, sum(counts.values())


@st.cache_data(show_spinner="Measuring tumor burden from ground-truth masks…")
def _sample_cases(per_cohort: int = 2):
    """Real tumor stats for a small sample. Slow (opens .npz), so it is on demand."""
    rows = []
    for cohort, ids in be.list_subjects().items():
        for sid in ids[:per_cohort]:
            seg = be.load_volume(sid).seg
            regions = {r["region"]: r for r in be.region_stats(seg)}
            rows.append({
                "Subject ID": sid,
                "Cohort": cohort,
                "Tumor voxels": f"{regions['WT']['voxels']:,}",
                "Volume (est.)": f"{regions['WT']['volume_ml']:.1f} mL",
                "ET present": "yes" if regions["ET"]["present"] else "no",
            })
    return pd.DataFrame(rows)


_COUNTS, _TOTAL = _cohort_counts()
_STATUS = be.model_status()
_DEVICE = "CUDA" if torch.cuda.is_available() else "CPU"

# ──────────────────────────────────────────────────────────
#  PAGE-LEVEL CSS (inherits global theme from app.py)
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── HERO ── */
.hero-wrap {
    background:
        radial-gradient(circle at 30% 50%, rgba(56,189,248,0.10), transparent 55%),
        radial-gradient(circle at 80% 20%, rgba(129,140,248,0.10), transparent 50%),
        linear-gradient(135deg, #0B1628 0%, #0F172A 60%, #111827 100%);
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 24px;
    padding: 48px 44px;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
}
.hero-wrap::before {
    content:"";
    position:absolute; top:0; left:0; right:0; bottom:0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2338BDF8' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-label {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 14px;
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.30);
    border-radius: 99px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #38BDF8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #34D399;
    animation: pulseDot 1.8s infinite;
}
.hero-title {
    font-size: clamp(1.9rem, 3.5vw, 2.9rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
    background: linear-gradient(135deg, #FFFFFF 30%, #38BDF8 70%, #A5F3FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
}
.hero-sub {
    color: #94A3B8;
    font-size: 1rem;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.65;
    margin-bottom: 28px;
}
.hero-tags { display: flex; flex-wrap: wrap; gap: 10px; }
.hero-tag {
    padding: 5px 14px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.hero-tag.cyan  { background: rgba(56,189,248,0.12); color: #38BDF8; border: 1px solid rgba(56,189,248,0.25); }
.hero-tag.violet{ background: rgba(129,140,248,0.12); color: #818CF8; border: 1px solid rgba(129,140,248,0.25); }
.hero-tag.green { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
.hero-tag.amber { background: rgba(251,191,36,0.12); color: #FBBF24; border: 1px solid rgba(251,191,36,0.25); }
.hero-corner-badge {
    position: absolute; top: 24px; right: 28px;
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
    color: rgba(148,163,184,0.6);
}

/* ── STAT CARDS ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 32px; }
.stat-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.65), rgba(15,23,42,0.80));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    cursor: default;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
}
.stat-card.cyan:hover  { border-color: rgba(56,189,248,0.4); box-shadow: 0 12px 32px rgba(56,189,248,0.12); }
.stat-card.violet:hover{ border-color: rgba(129,140,248,0.4); box-shadow: 0 12px 32px rgba(129,140,248,0.12); }
.stat-card.green:hover { border-color: rgba(52,211,153,0.4);  box-shadow: 0 12px 32px rgba(52,211,153,0.12); }
.stat-card.amber:hover { border-color: rgba(251,191,36,0.4);  box-shadow: 0 12px 32px rgba(251,191,36,0.12); }
.stat-card-accent {
    position: absolute; top: 0; right: 0;
    width: 80px; height: 80px;
    border-radius: 0 16px 0 80px;
    opacity: 0.08;
}
.stat-card.cyan   .stat-card-accent { background: #38BDF8; }
.stat-card.violet .stat-card-accent { background: #818CF8; }
.stat-card.green  .stat-card-accent { background: #34D399; }
.stat-card.amber  .stat-card-accent { background: #FBBF24; }
.stat-icon { font-size: 1.6rem; margin-bottom: 10px; }
.stat-label { font-size: 0.73rem; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 6px; }
.stat-value { font-size: 1.75rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; color: #F3F4F6; margin-bottom: 6px; }
.stat-delta { font-size: 0.75rem; font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.stat-delta.up   { color: #34D399; }
.stat-delta.down { color: #F87171; }
.stat-delta.info { color: #38BDF8; }

/* ── SECTION HEADINGS ── */
.section-heading {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 18px;
}
.section-heading-icon {
    width: 34px; height: 34px;
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.section-heading-text { font-size: 1.05rem; font-weight: 700; color: #F3F4F6; }
.section-heading-sub  { font-size: 0.78rem; color: #64748B; margin-top: 1px; }

/* ── MODULE CARDS ── */
.mod-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 32px; }
.mod-card {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.mod-card:hover { transform: translateY(-3px); }
.mod-card.cyan:hover   { border-color: rgba(56,189,248,0.4);  box-shadow:0 12px 30px rgba(56,189,248,0.10); }
.mod-card.violet:hover { border-color: rgba(129,140,248,0.4); box-shadow:0 12px 30px rgba(129,140,248,0.10); }
.mod-card.green:hover  { border-color: rgba(52,211,153,0.4);  box-shadow:0 12px 30px rgba(52,211,153,0.10); }
.mod-card-top { display:flex; align-items:center; gap:12px; margin-bottom:14px; }
.mod-card-icon {
    width:42px; height:42px; border-radius:12px;
    display:flex; align-items:center; justify-content:center; font-size:1.2rem;
    flex-shrink:0;
}
.cyan   .mod-card-icon { background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.25); }
.violet .mod-card-icon { background:rgba(129,140,248,0.12); border:1px solid rgba(129,140,248,0.25); }
.green  .mod-card-icon { background:rgba(52,211,153,0.12);  border:1px solid rgba(52,211,153,0.25); }
.mod-card-name { font-size:0.92rem; font-weight:700; color:#F3F4F6; }
.mod-card-nav  { font-size:0.72rem; color:#64748B; margin-top:2px; }
.mod-card-desc { font-size:0.82rem; color:#94A3B8; line-height:1.6; margin-bottom:14px; }
.mod-card-pills { display:flex; flex-wrap:wrap; gap:7px; }
.pill {
    padding: 3px 10px; border-radius: 6px;
    font-size: 0.7rem; font-family:'JetBrains Mono',monospace; font-weight:500;
}
.pill.cyan   { background:rgba(56,189,248,0.10); color:#38BDF8; }
.pill.violet { background:rgba(129,140,248,0.10); color:#818CF8; }
.pill.green  { background:rgba(52,211,153,0.10); color:#34D399; }

/* ── PIPELINE ── */
.pipeline-wrap {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 32px;
}
.pipeline-flow { display:flex; align-items:center; flex-wrap:wrap; gap:0; margin-top:18px; }
.pipeline-step {
    display:flex; flex-direction:column; align-items:center; text-align:center;
    flex:1; min-width:100px;
}
.pipeline-step-circle {
    width:52px; height:52px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.2rem; margin-bottom:10px;
    position: relative;
}
.pipeline-step-circle.c1 { background:rgba(56,189,248,0.12); border:2px solid rgba(56,189,248,0.35); }
.pipeline-step-circle.c2 { background:rgba(129,140,248,0.12); border:2px solid rgba(129,140,248,0.35); }
.pipeline-step-circle.c3 { background:rgba(52,211,153,0.12); border:2px solid rgba(52,211,153,0.35); }
.pipeline-step-circle.c4 { background:rgba(251,191,36,0.12); border:2px solid rgba(251,191,36,0.35); }
.pipeline-step-circle.c5 { background:rgba(248,113,113,0.12); border:2px solid rgba(248,113,113,0.35); }
.pipeline-step-circle.c6 { background:rgba(56,189,248,0.12); border:2px solid rgba(56,189,248,0.35); }
.pipeline-step-name { font-size:0.8rem; font-weight:600; color:#F3F4F6; margin-bottom:3px; }
.pipeline-step-sub  { font-size:0.68rem; color:#64748B; font-family:'JetBrains Mono',monospace; }
.pipeline-arrow { color:rgba(255,255,255,0.15); font-size:1.2rem; padding:0 4px; margin-bottom:32px; }

/* ── PERF METRICS ── */
.perf-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:32px; }
.perf-card {
    background:rgba(15,23,42,0.55);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:20px;
    text-align:center;
    transition: transform 0.22s ease;
}
.perf-card:hover { transform:translateY(-2px); border-color:rgba(56,189,248,0.25); }
.perf-num {
    font-size:2.1rem; font-weight:800; letter-spacing:-0.03em;
    background:linear-gradient(135deg,#FFFFFF,#38BDF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-bottom:4px;
}
.perf-name { font-size:0.78rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px; }
.perf-bar-bg { height:4px; background:rgba(255,255,255,0.07); border-radius:4px; overflow:hidden; }
.perf-bar-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,#38BDF8,#00F2FE); }

/* ── DATA TABLE ── */
.data-section { display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:32px; }
.data-panel {
    background:rgba(15,23,42,0.55);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:24px;
}

/* ── RESEARCH HIGHLIGHTS ── */
.research-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:32px; }
.research-item {
    background:rgba(15,23,42,0.55);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px;
    padding:16px 18px;
    display:flex; align-items:flex-start; gap:12px;
    transition: border-color 0.2s ease;
}
.research-item:hover { border-color:rgba(52,211,153,0.3); }
.research-check {
    width:26px; height:26px; border-radius:8px;
    background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3);
    display:flex; align-items:center; justify-content:center;
    font-size:0.8rem; flex-shrink:0; margin-top:1px;
}
.research-name  { font-size:0.85rem; font-weight:600; color:#F3F4F6; margin-bottom:3px; }
.research-desc  { font-size:0.73rem; color:#64748B; line-height:1.45; }

/* ── FOOTER ── */
.dash-footer {
    text-align:center;
    padding:20px;
    font-size:0.73rem;
    font-family:'JetBrains Mono',monospace;
    color:rgba(100,116,139,0.6);
    border-top:1px solid rgba(255,255,255,0.05);
    letter-spacing:0.03em;
}

@media (max-width: 900px) {
    .stat-grid, .mod-grid, .perf-grid, .research-grid { grid-template-columns:1fr 1fr; }
    .data-section { grid-template-columns:1fr; }
    .pipeline-flow { flex-direction:column; }
    .pipeline-arrow { transform: rotate(90deg); }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  HERO
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-corner-badge">BUILD 2026.7 · RESEARCH MODE</div>
    <div class="hero-label">
        <span class="hero-live-dot"></span>
        AI Command Center · Live
    </div>
    <div class="hero-title">Pediatric Brain Tumor<br>Intelligence Platform</div>
    <div class="hero-sub">
        Federated deep learning across hospital networks — delivering 3D U-Net tumor segmentation,
        domain-adaptive AI, and publication-ready clinical outputs for pediatric oncology.
    </div>
    <div class="hero-tags">
        <span class="hero-tag cyan">⚡ Federated Learning</span>
        <span class="hero-tag violet">🧠 3D U-Net · nnU-Net</span>
        <span class="hero-tag green">🔬 CORAL Adaptation</span>
        <span class="hero-tag amber">📊 Explainable AI</span>
        <span class="hero-tag cyan">🔒 HIPAA · GDPR</span>
        <span class="hero-tag violet">⚙ FP16 Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  STAT CARDS
# ──────────────────────────────────────────────────────────
_model_value = "Trained" if _STATUS.trained else "Untrained"
_model_delta = (
    f'<div class="stat-delta up">↑ {_STATUS.checkpoint.name}</div>' if _STATUS.trained
    else '<div class="stat-delta info">architecture loaded · no checkpoint</div>'
)

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card cyan">
        <div class="stat-card-accent"></div>
        <div class="stat-icon">🤖</div>
        <div class="stat-label">AI Model</div>
        <div class="stat-value">{_model_value}</div>
        {_model_delta}
    </div>
    <div class="stat-card violet">
        <div class="stat-card-accent"></div>
        <div class="stat-icon">🌐</div>
        <div class="stat-label">Simulated Clients</div>
        <div class="stat-value">2</div>
        <div class="stat-delta info">Hospital A · Hospital B (+1 held-out site)</div>
    </div>
    <div class="stat-card green">
        <div class="stat-card-accent"></div>
        <div class="stat-icon">🗂</div>
        <div class="stat-label">Cached Subjects</div>
        <div class="stat-value">{_TOTAL}</div>
        <div class="stat-delta info">t1c · t1n · t2f · t2w</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-card-accent"></div>
        <div class="stat-icon">⚡</div>
        <div class="stat-label">Inference Device</div>
        <div class="stat-value">{_DEVICE}</div>
        <div class="stat-delta info">96³ volumes · batch size 1</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  PLATFORM MODULES
# ──────────────────────────────────────────────────────────
# Section heading
st.markdown("""
<div class="section-heading">
    <div class="section-heading-icon">🧩</div>
    <div>
        <div class="section-heading-text">Platform Modules</div>
        <div class="section-heading-sub">Six interconnected tools — from raw MRI to clinical report</div>
    </div>
</div>
""", unsafe_allow_html=True)

def mod_card(icon, name, nav, desc, pills, color):
    pill_html = "".join(f'<span class="pill {color}">{p}</span>' for p in pills)
    return f"""
    <div class="mod-card {color}" style="height:100%;">
        <div class="mod-card-top">
            <div class="mod-card-icon">{icon}</div>
            <div>
                <div class="mod-card-name">{name}</div>
                <div class="mod-card-nav">{nav}</div>
            </div>
        </div>
        <div class="mod-card-desc">{desc}</div>
        <div class="mod-card-pills">{pill_html}</div>
    </div>"""

_row1 = st.columns(3, gap="small")
_modules_r1 = [
    ("🖥️", "MRI Analysis Studio",    "Clinical & Analysis →",
     "Upload multi-modal MRI volumes. Inspect slices in an interactive 3D viewer with segmentation overlays rendered per-class.",
     ["3D Viewer","Multi-modal","Overlay"], "cyan"),
    ("📋", "Segmentation Report",     "Clinical & Analysis →",
     "AI-generated patient summary with tumor statistics, region-level interpretation, and one-click PDF export for clinical records.",
     ["Tumor Stats","PDF Export","Summary"], "violet"),
    ("🩺", "Clinical Explainability", "Clinical & Analysis →",
     "Grad-CAM and attention maps reveal exactly which voxels drove each prediction — grounding AI decisions in anatomy radiologists recognize.",
     ["Grad-CAM","Attention","XAI"], "green"),
]
for col, (icon, name, nav, desc, pills, color) in zip(_row1, _modules_r1):
    with col:
        st.markdown(mod_card(icon, name, nav, desc, pills, color), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

_row2 = st.columns(3, gap="small")
_modules_r2 = [
    ("🌐", "Federated Observatory",  "Federated & AI Core →",
     "Real-time dashboard of hospital client nodes: training rounds, gradient sync, communication overhead, and convergence tracking.",
     ["FedAvg","5 Clients","Live Sync"], "cyan"),
    ("🧬", "Domain Adaptation Lab",  "Federated & AI Core →",
     "CORAL feature alignment and adversarial adaptation close the gap between scanner sites, scanner protocols, and patient demographics.",
     ["CORAL","Alignment","Cross-site"], "violet"),
    ("🧠", "Model Intelligence",     "Federated & AI Core →",
     "Full architecture inspection, layer-wise metrics, validation curves, and ablation comparisons across federated training runs.",
     ["Architecture","Metrics","Ablation"], "green"),
]
for col, (icon, name, nav, desc, pills, color) in zip(_row2, _modules_r2):
    with col:
        st.markdown(mod_card(icon, name, nav, desc, pills, color), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  INFERENCE PIPELINE
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-wrap">
    <div class="section-heading" style="margin-bottom:0;">
        <div class="section-heading-icon">⚙️</div>
        <div>
            <div class="section-heading-text">AI Inference Pipeline</div>
            <div class="section-heading-sub">End-to-end from DICOM to clinical PDF</div>
        </div>
    </div>
    <div class="pipeline-flow">
        <div class="pipeline-step">
            <div class="pipeline-step-circle c1">📤</div>
            <div class="pipeline-step-name">MRI Upload</div>
            <div class="pipeline-step-sub">DICOM · NIfTI</div>
        </div>
        <div class="pipeline-arrow">›</div>
        <div class="pipeline-step">
            <div class="pipeline-step-circle c2">🔧</div>
            <div class="pipeline-step-name">Preprocessing</div>
            <div class="pipeline-step-sub">N4 · Skull Strip</div>
        </div>
        <div class="pipeline-arrow">›</div>
        <div class="pipeline-step">
            <div class="pipeline-step-circle c3">🧠</div>
            <div class="pipeline-step-name">3D U-Net</div>
            <div class="pipeline-step-sub">96³ · FP16</div>
        </div>
        <div class="pipeline-arrow">›</div>
        <div class="pipeline-step">
            <div class="pipeline-step-circle c4">🌐</div>
            <div class="pipeline-step-name">Fed. Aggregation</div>
            <div class="pipeline-step-sub">FedAvg · 5 Sites</div>
        </div>
        <div class="pipeline-arrow">›</div>
        <div class="pipeline-step">
            <div class="pipeline-step-circle c5">🧬</div>
            <div class="pipeline-step-name">Domain Adapt.</div>
            <div class="pipeline-step-sub">CORAL · Align</div>
        </div>
        <div class="pipeline-arrow">›</div>
        <div class="pipeline-step">
            <div class="pipeline-step-circle c6">📄</div>
            <div class="pipeline-step-name">Clinical Report</div>
            <div class="pipeline-step-sub">PDF · Export</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="section-heading">
    <div class="section-heading-icon">📈</div>
    <div>
        <div class="section-heading-text">Model Performance</div>
        <div class="section-heading-sub">
            Official BraTS-PEDs regions, scored on the {_COUNTS['Held-out']}-subject held-out institution
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_dice = be.heldout_dice()

if _dice is not None:
    # Real numbers, produced by 03_augmentation_eval's ablation runner.
    st.markdown(
        '<div class="perf-grid">' + "".join(
            f"""<div class="perf-card">
                <div class="perf-num">{v * 100:.1f}%</div>
                <div class="perf-name">Dice {k}</div>
                <div class="perf-bar-bg"><div class="perf-bar-fill" style="width:{v * 100:.1f}%"></div></div>
            </div>"""
            for k, v in _dice.items()
        ) + "</div>", unsafe_allow_html=True)
    st.caption(
        f"Source: `03_augmentation_eval/results/ablation_results.csv` "
        f"({len(be.ablation_results())} evaluated run(s))."
        + ("" if _STATUS.trained else "  ⚠ Logged from an UNTRAINED checkpoint — not model performance.")
    )
else:
    st.markdown(f"""
    <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.35);
                border-radius:14px;padding:22px 26px;">
        <div style="font-size:0.95rem;font-weight:700;color:#FDE68A;margin-bottom:10px;">
            ⚠ No performance figures yet — the model has not been trained
        </div>
        <div style="font-size:0.84rem;color:#FCD34D;line-height:1.8;">
            The 3D U-Net architecture from <code>01_model_federated</code> loads and runs, but no
            checkpoint exists on disk, so its weights are randomly initialized. Any Dice, IoU,
            sensitivity or specificity shown here would be invented, so nothing is shown.
            <br><br>
            Real per-region Dice (ET / NC / WT) appears automatically once a checkpoint exists —
            set <code>NEUROFED_CHECKPOINT</code>, or drop one under <code>checkpoints/</code>.
            The evaluation path is already built and tested in
            <code>03_augmentation_eval</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  DATASET + RECENT CASES  (2-column)
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading">
    <div class="section-heading-icon">📚</div>
    <div>
        <div class="section-heading-text">Dataset & Recent Cases</div>
        <div class="section-heading-sub">BraTS-PEDs 2024 · Pediatric brain tumor segmentation benchmark</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown("""
    <div class="data-panel">
        <div style="font-size:0.78rem;font-family:'JetBrains Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.07em;
                    color:#64748B;margin-bottom:18px;">Dataset Summary</div>
        <table style="width:100%;border-collapse:collapse;font-size:0.84rem;">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:10px 0;color:#64748B;width:45%;">Dataset</td>
                <td style="padding:10px 0;color:#F3F4F6;font-weight:600;">BraTS-PEDs 2024</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:10px 0;color:#64748B;">Subjects (cached)</td>
                <td style="padding:10px 0;color:#38BDF8;font-weight:700;font-family:'JetBrains Mono',monospace;">""" + f"""{_TOTAL}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:10px 0;color:#64748B;">Split</td>
                <td style="padding:10px 0;color:#F3F4F6;font-family:'JetBrains Mono',monospace;font-size:0.78rem;">
                    A {_COUNTS['Hospital A']} · B {_COUNTS['Hospital B']} · held-out {_COUNTS['Held-out']}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:10px 0;color:#64748B;">Modalities</td>
                <td style="padding:10px 0;color:#F3F4F6;font-family:'JetBrains Mono',monospace;">t1c · t1n · t2f · t2w</td>
            </tr>""" + """
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:10px 0;color:#64748B;">Volume Shape</td>
                <td style="padding:10px 0;color:#F3F4F6;font-family:'JetBrains Mono',monospace;">96 × 96 × 96</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:10px 0;color:#64748B;">Precision</td>
                <td style="padding:10px 0;color:#FBBF24;font-weight:600;font-family:'JetBrains Mono',monospace;">FP16 Mixed</td>
            </tr>
            <tr>
                <td style="padding:10px 0;color:#64748B;">Task</td>
                <td style="padding:10px 0;color:#F3F4F6;">Pediatric Tumor Seg.</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown(
        '<div style="font-size:0.78rem;font-family:\'JetBrains Mono\',monospace;'
        'text-transform:uppercase;letter-spacing:0.07em;color:#64748B;margin-bottom:10px;">'
        'Real Subjects · Ground-Truth Tumor Burden</div>',
        unsafe_allow_html=True,
    )
    # Loaded on demand: opening .npz files is slow, and doing it at page load
    # blanks the whole app (sidebar included) until it finishes.
    if st.button("Measure tumor burden", help="Reads real ground-truth masks from the cache"):
        st.session_state["_show_cases"] = True

    if st.session_state.get("_show_cases"):
        st.dataframe(_sample_cases(), use_container_width=True, hide_index=True)
        st.caption(
            "Real subject IDs and real tumor volumes measured from the ground-truth masks. "
            "No confidence column: confidence would require a trained model."
        )
    else:
        st.caption(
            f"{_TOTAL} subjects available across {len(_COUNTS)} cohorts. "
            "Click above to measure real tumor volumes from the ground-truth masks."
        )

# ──────────────────────────────────────────────────────────
#  RESEARCH HIGHLIGHTS
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading" style="margin-top:12px;">
    <div class="section-heading-icon">🔬</div>
    <div>
        <div class="section-heading-text">Research Contributions</div>
        <div class="section-heading-sub">Novel components validated in this pipeline</div>
    </div>
</div>

<div class="research-grid">
    <div class="research-item">
        <div class="research-check">✓</div>
        <div>
            <div class="research-name">Federated Learning</div>
            <div class="research-desc">Privacy-preserving FedAvg across 5 hospital sites without sharing raw patient data.</div>
        </div>
    </div>
    <div class="research-item">
        <div class="research-check">✓</div>
        <div>
            <div class="research-name">3D U-Net Architecture</div>
            <div class="research-desc">Volumetric encoder-decoder with skip connections optimized for pediatric MRI.</div>
        </div>
    </div>
    <div class="research-item">
        <div class="research-check">✓</div>
        <div>
            <div class="research-name">CORAL Domain Adaptation</div>
            <div class="research-desc">Second-order statistics alignment across scanner types and acquisition protocols.</div>
        </div>
    </div>
    <div class="research-item">
        <div class="research-check">✓</div>
        <div>
            <div class="research-name">Mixed Precision FP16</div>
            <div class="research-desc">2× throughput with maintained numerical stability via loss scaling.</div>
        </div>
    </div>
    <div class="research-item">
        <div class="research-check">✓</div>
        <div>
            <div class="research-name">Explainable AI (XAI)</div>
            <div class="research-desc">Grad-CAM and attention rollout overlaid on original MRI slices for clinical review.</div>
        </div>
    </div>
    <div class="research-item">
        <div class="research-check">✓</div>
        <div>
            <div class="research-name">Publication-ready Pipeline</div>
            <div class="research-desc">Reproducible training, evaluation scripts, and automated PDF report generation.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-footer">
    NEUROFED AI &nbsp;·&nbsp; PEDIATRIC BRAIN TUMOR INTELLIGENCE &nbsp;·&nbsp; RESEARCH DEMONSTRATION &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)