import streamlit as st
import time
from datetime import datetime
import json
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

# ──────────────────────────────────────────────────────────
#  CONFIG & LOAD DATA
# ──────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "roc_cache.json"
MODEL_METRICS_PATH = Path(__file__).resolve().parents[1] / "data" / "model_metrics.json"   # utils/build_model_metrics.py
FLAG_STUDY_PATH = Path(__file__).resolve().parents[1] / "data" / "review_flag_study.json"

@st.cache_data
def load_metrics():
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

metrics_data = load_metrics()


@st.cache_data
def load_model_metrics():
    try:
        with open(MODEL_METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


model_metrics = load_model_metrics()

if metrics_data:
    epoch_val = metrics_data.get("checkpoint", {}).get("epochs_completed", 0)
    subjects = metrics_data.get("n_subjects", 82)
    mean_dice = metrics_data.get("checkpoint", {}).get("best_mean_dice", 0.0)
    wt_hd95 = metrics_data["regions"]["WT"]["hd95_median_mm"]
else:
    epoch_val = 0; subjects = 0; mean_dice = 0.0; wt_hd95 = 0.0

# ──────────────────────────────────────────────────────────
#  PAGE-LEVEL CSS
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-wrap {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
    border: 1px solid rgba(14, 165, 233, 0.15);
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 32px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}
.hero-label {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px;
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.2);
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #0EA5E9;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10B981;
    animation: pulseDot 2s infinite;
}
@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.hero-title {
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 12px;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}
.hero-sub { color: #475569; font-size: 1.05rem; max-width: 700px; margin-bottom: 24px; }

.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 40px; }
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-4px); border-color: #0EA5E9; box-shadow: 0 8px 20px rgba(14, 165, 233, 0.08); }
.stat-icon { font-size: 1.8rem; margin-bottom: 12px; }
.stat-label { font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; margin-bottom: 8px; }
.stat-value { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 4px; font-family: 'Outfit', sans-serif;}
.stat-delta { font-size: 0.8rem; font-weight: 600; color: #10B981; }

.section-title { font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; color: #0F172A; margin-bottom: 8px; }
.section-subtitle { font-size: 0.95rem; color: #64748B; margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  HERO SECTION
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">
        <span class="hero-live-dot"></span> Clinical Dashboard
    </div>
    <div class="hero-title">Pediatric Brain Tumor<br>Performance Analytics</div>
    <div class="hero-sub">
        How accurate each model is on patients it never trained on. R* (the 2D model and the 3D family combined) is the model we present; the 2D and 3D models are shown for comparison.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  DYNAMIC STAT CARDS
# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
#  R* vs 2D vs 3D — same patients, same measurement
# ──────────────────────────────────────────────────────────
MODEL_NAMES = {"rstar": "R* (2D + 3D combined)", "2d": "2D model", "3d": "3D family"}
if model_metrics:
    f30, ho = model_metrics["sets"]["fresh30"], model_metrics["sets"]["heldout"]
    r_f30, r_ho = f30["models"]["rstar"]["per_patient_mean_dice"], ho["models"]["rstar"]["per_patient_mean_dice"]
    st.markdown('<div class="section-title">🏆 R* compared with the 2D and 3D models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Mean Dice per patient, averaged over the three tumour regions (ET, TC, WT). '
                'Fresh-30 is the clean test: 30 patients never used for any training or tuning decision.</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card" style="border-color:#0EA5E9;">
        <div class="stat-icon">🎯</div>
        <div class="stat-label">R* · clean test</div>
        <div class="stat-value">{r_f30:.3f}</div>
        <div class="stat-delta">{f30['n_patients']} unseen patients</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🧪</div>
        <div class="stat-label">R* · held-out</div>
        <div class="stat-value">{r_ho:.3f}</div>
        <div class="stat-delta" style="color:#64748B;">{ho['n_patients']} patients (some R* settings tuned here)</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🖼️</div>
        <div class="stat-label">2D model · clean test</div>
        <div class="stat-value">{f30['models']['2d']['per_patient_mean_dice']:.3f}</div>
        <div class="stat-delta" style="color:#64748B;">R* is {r_f30 - f30['models']['2d']['per_patient_mean_dice']:+.3f}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🧊</div>
        <div class="stat-label">3D family · clean test</div>
        <div class="stat-value">{f30['models']['3d']['per_patient_mean_dice']:.3f}</div>
        <div class="stat-delta" style="color:#64748B;">R* is {r_f30 - f30['models']['3d']['per_patient_mean_dice']:+.3f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    set_choice = st.radio("Test set for the table", ["Fresh-30 (clean test)", "Held-out (81)"], horizontal=True,
                          key="dash_set")
    chosen = f30 if set_choice.startswith("Fresh") else ho
    region_names = {"ET": "Enhancing (ET)", "TC": "Core (TC)", "WT": "Whole (WT)"}
    rows = []
    for reg in ("ET", "TC", "WT"):
        for key in ("rstar", "2d", "3d"):
            m = chosen["models"][key]["pooled"][reg]
            rows.append({"Region": region_names[reg], "Model": MODEL_NAMES[key], "Dice": f"{m['dice']:.3f}",
                         "Sensitivity": f"{m['sensitivity']:.3f}", "Precision": f"{m['precision']:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Table: voxels of all patients in the set pooled together. Sensitivity = share of the real tumour the "
               "model found; precision = share of what it marked that is really tumour. Specificity is left out: "
               "background is almost the whole scan, so it is about 0.999 for every model and tells them apart by nothing.")

    try:
        flag_study = json.loads(FLAG_STUDY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        flag_study = None
    if flag_study:
        st.divider()
        st.markdown('<div class="section-title">🟣 Enhancing tumour: silent errors with review flags</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">A silent error is a mistake nobody is told about. Today R* deletes '
                    "all enhancing tumour when a patient's total is small. With review flags, uncertain spots are "
                    'shown as "please review" instead (magenta on the Model Comparison page).</div>',
                    unsafe_allow_html=True)
        fs = flag_study["sets"]["fresh30"]; t, w = fs["today"], fs["with_flags"]
        st.markdown(f"""
<div class="stat-grid" style="grid-template-columns: repeat(3,1fr);">
    <div class="stat-card">
        <div class="stat-label">Silent false alarms</div>
        <div class="stat-value">{t['silent_false']} → {w['silent_false']}</div>
        <div class="stat-delta">fresh-30, today → with flags</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Silently missed lesions</div>
        <div class="stat-value">{t['silent_missed']} → {w['silent_missed']}</div>
        <div class="stat-delta" style="color:#64748B;">of {fs['lesions']} expert lesions</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Spots to review</div>
        <div class="stat-value">{w['review_load']:.1f}</div>
        <div class="stat-delta" style="color:#64748B;">per patient ({w['flags_real']} real, {w['flags_false']} not)</div>
    </div>
</div>
""", unsafe_allow_html=True)
        ho = flag_study["sets"]["heldout"]
        st.caption(f"Rule: {flag_study['rule']} Held-out (81 patients, where the rule was chosen): silent false alarms "
                   f"{ho['today']['silent_false']} → {ho['with_flags']['silent_false']}, silently missed lesions "
                   f"{ho['today']['silent_missed']} → {ho['with_flags']['silent_missed']} of {ho['lesions']}. "
                   "Flags remove silent false alarms; most missed lesions are never predicted by any model, so "
                   "flags cannot recover them.")
    st.divider()

if not metrics_data:
    st.error("⚠ Metrics cache (`roc_cache.json`) is missing.")
    st.stop()

st.markdown('<div class="section-title">🔬 Shipped 2D model: detailed evaluation</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-subtitle">Everything below describes the 2D model only (the model the live MRI '
            f'Analysis page runs), on {metrics_data.get("n_subjects", 82)} held-out patients, '
            f'{metrics_data.get("slices_per_subject", 6)} sampled axial slices per patient. Not directly comparable '
            f'with the R* table above, which uses whole volumes.</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-label">Evaluated Subjects</div>
        <div class="stat-value">{subjects}</div>
        <div class="stat-delta">Held-out cohort</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚙️</div>
        <div class="stat-label">Training State</div>
        <div class="stat-value">Ep {epoch_val}</div>
        <div class="stat-delta" style="color: #0EA5E9;">2D Ensemble</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-label">2D mean Dice</div>
        <div class="stat-value">{mean_dice:.3f}</div>
        <div class="stat-delta">ET, TC, WT average · sampled slices</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📏</div>
        <div class="stat-label">WT Median HD95</div>
        <div class="stat-value">{wt_hd95:.2f} <span style="font-size:1rem;">mm</span></div>
        <div class="stat-delta" style="color: #64748B;">Lower is better</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  INTERACTIVE CHARTS (PLOTLY)
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 2D model: accuracy by region</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Comparing model precision, sensitivity, and surface distance across tumor sub-regions.</div>', unsafe_allow_html=True)

col_radar, col_bar = st.columns([1.2, 1])
regions_data = metrics_data["regions"]

with col_radar:
    categories = ['Dice Score', 'Sensitivity', 'Precision (Est.)']
    fig_radar = go.Figure()
    
    colors = {'ET': '#EF4444', 'TC': '#3B82F6', 'WT': '#10B981'}
    names = {'ET': 'Enhancing (ET)', 'TC': 'Core (TC)', 'WT': 'Whole (WT)'}
    
    for reg, color in colors.items():
        r_data = [
            regions_data[reg].get("dice", 0),
            regions_data[reg].get("sensitivity", 0),
            regions_data[reg].get("precision", 0)
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=r_data, theta=categories, fill='toself', name=names[reg], marker_color=color
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, title="Model Accuracy Radar",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

with col_bar:
    fig_bar = go.Figure()
    hd95_vals = [regions_data[reg]["hd95_median_mm"] for reg in ['ET', 'TC', 'WT']]
    
    fig_bar.add_trace(go.Bar(
        x=list(names.values()), y=hd95_vals, 
        marker_color=list(colors.values()),
        text=[f"{v:.1f} mm" for v in hd95_vals], textposition='auto'
    ))
    fig_bar.update_layout(
        title="Hausdorff Distance (HD95)",
        yaxis_title="Millimeters (mm)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

st.divider()

# ──────────────────────────────────────────────────────────
#  ADVANCED ANALYTICS (ROC & TABLE)
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 2D model: ROC curves and summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Receiver Operating Characteristic (ROC) curves and detailed statistical summary.</div>', unsafe_allow_html=True)

col_roc, col_table = st.columns([1.2, 1])

with col_roc:
    fig_roc = go.Figure()
    # Plotting the ROC curve using FPR and TPR from the JSON cache
    for reg, color in colors.items():
        fpr = regions_data[reg].get("fpr", [])
        tpr = regions_data[reg].get("tpr", [])
        auc = regions_data[reg].get("auc", 0.0)
        
        # Subsampling for performance if arrays are too large, but Plotly handles it well
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, 
            mode='lines', 
            name=f"{names[reg]} (AUC = {auc:.3f})",
            line=dict(color=color, width=2.5)
        ))
        
    # Add random guess diagonal line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode='lines', 
        name="Random Guess",
        line=dict(color='gray', width=1.5, dash='dash'),
        showlegend=False
    ))

    fig_roc.update_layout(
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=400, margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(x=0.5, y=0.1, bgcolor='rgba(255,255,255,0.7)')
    )
    st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})

with col_table:
    # Build a clean DataFrame for the metrics
    table_data = []
    for reg in ['ET', 'TC', 'WT']:
        table_data.append({
            "Region": names[reg],
            "Dice": f"{regions_data[reg].get('dice', 0):.3f}",
            "Sensitivity": f"{regions_data[reg].get('sensitivity', 0):.3f}",
            "Specificity": f"{regions_data[reg].get('specificity', 0):.4f}",
            "Precision": f"{regions_data[reg].get('precision', 0):.3f}",
            "AUC": f"{regions_data[reg].get('auc', 0):.3f}"
        })
    
    df = pd.DataFrame(table_data)
    
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    st.dataframe(
        df,
        column_config={
            "Region": st.column_config.TextColumn("Tumor Region", width="medium"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.info(f"💡 **Note:** 2D model only. Pixels of {metrics_data.get('n_slices', 492)} sampled axial slices "
            f"({metrics_data.get('slices_per_subject', 6)} per patient, up to {metrics_data.get('max_pixels_per_slice', 4000):,} pixels each) "
            f"from {metrics_data.get('n_subjects', 82)} held-out patients, pooled. Specificity is near 1 for any model "
            "because background dominates.")

# ──────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:30px; font-size:0.85rem; color:#94A3B8; border-top:1px solid #E2E8F0; margin-top: 40px; font-weight: 500;">
    NEUROPEDS AI &nbsp;·&nbsp; CLINICAL DECISION SUPPORT &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)