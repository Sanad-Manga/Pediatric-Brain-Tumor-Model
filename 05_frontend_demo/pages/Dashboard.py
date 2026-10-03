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

@st.cache_data
def load_metrics():
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

metrics_data = load_metrics()

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
        Interactive visualization of the R* model ensemble performance evaluated on held-out BraTS-PEDs 2024 subjects. Data is bound dynamically to the validation cache.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  DYNAMIC STAT CARDS
# ──────────────────────────────────────────────────────────
if not metrics_data:
    st.error("⚠ Metrics cache (`roc_cache.json`) is missing.")
    st.stop()

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
        <div class="stat-label">Global Mean Dice</div>
        <div class="stat-value">{mean_dice:.3f}</div>
        <div class="stat-delta">ET, TC, WT Average</div>
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
st.markdown('<div class="section-title">📊 Regional Segmentation Profiling</div>', unsafe_allow_html=True)
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
st.markdown('<div class="section-title">📈 Clinical Validation Metrics</div>', unsafe_allow_html=True)
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
    
    st.info("💡 **Note:** Metrics are computed over all non-background pixels across 82 held-out subjects. Specificity appears artificially high due to class imbalance (background dominance).")

# ──────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:30px; font-size:0.85rem; color:#94A3B8; border-top:1px solid #E2E8F0; margin-top: 40px; font-weight: 500;">
    NEUROPEDS AI &nbsp;·&nbsp; CLINICAL DECISION SUPPORT &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)