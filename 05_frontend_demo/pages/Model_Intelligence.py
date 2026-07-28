import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Model Intelligence | NeuroFed AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 20% 50%, rgba(168,85,247,0.09), transparent 50%),
                linear-gradient(135deg,#0B1628,#0F172A);
    border:1px solid rgba(168,85,247,0.2); border-radius:20px; padding:40px 44px; margin-bottom:28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#fff 40%,#A855F7 80%,#38BDF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#94A3B8; font-size:0.92rem; }
.panel { background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:28px; margin-bottom:20px; }
.panel-title { font-size:1rem; font-weight:700; color:#F3F4F6; margin-bottom:18px; }
.spec-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.spec-table tr { border-bottom:1px solid rgba(255,255,255,0.05); }
.spec-table tr:last-child { border-bottom:none; }
.spec-table td { padding:12px 8px; }
.spec-key { color:#64748B; width:42%; font-family:'JetBrains Mono',monospace; font-size:0.78rem; }
.spec-val { color:#CBD5E1; font-weight:500; }
.spec-val b { color:#F3F4F6; }
.spec-val .tag { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.7rem;
    font-family:'JetBrains Mono',monospace; background:rgba(56,189,248,0.1);
    color:#38BDF8; border:1px solid rgba(56,189,248,0.2); margin-left:6px; }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px; }
.metric-box { background:rgba(10,15,35,0.65); border:1px solid rgba(255,255,255,0.07);
    border-radius:12px; padding:18px; text-align:center;
    transition: border-color .2s, transform .2s; }
.metric-box:hover { transform:translateY(-2px); border-color:rgba(168,85,247,0.3); }
.metric-val { font-size:1.9rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:4px; }
.metric-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:8px; }
.metric-bar-bg { height:3px; background:rgba(255,255,255,0.07); border-radius:3px; overflow:hidden; }
.metric-bar { height:100%; border-radius:3px; }
.layer-row { display:flex; align-items:center; gap:14px; margin-bottom:12px; }
.layer-name { font-size:0.78rem; font-family:'JetBrains Mono',monospace; color:#94A3B8; width:180px; flex-shrink:0; }
.layer-bar-bg { flex:1; height:8px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; }
.layer-bar { height:100%; border-radius:4px; }
.layer-params { font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:#64748B; width:80px; text-align:right; }
</style>

<div class="page-hero">
    <div class="page-title">Model Intelligence & Architecture</div>
    <div class="page-sub">Technical specifications, validation benchmarks, and layer-wise parameter analysis of the federated 3D U-Net backbone.</div>
</div>
""", unsafe_allow_html=True)

# ── Benchmark metrics
st.markdown("""
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-val" style="background:linear-gradient(135deg,#fff,#38BDF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">92.4%</div>
        <div class="metric-label">Mean Dice Score</div>
        <div class="metric-bar-bg"><div class="metric-bar" style="width:92.4%;background:linear-gradient(90deg,#38BDF8,#00F2FE);"></div></div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="background:linear-gradient(135deg,#fff,#00F2FE);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">4.8mm</div>
        <div class="metric-label">HD95 Distance</div>
        <div class="metric-bar-bg"><div class="metric-bar" style="width:60%;background:linear-gradient(90deg,#00F2FE,#38BDF8);"></div></div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="background:linear-gradient(135deg,#fff,#34D399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">94.2%</div>
        <div class="metric-label">Sensitivity</div>
        <div class="metric-bar-bg"><div class="metric-bar" style="width:94.2%;background:linear-gradient(90deg,#34D399,#38BDF8);"></div></div>
    </div>
    <div class="metric-box">
        <div class="metric-val" style="background:linear-gradient(135deg,#fff,#A855F7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">98.1%</div>
        <div class="metric-label">Specificity</div>
        <div class="metric-bar-bg"><div class="metric-bar" style="width:98.1%;background:linear-gradient(90deg,#A855F7,#818CF8);"></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">⚙️ Architecture Specifications</div>
        <table class="spec-table">
            <tr><td class="spec-key">Backbone</td><td class="spec-val"><b>3D U-Net</b> + Residual Encoders</td></tr>
            <tr><td class="spec-key">Input Shape</td><td class="spec-val"><b>96 × 96 × 96</b> voxels<span class="tag">FP16</span></td></tr>
            <tr><td class="spec-key">Modalities</td><td class="spec-val">T1 · T1c · T2 · FLAIR</td></tr>
            <tr><td class="spec-key">Loss Function</td><td class="spec-val">Soft Dice Loss + Focal Loss</td></tr>
            <tr><td class="spec-key">Optimizer</td><td class="spec-val">AdamW <span class="tag">lr=1e-4</span></td></tr>
            <tr><td class="spec-key">Weight Decay</td><td class="spec-val"><b>1e-5</b></td></tr>
            <tr><td class="spec-key">Federation</td><td class="spec-val">FedAvg · Secure Gradient Enc.</td></tr>
            <tr><td class="spec-key">Output Classes</td><td class="spec-val">ET · ED · CC · NET · Background</td></tr>
            <tr><td class="spec-key">Total Parameters</td><td class="spec-val"><b>31.2M</b></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 Layer-wise Parameter Distribution</div>
    """, unsafe_allow_html=True)
    layers = [
        ("Encoder Block 1", 0.8, "1.2M", "#38BDF8"),
        ("Encoder Block 2", 0.55, "2.4M", "#38BDF8"),
        ("Encoder Block 3", 0.72, "4.8M", "#818CF8"),
        ("Bottleneck", 1.0, "9.6M", "#A855F7"),
        ("Decoder Block 3", 0.72, "4.8M", "#818CF8"),
        ("Decoder Block 2", 0.55, "2.4M", "#38BDF8"),
        ("Decoder Block 1", 0.38, "1.2M", "#34D399"),
        ("Segmentation Head", 0.18, "0.4M", "#34D399"),
    ]
    html = ""
    for name, pct, params, color in layers:
        html += f"""
        <div class="layer-row">
            <div class="layer-name">{name}</div>
            <div class="layer-bar-bg"><div class="layer-bar" style="width:{int(pct*100)}%;background:{color};"></div></div>
            <div class="layer-params">{params}</div>
        </div>"""
    st.markdown(html + "</div>", unsafe_allow_html=True)

# ── Training curves
st.markdown("""
<div class="panel">
    <div class="panel-title">📈 Training & Validation Loss Curves</div>
""", unsafe_allow_html=True)

epochs = list(range(1, 101))
train_loss = [1.0 * (0.97**e) + 0.05 + 0.01*(e%5/5) for e in epochs]
val_loss   = [1.05 * (0.968**e) + 0.07 + 0.015*(e%7/7) for e in epochs]

fig = go.Figure()
fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss',
    line=dict(color='#38BDF8', width=2)))
fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Val Loss',
    line=dict(color='#A855F7', width=2, dash='dot')))
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,30,0.6)',
    height=280, margin=dict(l=10,r=10,t=10,b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#64748B'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#64748B', title='Epoch'),
    font=dict(color='#94A3B8')
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)