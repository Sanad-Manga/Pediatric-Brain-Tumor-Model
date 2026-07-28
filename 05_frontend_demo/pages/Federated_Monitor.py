import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Federated Observatory | NeuroFed AI", page_icon="🌐", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 80% 50%, rgba(52,211,153,0.09), transparent 50%),
                linear-gradient(135deg,#0B1628,#0F172A);
    border:1px solid rgba(52,211,153,0.2); border-radius:20px; padding:40px 44px; margin-bottom:28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#fff 40%,#34D399 80%,#38BDF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#94A3B8; font-size:0.92rem; }
.panel { background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:28px; margin-bottom:20px; }
.panel-title { font-size:1rem; font-weight:700; color:#F3F4F6; margin-bottom:18px; }
.node-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:20px; }
.node-card {
    background:rgba(10,15,35,0.65); border-radius:14px; padding:18px;
    border: 1px solid rgba(255,255,255,0.07);
    transition: border-color .2s ease, transform .2s ease;
}
.node-card:hover { transform:translateY(-2px); }
.node-card.active   { border-color:rgba(52,211,153,0.3); }
.node-card.syncing  { border-color:rgba(56,189,248,0.3); }
.node-card.holdout  { border-color:rgba(129,140,248,0.3); }
.node-status { font-size:0.68rem; font-family:'JetBrains Mono',monospace;
    letter-spacing:0.05em; text-transform:uppercase; display:flex; align-items:center; gap:6px; margin-bottom:8px; }
.node-status.active  { color:#34D399; }
.node-status.syncing { color:#38BDF8; }
.node-status.holdout { color:#818CF8; }
.dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.dot.active  { background:#34D399; animation:pulseDot 1.8s infinite; }
.dot.syncing { background:#38BDF8; }
.dot.holdout { background:#818CF8; }
@keyframes pulseDot {
    0%  { box-shadow:0 0 0 0 rgba(52,211,153,.6); }
    70% { box-shadow:0 0 0 7px rgba(52,211,153,0); }
    100%{ box-shadow:0 0 0 0 rgba(52,211,153,0); }
}
.node-name  { font-size:0.88rem; font-weight:700; color:#F3F4F6; margin-bottom:6px; }
.node-stat  { font-size:0.75rem; color:#64748B; margin-bottom:3px; }
.node-dice  { font-size:0.8rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.node-card.active  .node-dice { color:#34D399; }
.node-card.syncing .node-dice { color:#38BDF8; }
.node-card.holdout .node-dice { color:#818CF8; }
.fed-stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px; }
.fed-stat { background:rgba(10,15,35,0.65); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:16px; text-align:center; }
.fed-stat-val { font-size:1.6rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#fff,#38BDF8); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; background-clip:text; }
.fed-stat-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-top:4px; }
</style>

<div class="page-hero">
    <div class="page-title">Federated Learning Observatory</div>
    <div class="page-sub">Live cross-institutional training telemetry and FedAvg convergence monitor across 5 hospital nodes.</div>
</div>
""", unsafe_allow_html=True)

# ── Federation summary stats
st.markdown("""
<div class="fed-stats-row">
    <div class="fed-stat"><div class="fed-stat-val">50</div><div class="fed-stat-label">Training Rounds</div></div>
    <div class="fed-stat"><div class="fed-stat-val">5</div><div class="fed-stat-label">Active Nodes</div></div>
    <div class="fed-stat"><div class="fed-stat-val">257</div><div class="fed-stat-label">Total Subjects</div></div>
    <div class="fed-stat"><div class="fed-stat-val" style="background:linear-gradient(135deg,#fff,#34D399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">92.4%</div><div class="fed-stat-label">Global Dice</div></div>
</div>
""", unsafe_allow_html=True)

# ── Node cards
st.markdown("""
<div class="panel">
    <div class="panel-title">🏥 Hospital Node Status</div>
    <div class="node-grid">
        <div class="node-card active">
            <div class="node-status active"><span class="dot active"></span>Training</div>
            <div class="node-name">Hospital A</div>
            <div class="node-stat">53 patients · Siemens 3T</div>
            <div class="node-dice">Dice 91.2%</div>
        </div>
        <div class="node-card active">
            <div class="node-status active"><span class="dot active"></span>Training</div>
            <div class="node-name">Hospital B</div>
            <div class="node-stat">92 patients · GE 1.5T</div>
            <div class="node-dice">Dice 92.8%</div>
        </div>
        <div class="node-card active">
            <div class="node-status active"><span class="dot active"></span>Training</div>
            <div class="node-name">Hospital C</div>
            <div class="node-stat">48 patients · Philips 3T</div>
            <div class="node-dice">Dice 90.5%</div>
        </div>
        <div class="node-card syncing">
            <div class="node-status syncing"><span class="dot syncing"></span>Syncing</div>
            <div class="node-name">Hospital D</div>
            <div class="node-stat">38 patients · Siemens 1.5T</div>
            <div class="node-dice">Dice 89.9%</div>
        </div>
        <div class="node-card holdout">
            <div class="node-status holdout"><span class="dot holdout"></span>Validation</div>
            <div class="node-name">Held-out Site</div>
            <div class="node-stat">82 patients · Mixed</div>
            <div class="node-dice">Dice 92.4%</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Convergence chart
st.markdown("""
<div class="panel">
    <div class="panel-title">📈 Global Convergence — FedAvg Rounds</div>
""", unsafe_allow_html=True)

rounds = list(range(1, 51))
global_dice  = [0.50 + 0.424 * (1 - 2.5**(-r/10)) for r in rounds]
site_a_dice  = [g - 0.01 + 0.008*((r%5)/5) for r,g in zip(rounds,global_dice)]
site_b_dice  = [g + 0.012 - 0.005*((r%7)/7) for r,g in zip(rounds,global_dice)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=rounds, y=site_a_dice, name='Hospital A', line=dict(color='#38BDF8', width=1.5, dash='dot')))
fig.add_trace(go.Scatter(x=rounds, y=site_b_dice, name='Hospital B', line=dict(color='#818CF8', width=1.5, dash='dot')))
fig.add_trace(go.Scatter(x=rounds, y=global_dice, name='Global FedAvg', line=dict(color='#34D399', width=3),
                          fill='tozeroy', fillcolor='rgba(52,211,153,0.05)'))
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,30,0.6)',
    height=340, margin=dict(l=10,r=10,t=10,b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#64748B', tickformat='.2f'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#64748B', title='Round'),
    font=dict(color='#94A3B8')
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Per-node dice comparison bar
st.markdown("""
<div class="panel">
    <div class="panel-title">📊 Per-Site Dice Score Comparison</div>
""", unsafe_allow_html=True)

sites = ['Hospital A','Hospital B','Hospital C','Hospital D','Held-out']
dices = [91.2, 92.8, 90.5, 89.9, 92.4]
colors = ['#38BDF8','#818CF8','#38BDF8','#818CF8','#34D399']

fig2 = go.Figure(go.Bar(x=sites, y=dices, marker_color=colors,
                         text=[f'{d}%' for d in dices], textposition='outside',
                         textfont=dict(color='#94A3B8', size=12)))
fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,30,0.6)',
    height=260, margin=dict(l=10,r=10,t=20,b=10),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#64748B', range=[85,96]),
    xaxis=dict(color='#64748B'), font=dict(color='#94A3B8')
)
st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)