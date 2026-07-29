import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="Domain Adaptation Lab | NeuroFed AI", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 20% 50%, rgba(129,140,248,0.10), transparent 50%),
                linear-gradient(135deg,#F8FAFC,#E0F2FE);
    border:1px solid rgba(129,140,248,0.25); border-radius:20px; padding:40px 44px; margin-bottom:28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#0F172A 30%,#4F46E5 80%,#0284C7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#64748B; font-size:0.92rem; }
.panel { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:28px; margin-bottom:20px; }
.panel-title { font-size:1rem; font-weight:700; color:#0F172A; margin-bottom:18px; }
.gap-row { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }
.gap-card { background:#F8FAFC; border-radius:14px; padding:20px; }
.gap-card-head { font-size:0.85rem; font-weight:700; margin-bottom:6px; }
.gap-card-head.bad  { color:#DC2626; }
.gap-card-head.good { color:#059669; }
.gap-score { font-size:0.8rem; font-family:'JetBrains Mono',monospace; margin-top:6px; }
.coral-steps { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
.coral-step {
    background:#F8FAFC; border:1px solid rgba(129,140,248,0.15);
    border-radius:12px; padding:18px; text-align:center;
    transition: border-color .2s ease, transform .2s ease;
}
.coral-step:hover { border-color:rgba(129,140,248,0.4); transform:translateY(-2px); }
.coral-step-num { font-size:0.7rem; font-family:'JetBrains Mono',monospace;
    color:#4F46E5; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px; }
.coral-step-icon { font-size:1.5rem; margin-bottom:8px; }
.coral-step-name { font-size:0.82rem; font-weight:700; color:#0F172A; margin-bottom:4px; }
.coral-step-desc { font-size:0.72rem; color:#64748B; line-height:1.4; }
</style>

<div class="page-hero">
    <div class="page-title">CORAL Domain Adaptation Lab</div>
    <div class="page-sub">Visualizing feature distribution alignment and scanner bias reduction across hospital institutions.</div>
</div>
""", unsafe_allow_html=True)

# ── Gap score cards + scatter plots
st.markdown('<div class="gap-row">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

np.random.seed(42)
with col1:
    st.markdown("""
    <div class="panel" style="margin-bottom:0;">
        <div class="panel-title">❌ Before Adaptation — High Scanner Bias</div>
    """, unsafe_allow_html=True)
    df_before = pd.DataFrame({
        'PC1': np.concatenate([np.random.normal(-2, 0.5, 60), np.random.normal(2, 0.5, 60)]),
        'PC2': np.concatenate([np.random.normal(-1, 0.5, 60), np.random.normal(1, 0.5, 60)]),
        'Site': ['Hospital A']*60 + ['Hospital B']*60
    })
    fig1 = px.scatter(df_before, x='PC1', y='PC2', color='Site',
                      color_discrete_map={'Hospital A':'#0284C7','Hospital B':'#4F46E5'},
                      template='plotly_dark', labels={'PC1':'PCA Component 1','PC2':'PCA Component 2'})
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F8FAFC',
                       height=300, margin=dict(l=10,r=10,t=10,b=10),
                       legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#64748B')))
    fig1.update_xaxes(gridcolor='#E2E8F0', zerolinecolor='#E2E8F0')
    fig1.update_yaxes(gridcolor='#E2E8F0', zerolinecolor='#E2E8F0')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('<div style="font-size:0.8rem;font-family:monospace;color:#DC2626;margin-top:4px;">Domain Gap Score: <b>0.684</b> — Significant Divergence</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="panel" style="margin-bottom:0;">
        <div class="panel-title">✅ After CORAL Alignment — Harmonized Space</div>
    """, unsafe_allow_html=True)
    df_after = pd.DataFrame({
        'PC1': np.concatenate([np.random.normal(0, 0.55, 60), np.random.normal(0.08, 0.55, 60)]),
        'PC2': np.concatenate([np.random.normal(0, 0.55, 60), np.random.normal(-0.08, 0.55, 60)]),
        'Site': ['Hospital A']*60 + ['Hospital B']*60
    })
    fig2 = px.scatter(df_after, x='PC1', y='PC2', color='Site',
                      color_discrete_map={'Hospital A':'#0284C7','Hospital B':'#4F46E5'},
                      template='plotly_dark', labels={'PC1':'PCA Component 1','PC2':'PCA Component 2'})
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F8FAFC',
                       height=300, margin=dict(l=10,r=10,t=10,b=10),
                       legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#64748B')))
    fig2.update_xaxes(gridcolor='#E2E8F0', zerolinecolor='#E2E8F0')
    fig2.update_yaxes(gridcolor='#E2E8F0', zerolinecolor='#E2E8F0')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div style="font-size:0.8rem;font-family:monospace;color:#059669;margin-top:4px;">Domain Gap Score: <b>0.042</b> — Harmonized Covariance</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Domain gap over adaptation steps (bar chart)
st.markdown("""
<div class="panel">
    <div class="panel-title">📉 Domain Gap Reduction Progress</div>
""", unsafe_allow_html=True)

steps = ['Baseline', 'Normalization', 'PCA Align', 'CORAL Step 1', 'CORAL Step 2', 'Final']
gaps  = [0.684, 0.520, 0.380, 0.210, 0.085, 0.042]
colors = ['#DC2626','#FB923C','#B45309','#4F46E5','#0284C7','#059669']

fig3 = go.Figure(go.Bar(x=steps, y=gaps, marker_color=colors,
                         text=[f'{g:.3f}' for g in gaps], textposition='outside',
                         textfont=dict(color='#64748B', size=11)))
fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F8FAFC',
                   height=280, margin=dict(l=10,r=10,t=20,b=10),
                   yaxis=dict(gridcolor='#E2E8F0', color='#64748B'),
                   xaxis=dict(color='#64748B'), font=dict(color='#64748B'))
st.plotly_chart(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── CORAL method steps
st.markdown("""
<div class="panel">
    <div class="panel-title">🧬 CORAL Alignment Pipeline</div>
    <div class="coral-steps">
        <div class="coral-step">
            <div class="coral-step-num">Step 01</div>
            <div class="coral-step-icon">📥</div>
            <div class="coral-step-name">Feature Extraction</div>
            <div class="coral-step-desc">Extract deep feature maps from source & target domain encoders.</div>
        </div>
        <div class="coral-step">
            <div class="coral-step-num">Step 02</div>
            <div class="coral-step-icon">📐</div>
            <div class="coral-step-name">Covariance Estimation</div>
            <div class="coral-step-desc">Compute second-order statistics matrices C_S and C_T per domain.</div>
        </div>
        <div class="coral-step">
            <div class="coral-step-num">Step 03</div>
            <div class="coral-step-icon">🔄</div>
            <div class="coral-step-name">Whitening Transform</div>
            <div class="coral-step-desc">Decorrelate source features via C_S^{-1/2} whitening operation.</div>
        </div>
        <div class="coral-step">
            <div class="coral-step-num">Step 04</div>
            <div class="coral-step-icon">✅</div>
            <div class="coral-step-name">Coloring Projection</div>
            <div class="coral-step-desc">Re-color with C_T^{1/2} to match target distribution — gap closed.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)