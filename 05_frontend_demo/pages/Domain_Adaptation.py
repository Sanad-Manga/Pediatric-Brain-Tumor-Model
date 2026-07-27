import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="Domain Adaptation Lab | NeuroFed AI", page_icon="🧬", layout="wide")

st.markdown("""
    <div style="padding: 10px 0;">
        <h2 style="font-size: 2rem; color: #FFFFFF;">CORAL Domain Adaptation Lab</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;">Visualizing feature distribution alignment and scanner bias reduction across institutions.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px;">
            <h3 style="color: #FFFFFF; font-size: 1.1rem; margin-bottom: 12px;">❌ Before Adaptation (High Scanner Bias)</h3>
    """, unsafe_allow_html=True)
    
    # Mock scatter before adaptation
    df_before = pd.DataFrame({
        'PC1': np.concatenate([np.random.normal(-2, 0.5, 50), np.random.normal(2, 0.5, 50)]),
        'PC2': np.concatenate([np.random.normal(-1, 0.5, 50), np.random.normal(1, 0.5, 50)]),
        'Hospital': ['Hospital A']*50 + ['Hospital B']*50
    })
    fig1 = px.scatter(df_before, x='PC1', y='PC2', color='Hospital', template='plotly_dark')
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("<p style='color: #EF4444; font-size: 0.8rem;'>Domain Gap Score: <b>0.684</b> (Significant Divergence)</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px;">
            <h3 style="color: #FFFFFF; font-size: 1.1rem; margin-bottom: 12px;">✅ After CORAL Alignment (Harmonized Space)</h3>
    """, unsafe_allow_html=True)
    
    # Mock scatter after adaptation
    df_after = pd.DataFrame({
        'PC1': np.concatenate([np.random.normal(0, 0.6, 50), np.random.normal(0.1, 0.6, 50)]),
        'PC2': np.concatenate([np.random.normal(0, 0.6, 50), np.random.normal(-0.1, 0.6, 50)]),
        'Hospital': ['Hospital A']*50 + ['Hospital B']*50
    })
    fig2 = px.scatter(df_after, x='PC1', y='PC2', color='Hospital', template='plotly_dark')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<p style='color: #10B981; font-size: 0.8rem;'>Domain Gap Score: <b>0.042</b> (Harmonized Covariance)</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)