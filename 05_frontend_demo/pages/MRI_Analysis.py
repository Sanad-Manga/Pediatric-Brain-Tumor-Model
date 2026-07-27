import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="MRI Analysis Studio | NeuroFed AI", page_icon="🖥️", layout="wide")

st.markdown("""
    <div style="padding: 10px 0;">
        <h2 style="font-size: 2rem; color: #FFFFFF;">MRI Analysis Studio</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;">Multi-modality radiological workspace with 3D U-Net segmentation overlays.</p>
    </div>
""", unsafe_allow_html=True)

col_upload, col_viewer = st.columns([1, 2])

with col_upload:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 20px;">
            <h4 style="color: #FFFFFF; font-size: 1rem; margin-bottom: 12px;">📁 Volume Input</h4>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload NIfTI Volume (.nii / .nii.gz)", type=["nii", "gz"])
    
    modality = st.selectbox("Select MRI Modality", ["T1", "T1c (Contrast)", "T2", "FLAIR"])
    slice_idx = st.slider("Axial Slice Navigation", 0, 100, 45)
    
    st.markdown("""
            <hr style="border-color: rgba(56, 189, 248, 0.2); margin: 15px 0;">
            <h4 style="color: #FFFFFF; font-size: 1rem; margin-bottom: 8px;">🧠 Subregion Statistics</h4>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.6;">
                • <b style="color: #EF4444;">Enhancing Tumor (ET):</b> 24.2%<br>
                • <b style="color: #EAB308;">Peritumoral Edema (ED):</b> 48.6%<br>
                • <b style="color: #3B82F6;">Cystic Component (CC):</b> 18.1%<br>
                • <b style="color: #10B981;">Non-enhancing Tumor:</b> 9.1%
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_viewer:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 20px; text-align: center;">
            <h4 style="color: #FFFFFF; font-size: 1rem; margin-bottom: 12px; text-align: left;">🖥️ Interactive Viewer & Overlay</h4>
    """, unsafe_allow_html=True)
    
    # Generate mock 2D heatmap placeholder using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=np.random.rand(50, 50),
        colorscale='Viridis',
        showscale=False
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #38BDF8; font-family: monospace; background: rgba(5, 10, 31, 0.6); padding: 10px; border-radius: 8px;">
                <span>Status: Inference Ready (FP16)</span>
                <span>Model: Federated 3D U-Net + CORAL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)