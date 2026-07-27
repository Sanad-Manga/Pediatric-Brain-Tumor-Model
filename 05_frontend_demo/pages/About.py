import streamlit as st

st.set_page_config(page_title="About Research | NeuroFed AI", page_icon="📚", layout="wide")

st.markdown("""
    <div style="padding: 10px 0;">
        <h2 style="font-size: 2rem; color: #FFFFFF;">About BraTS-PEDs Research & Platform</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;"> Advancing decentralized pediatric neuro-oncology through privacy-first AI.</p>
    </div>
    
    <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px; margin-top: 20px;">
        <h3 style="color: #FFFFFF; font-size: 1.25rem; margin-bottom: 14px;">🎯 Research Objectives & Motivation</h3>
        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.8; margin-bottom: 16px;">
            Pediatric brain tumors present unique morphological variations and diffuse margins compared to adult neuro-oncology cases. Training robust deep learning models requires large multi-institutional datasets; however, patient privacy regulations and hospital data silos strictly limit raw data sharing.
        </p>
        <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.8;">
            <b>NeuroFed AI</b> solves this challenge by combining <b>Federated Learning (FedAvg)</b> with <b>CORAL Domain Adaptation</b>, allowing hospitals to collaboratively train state-of-the-art 3D U-Net segmentation models without ever transferring sensitive patient MRI scans across institutional boundaries.
        </p>
    </div>
""", unsafe_allow_html=True)