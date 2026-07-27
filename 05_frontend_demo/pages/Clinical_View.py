import streamlit as st

st.set_page_config(page_title="Clinical Explainability | NeuroFed AI", page_icon="🩺", layout="wide")

st.markdown("""
    <div style="padding: 10px 0;">
        <h2 style="font-size: 2rem; color: #FFFFFF;">Clinical Explainability & Tumor Class Guide</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;">Translating deep learning feature maps into actionable pediatric neuro-oncology insights.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #FFFFFF; font-size: 1.25rem; margin-bottom: 16px;">🏷️ BraTS-PEDs Tumor Subregions</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="background: rgba(5, 10, 31, 0.6); padding: 16px; border-radius: 10px; border-left: 4px solid #EF4444;">
                <h4 style="color: #FFFFFF; margin-bottom: 6px;">1. Enhancing Tumor (ET)</h4>
                <p style="color: #94A3B8; font-size: 0.85rem;">Represents active tumor regions with blood-brain barrier breakdown, highlighted clearly on T1-contrast sequences.</p>
            </div>
            <div style="background: rgba(5, 10, 31, 0.6); padding: 16px; border-radius: 10px; border-left: 4px solid #EAB308;">
                <h4 style="color: #FFFFFF; margin-bottom: 6px;">2. Peritumoral Edema (ED)</h4>
                <p style="color: #94A3B8; font-size: 0.85rem;">Surrounding tissue swelling and vasogenic edema identified by high signal intensity on FLAIR scans.</p>
            </div>
            <div style="background: rgba(5, 10, 31, 0.6); padding: 16px; border-radius: 10px; border-left: 4px solid #3B82F6;">
                <h4 style="color: #FFFFFF; margin-bottom: 6px;">3. Cystic Component (CC)</h4>
                <p style="color: #94A3B8; font-size: 0.85rem;">Fluid-filled necrotic or cystic regions within the tumor core structure.</p>
            </div>
            <div style="background: rgba(5, 10, 31, 0.6); padding: 16px; border-radius: 10px; border-left: 4px solid #10B981;">
                <h4 style="color: #FFFFFF; margin-bottom: 6px;">4. Non-enhancing Tumor (NET)</h4>
                <p style="color: #94A3B8; font-size: 0.85rem;">Solid tumor core regions without active contrast enhancement.</p>
            </div>
        </div>
    </div>
    
    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 16px; font-size: 0.85rem; color: #FCA5A5;">
        <b>⚠️ Medical Disclaimer:</b> NeuroFed AI is a research-grade decision support platform designed to assist clinicians and researchers. It does not replace independent clinical diagnosis or professional medical judgment.
    </div>
""", unsafe_allow_html=True)