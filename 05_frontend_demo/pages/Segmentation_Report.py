import streamlit as st

st.set_page_config(page_title="About Suite", page_icon="ℹ️", layout="wide")
st.markdown("<style>.stApp { background-color: #050A1F; color: #E2E8F0; } h1, h2, h3 { color: #64FFDA !important; } #MainMenu, header, footer {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("ℹ️ Technical Specifications & Architecture")
st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div style='background: rgba(16,24,46,0.7); padding: 30px; border-radius: 20px; border: 1px solid rgba(100,255,218,0.2); height: 100%;'>
        <h3 style='color: #64FFDA; margin-top: 0;'>System Contracts</h3>
        <ul style='color: #8892B0; font-size: 16px; line-height: 2;'>
            <li><b>Target:</b> 3D Segmentation of Pediatric Brain Tumors</li>
            <li><b>Dataset:</b> BraTS-PEDs (257 verified subjects)</li>
            <li><b>Data Constraint:</b> Fully 3D Volumes only (No 2D planar ops)</li>
            <li><b>Resolution Target:</b> $96 \\times 96 \\times 96$ standard geometry</li>
            <li><b>Precision Layer:</b> FP16 Mixed Precision for memory scaling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style='background: rgba(16,24,46,0.7); padding: 30px; border-radius: 20px; border: 1px solid rgba(100,255,218,0.2); height: 100%;'>
        <h3 style='color: #64FFDA; margin-top: 0;'>Modular Implementation</h3>
        <ul style='color: #8892B0; font-size: 16px; line-height: 2;'>
            <li><b style='color:#E2E8F0;'>01_model_federated:</b> 3D U-Net & FedAvg Strategy</li>
            <li><b style='color:#E2E8F0;'>02_domain_adaptation:</b> CORAL Implementation</li>
            <li><b style='color:#E2E8F0;'>03_augmentation_eval:</b> Spatial Mixup & Volume scaling</li>
            <li><b style='color:#E2E8F0;'>04_clinical_bio:</b> Medical validation & Narrative</li>
            <li><b style='color:#64FFDA;'>05_frontend_demo:</b> Clinical Interface Suite (Current)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)