import streamlit as st

st.set_page_config(page_title="Model Intelligence | NeuroFed AI", page_icon="🧠", layout="wide")

st.markdown("""
    <div style="padding: 10px 0;">
        <h2 style="font-size: 2rem; color: #FFFFFF;">Model Intelligence & Architecture</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;">Technical specifications of the federated 3D U-Net backbone.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px;">
            <h3 style="color: #FFFFFF; font-size: 1.25rem; margin-bottom: 16px;">⚙️ Architecture Specifications</h3>
            <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.8; padding-left: 20px;">
                <li><b>Backbone:</b> 3D U-Net with Residual Encoders</li>
                <li><b>Input Dimensions:</b> 96 × 96 × 96 voxels (FP16)</li>
                <li><b>Loss Function:</b> Combined Soft Dice Loss & Focal Loss</li>
                <li><b>Optimization:</b> AdamW (lr=1e-4, weight_decay=1e-5)</li>
                <li><b>Federation Protocol:</b> FedAvg with Secure Gradient Encryption</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px;">
            <h3 style="color: #FFFFFF; font-size: 1.25rem; margin-bottom: 16px;">📊 Validation Benchmark Metrics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div style="background: rgba(5, 10, 31, 0.6); padding: 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #38BDF8;">92.4%</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Mean Dice Score</div>
                </div>
                <div style="background: rgba(5, 10, 31, 0.6); padding: 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #00F2FE;">4.8 mm</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">HD95 Distance</div>
                </div>
                <div style="background: rgba(5, 10, 31, 0.6); padding: 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #10B981;">94.2%</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Sensitivity</div>
                </div>
                <div style="background: rgba(5, 10, 31, 0.6); padding: 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #A855F7;">98.1%</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Specificity</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)