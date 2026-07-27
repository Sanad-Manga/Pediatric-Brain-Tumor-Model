import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Federated Observatory | NeuroFed AI", page_icon="🌐", layout="wide")

st.markdown("""
    <div style="padding: 10px 0;">
        <h2 style="font-size: 2rem; color: #FFFFFF;">Federated Learning Observatory</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;">Live cross-institutional training telemetry and FedAvg convergence monitor.</p>
    </div>
""", unsafe_allow_html=True)

# Hospital Nodes Status Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 20px;">
            <div style="font-size: 0.75rem; color: #10B981; font-family: monospace;">● ACTIVE TRAINING NODE</div>
            <h4 style="color: #FFFFFF; margin: 8px 0 4px 0;">Hospital A</h4>
            <p style="color: #94A3B8; font-size: 0.8rem;">53 Patients | Local Dice: 91.2%</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 20px;">
            <div style="font-size: 0.75rem; color: #10B981; font-family: monospace;">● ACTIVE TRAINING NODE</div>
            <h4 style="color: #FFFFFF; margin: 8px 0 4px 0;">Hospital B</h4>
            <p style="color: #94A3B8; font-size: 0.8rem;">92 Patients | Local Dice: 92.8%</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 20px;">
            <div style="font-size: 0.75rem; color: #38BDF8; font-family: monospace;">● VALIDATION SITE</div>
            <h4 style="color: #FFFFFF; margin: 8px 0 4px 0;">Held-out Hospital</h4>
            <p style="color: #94A3B8; font-size: 0.8rem;">82 Patients | Test Dice: 92.4%</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Training Chart Simulation
st.markdown("""
    <div style="background: rgba(16, 24, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 24px;">
        <h3 style="color: #FFFFFF; font-size: 1.25rem; margin-bottom: 16px;">📈 Global Convergence Curve (FedAvg Rounds)</h3>
""", unsafe_allow_html=True)

rounds = list(range(1, 51))
dice_scores = [0.5 + 0.42 * (1 - 2.5**(-r/10)) for r in rounds]
df_chart = pd.DataFrame({"Round": rounds, "Global Dice Score": dice_scores})

fig = px.line(df_chart, x="Round", y="Global Dice Score", template="plotly_dark")
fig.update_traces(line=dict(color="#00F2FE", width=3))
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=20, b=20),
    height=350
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)