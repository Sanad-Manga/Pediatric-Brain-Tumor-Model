import streamlit as st

def apply_custom_theme():
    """Applies the futuristic medical command center dark theme styling."""
    st.markdown("""
    <style>
        .stApp {
            background-color: #060913;
            color: #E6EDF3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .glass-card {
            background: rgba(16, 24, 46, 0.7);
            border: 1px solid rgba(0, 242, 254, 0.15);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 20px;
        }
        
        .glass-metric {
            background: rgba(10, 17, 40, 0.8);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            backdrop-filter: blur(8px);
        }

        h1, h2, h3 {
            color: #FFFFFF;
            font-weight: 700;
            letter-spacing: -0.025em;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 50%, #00F2FE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)