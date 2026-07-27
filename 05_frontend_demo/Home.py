import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroFed AI | Pediatric Brain Tumor Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Navigation Setup (This forces all pages to appear neatly in the sidebar)
pages = {
    "Platform": [
        st.Page("Home.py", title="AI Command Center", icon="⚡"),
    ],
    "Clinical & Analysis": [
        st.Page("pages/MRI_Analysis.py", title="MRI Analysis Studio", icon="🖥️"),
        st.Page("pages/Segmentation_Report.py", title="Segmentation Report", icon="📋"),
        st.Page("pages/Clinical_View.py", title="Clinical Explainability", icon="🩺"),
    ],
    "Federated & AI Core": [
        st.Page("pages/Federated_Monitor.py", title="Federated Observatory", icon="🌐"),
        st.Page("pages/Domain_Adaptation.py", title="Domain Adaptation Lab", icon="🧬"),
        st.Page("pages/Model_Intelligence.py", title="Model Intelligence", icon="🧠"),
    ],
    "System": [
        st.Page("pages/About.py", title="About Research", icon="📚"),
    ]
}

pg = st.navigation(pages, position="sidebar")

# 3. Global UI Theme Injection
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #030712 0%, #0B132B 50%, #060913 100%);
        color: #F3F4F6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.2);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 40px 30px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        margin-bottom: 30px;
        text-align: center;
    }
    
    .content-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    h1, h2, h3 { color: #FFFFFF; font-weight: 700; }
    
    .gradient-title {
        background: linear-gradient(135deg, #FFFFFF 30%, #38BDF8 70%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 30px;
        font-size: 0.75rem;
        font-family: monospace;
        color: #38BDF8;
        margin-bottom: 16px;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# 4. Run the navigation (This renders the sidebar links automatically)
pg.run()