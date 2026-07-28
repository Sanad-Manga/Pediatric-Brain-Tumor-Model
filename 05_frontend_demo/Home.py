import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="NeuroFed AI | Pediatric Brain Tumor Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Navigation Setup (This forces all pages to appear neatly in the sidebar)
# NOTE: the landing page lives in pages/Command_Center.py, not here. This file is the
# entry script — registering it as a page too would make pg.run() re-execute Home.py,
# which recurses into pg.run() again (RecursionError).
pages = {
    "Platform": [
        st.Page("pages/Dashboard.py", title="AI Command Center", icon="⚡"),
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


# CSS FIRST
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color:#030712;
}

.stApp {
    background:
    linear-gradient(
        135deg,
        #030712 0%,
        #0B132B 50%,
        #060913 100%
    );
}
</style>
""", unsafe_allow_html=True)


# NAVIGATION LAST
pg = st.navigation(
    pages,
    position="sidebar"
)

# 4. Run the navigation (This renders the sidebar links automatically)
pg.run()