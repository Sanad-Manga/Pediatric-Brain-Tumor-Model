import streamlit as st
from datetime import datetime

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="NeuroPeds AI | Clinical Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "NeuroPeds AI — Clinical Decision Support for Pediatric Brain Tumor Segmentation."
    }
)

# ==========================================================
# 2. NAVIGATION STRUCTURE (Original Icons Restored)
# ==========================================================
pages = {
    "Platform": [
        st.Page("pages/Dashboard.py", title="Clinical Dashboard", icon="⚡"),
    ],
    "Clinical & Analysis": [
        st.Page("pages/MRI_Analysis.py", title="MRI Analysis Studio", icon="🖥️"),
        st.Page("pages/Model_Comparison.py", title="Model Comparison", icon="🔍"),
        st.Page("pages/Segmentation_Report.py", title="Segmentation Report", icon="📋"),
        st.Page("pages/Clinical_View.py", title="Tumor Subregion Guide", icon="🩻"),
    ],
    "System": [
        st.Page("pages/About.py", title="About NeuroPeds AI", icon="📚"),
    ]
}

pg = st.navigation(pages, position="sidebar")

# ==========================================================
# 3. GLOBAL THEME — MODERN SAAS UI
# ==========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;700;800&display=swap" rel="stylesheet">

<style>
    /* Force Colorful Emojis everywhere instead of flat black */
    * {
        font-family: 'Inter', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif;
    }
    
    :root {
        --bg-base: #F8FAFC;
        --sidebar-bg: #FFFFFF;
        --accent: #0EA5E9;
        --text-main: #0F172A;
        --text-muted: #64748B;
        --border-color: #E2E8F0;
    }

    /* Background and main app area */
    .stApp {
        background-color: var(--bg-base);
        color: var(--text-main);
    }

    /* Sleek Sidebar Design */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.015) !important;
    }
    
    /* Navigation Links Styling */
    [data-testid="stSidebar"] .stPageLink {
        border-radius: 10px !important;
        margin-bottom: 4px;
        transition: all 0.2s ease !important;
        color: var(--text-main) !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
    }
    
    [data-testid="stSidebar"] .stPageLink:hover {
        background-color: #F0F9FF !important;
        color: var(--accent) !important;
    }

    /* Branding Section */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 8px 4px 24px 4px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 20px;
    }
    .brand-emoji {
        font-size: 2.2rem;
        filter: drop-shadow(0 2px 4px rgba(14, 165, 233, 0.2));
    }
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.3rem;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        color: var(--accent);
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Modern Status Card */
    .status-card {
        background-color: #F8FAFC;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin-top: 24px;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
    .status-row:last-child { margin-bottom: 0; }
    .status-label {
        color: var(--text-muted);
        font-weight: 500;
    }
    .status-value {
        font-weight: 600;
        color: #0F172A;
    }
    
    /* Live Dot Animation */
    .live-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #10B981;
        font-weight: 600;
        background: #D1FAE5;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    .live-dot {
        width: 6px;
        height: 6px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. SIDEBAR BRANDING & FOOTER
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-emoji">🧠</div>
        <div>
            <div class="brand-title">NeuroPeds AI</div>
            <div class="brand-subtitle">Clinical Support</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-row">
            <span class="status-label">System Status</span>
            <span class="live-badge"><div class="live-dot"></div>Online</span>
        </div>
        <div class="status-row">
            <span class="status-label">Analysis Engine</span>
            <span class="status-value" style="color: #0EA5E9;">Ready</span>
        </div>
        <div class="status-row" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #E2E8F0;">
            <span class="status-label" style="font-size: 0.75rem;">Session Time</span>
            <span style="color: #94A3B8; font-family: monospace; font-size: 0.75rem;">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 5. RUN NAVIGATION
# ==========================================================
pg.run()