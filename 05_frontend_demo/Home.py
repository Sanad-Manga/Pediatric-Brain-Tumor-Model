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
        "Get Help": None,
        "Report a bug": None,
        "About": "NeuroPeds AI — Clinical Decision Support for Pediatric Brain Tumor Segmentation."
    }
)

# ==========================================================
# 2. NAVIGATION STRUCTURE
# ==========================================================
pages = {
    "Platform": [
        st.Page("pages/Dashboard.py", title="Clinical Dashboard", icon="⚡"),
    ],
    "Clinical & Analysis": [
        st.Page("pages/MRI_Analysis.py", title="MRI Analysis Studio", icon="🖥️"),
        st.Page("pages/Segmentation_Report.py", title="Segmentation Report", icon="📋"),
        st.Page("pages/Clinical_View.py", title="Clinical Explainability", icon="🩺"),
    ],
    "System": [
        st.Page("pages/About.py", title="About NeuroPeds AI", icon="📚"),
    ]
}

pg = st.navigation(pages, position="sidebar")

# ==========================================================
# 3. GLOBAL THEME — FONTS, VARIABLES, COMPONENT STYLING
# ==========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    :root {
        --bg-base: #F4F7F9;
        --bg-surface: #FFFFFF;
        --border-soft: #E2E8F0;
        --border-accent: rgba(2, 132, 199, 0.25);
        --accent: #0284C7;
        --accent-2: #0891B2;
        --accent-violet: #4F46E5;
        --success: #059669;
        --warning: #B45309;
        --danger: #DC2626;
        --text-primary: #1E293B;
        --text-muted: #64748B;
        --sidebar-bg: #FFFFFF;
    }

    /* ---------- BASE APP ---------- */
    .stApp {
        background-color: var(--bg-base);
        background-image: 
            radial-gradient(at 0% 0%, rgba(2, 132, 199, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(79, 70, 229, 0.03) 0px, transparent 50%);
        background-attachment: fixed;
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    * { scrollbar-width: thin; scrollbar-color: rgba(2,132,199,0.3) transparent; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(2,132,199,0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(2,132,199,0.5);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseDot {
        0% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(5, 150, 105, 0); }
        100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
    }

    /* ---------- TYPOGRAPHY ---------- */
    h1, h2, h3 { 
        font-family: 'Outfit', sans-serif;
        color: #0F172A; 
        font-weight: 700; 
        letter-spacing: -0.01em; 
    }
    p, span, label, div { color: var(--text-primary); }

    .subtitle-muted {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* ---------- BADGES ---------- */
    .badge-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--success);
        animation: pulseDot 2s infinite;
        display: inline-block;
    }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-soft);
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }
    [data-testid="stSidebar"] .stPageLink,
    [data-testid="stSidebarNav"] a {
        border-radius: 8px;
        transition: all 0.2s ease;
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stPageLink:hover {
        background: rgba(2, 132, 199, 0.05);
        color: var(--accent) !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 4px 20px 4px;
        border-bottom: 1px solid var(--border-soft);
        margin-bottom: 14px;
    }
    .sidebar-brand-icon {
        font-size: 1.8rem;
    }
    .sidebar-brand-title { 
        font-weight: 700; 
        font-size: 1.2rem; 
        color: #0F172A; 
        line-height: 1.1; 
        font-family: 'Outfit', sans-serif;
    }
    .sidebar-brand-sub { 
        font-size: 0.75rem; 
        color: var(--accent); 
        font-weight: 500;
        letter-spacing: 0.02em; 
        margin-top: 2px;
    }

    .sidebar-status-box {
        background: #F8FAFC;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 14px;
        margin-top: 18px;
        font-size: 0.8rem;
    }
    .sidebar-status-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .sidebar-status-row:last-child { margin-bottom: 0; }
    .sidebar-status-label { color: var(--text-muted); font-weight: 500; }
    .sidebar-status-value { color: var(--success); font-weight: 600; display:flex; align-items:center; gap:6px; }

    /* ---------- BUTTONS ---------- */
    .stButton > button, .stDownloadButton > button {
        background: #FFFFFF;
        border: 1px solid var(--border-soft);
        color: #1E293B;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55em 1.4em;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(2, 132, 199, 0.1);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        border: none;
        color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
        color: #FFFFFF;
    }

    /* ---------- METRIC ---------- */
    [data-testid="stMetric"] {
        background: var(--bg-surface);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    /* ---------- DATAFRAMES / TABLES ---------- */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* ---------- CODE BLOCKS ---------- */
    code, pre {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        background: #F1F5F9;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 4. SIDEBAR BRANDING + LIVE STATUS FOOTER
# ==========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🧠</div>
        <div>
            <div class="sidebar-brand-title">NeuroPeds AI</div>
            <div class="sidebar-brand-sub">CLINICAL DECISION SUPPORT</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation renders automatically between the brand block and the footer
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-status-box">
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">System Status</span>
            <span class="sidebar-status-value"><span class="badge-dot"></span>Online</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Analysis Engine</span>
            <span class="sidebar-status-value" style="color: var(--accent);">Ready</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Compliance</span>
            <span class="sidebar-status-value" style="color: var(--accent-violet);">HIPAA · GDPR</span>
        </div>
        <div class="sidebar-status-row" style="margin-top:10px; border-top:1px solid #E2E8F0; padding-top:10px;">
            <span class="sidebar-status-label">Session Time</span>
            <span class="sidebar-status-value" style="color: var(--text-muted); font-weight:400; font-size: 0.75rem;">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 5. RUN NAVIGATION
# ==========================================================
pg.run()