import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="NeuroFed AI | Pediatric Brain Tumor Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "NeuroFed AI — Federated Deep Learning Platform for Pediatric Brain Tumor Segmentation & Clinical Decision Support."
    }
)

# ==========================================================
# 2. NAVIGATION STRUCTURE
# ==========================================================
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

pg = st.navigation(pages, position="sidebar")

# ==========================================================
# 3. GLOBAL THEME — FONTS, VARIABLES, COMPONENT STYLING
# ==========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    :root {
        --bg-deep: #030712;
        --bg-mid: #0B132B;
        --surface: rgba(15, 23, 42, 0.60);
        --surface-strong: rgba(15, 23, 42, 0.80);
        --border-soft: rgba(255, 255, 255, 0.07);
        --border-accent: rgba(56, 189, 248, 0.25);
        --accent: #38BDF8;
        --accent-2: #00F2FE;
        --accent-violet: #818CF8;
        --success: #34D399;
        --warning: #FBBF24;
        --danger: #F87171;
        --text-primary: #F3F4F6;
        --text-muted: #94A3B8;
    }

    /* ---------- BASE APP ---------- */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.08), transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(129, 140, 248, 0.08), transparent 40%),
            linear-gradient(135deg, var(--bg-deep) 0%, var(--bg-mid) 50%, #060913 100%);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    footer {visibility: hidden;}

    /* DO NOT hide the header. The only control that reopens a collapsed sidebar
       lives inside it, so `header {visibility: hidden}` made one accidental click
       on the ‹‹ collapse arrow remove the navigation permanently — the sidebar
       ends up translated off-screen with no way back.
       Hide just the toolbar contents (Deploy button, hamburger menu) instead and
       leave the header itself alive. */
    header, [data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stMainMenuButton"],
    #MainMenu {
        display: none !important;
    }

    /* ---------- SIDEBAR IS STATIC ----------
       The sidebar is pinned open and cannot be collapsed. Streamlit persists its
       collapsed state in localStorage (`stSidebarCollapsed-`), so one accidental
       click on the ‹‹ arrow hid the navigation on every future visit, across
       reloads and server restarts. Rather than depend on that toggle behaving,
       the collapse control is removed and the open state is forced here, which
       also neutralises any collapsed flag already saved in a user's browser. */
    [data-testid="stSidebar"] {
        transform: none !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        margin-left: 0 !important;
        left: 0 !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
    }
    /* the ‹‹ collapse arrow, and the ›› control that would reopen it */
    [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    /* stop the drag-to-resize handle from shrinking it to nothing */
    [data-testid="stSidebarResizeHandle"] { display: none !important; }

    * { scrollbar-width: thin; scrollbar-color: rgba(56,189,248,0.35) transparent; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(56,189,248,0.5), rgba(129,140,248,0.5));
        border-radius: 10px;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseDot {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }
        70% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }
    @keyframes shimmer {
        0% { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }

    /* ---------- CARDS ---------- */
    .hero-card {
        background: var(--surface);
        border: 1px solid var(--border-accent);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 40px 30px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 30px;
        text-align: center;
        animation: fadeInUp 0.5s ease-out;
    }

    .content-card {
        background: var(--surface);
        border: 1px solid var(--border-soft);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    .content-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-accent);
        box-shadow: 0 16px 40px rgba(56, 189, 248, 0.12);
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-card.success { border-color: rgba(52, 211, 153, 0.3); }
    .metric-card.warning { border-color: rgba(251, 191, 36, 0.3); }
    .metric-card.danger  { border-color: rgba(248, 113, 113, 0.3); }

    /* ---------- TYPOGRAPHY ---------- */
    h1, h2, h3 { color: #FFFFFF; font-weight: 700; letter-spacing: -0.01em; }
    p, span, label, div { color: var(--text-primary); }

    .gradient-title {
        background: linear-gradient(135deg, #FFFFFF 30%, #38BDF8 70%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle-muted {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* ---------- BADGES ---------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 30px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--success);
        animation: pulseDot 1.8s infinite;
    }
    .badge-violet { color: var(--accent-violet); border-color: rgba(129, 140, 248, 0.3); background: rgba(129, 140, 248, 0.1); }
    .badge-success { color: var(--success); border-color: rgba(52, 211, 153, 0.3); background: rgba(52, 211, 153, 0.1); }
    .badge-warning { color: var(--warning); border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.1); }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-deep);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] .stPageLink,
    [data-testid="stSidebarNav"] a {
        border-radius: 10px;
        transition: background 0.2s ease, padding-left 0.2s ease;
    }
    [data-testid="stSidebar"] .stPageLink:hover {
        background: rgba(56, 189, 248, 0.08);
        padding-left: 4px;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #FFFFFF; }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 4px 18px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 14px;
    }
    .sidebar-brand-icon {
        font-size: 1.6rem;
        filter: drop-shadow(0 0 10px rgba(56,189,248,0.5));
    }
    .sidebar-brand-title { font-weight: 700; font-size: 1.05rem; color: #fff; line-height: 1.1; }
    .sidebar-brand-sub { font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; letter-spacing: 0.03em; }

    .sidebar-status-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 18px;
        font-size: 0.75rem;
    }
    .sidebar-status-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
    .sidebar-status-row:last-child { margin-bottom: 0; }
    .sidebar-status-label { color: var(--text-muted); }
    .sidebar-status-value { color: var(--success); font-family: 'JetBrains Mono', monospace; font-weight: 600; display:flex; align-items:center; gap:6px; }

    /* ---------- BUTTONS ---------- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #F3F4F6;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55em 1.4em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
        color: var(--accent);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7, #0ea5e9);
        border: none;
        color: #04121c;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px 10px 0 0;
        color: var(--text-muted);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ---------- EXPANDER ---------- */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background: var(--surface) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px !important;
        font-weight: 600;
    }

    /* ---------- ALERTS ---------- */
    div[data-testid="stAlertContainer"] { border-radius: 12px; border: 1px solid var(--border-soft); }

    /* ---------- DATAFRAMES / TABLES ---------- */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
    }

    /* ---------- PROGRESS BAR ---------- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }

    /* ---------- METRIC (native st.metric) ---------- */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 14px 18px;
    }

    /* ---------- DIVIDER ---------- */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);
        margin: 1.6em 0;
    }

    /* ---------- CODE BLOCKS ---------- */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
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
            <div class="sidebar-brand-title">NeuroFed AI</div>
            <div class="sidebar-brand-sub">PEDIATRIC ONCOLOGY · v2.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation renders automatically between the brand block and the footer
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-status-box">
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Federated Network</span>
            <span class="sidebar-status-value"><span class="badge-dot"></span>Active</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Model Registry</span>
            <span class="sidebar-status-value" style="color: var(--accent);">Synced</span>
        </div>
        <div class="sidebar-status-row">
            <span class="sidebar-status-label">Compliance</span>
            <span class="sidebar-status-value" style="color: var(--accent-violet);">HIPAA · GDPR</span>
        </div>
        <div class="sidebar-status-row" style="margin-top:8px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">
            <span class="sidebar-status-label">Session</span>
            <span class="sidebar-status-value" style="color: var(--text-muted); font-weight:400;">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 5. RUN NAVIGATION
# ==========================================================
# ==========================================================
# 5. PIN THE SIDEBAR OPEN
# ==========================================================
# Streamlit stores the sidebar's collapsed state in localStorage
# (`stSidebarCollapsed-`), so one click on the collapse arrow hid the navigation
# on every later visit - across reloads, new tabs and server restarts - and the
# reopen control sat inside the header this app hides. CSS alone does not win:
# Streamlit re-applies an inline width and a -300px transform after every rerun.
# This clears the stored flag and re-expands the sidebar if anything collapses
# it. Runs in a zero-height iframe, reaching the host page via window.parent.
components.html(
    """
    <script>
    (function () {
        const doc = window.parent.document;
        function pin() {
            try { window.parent.localStorage.setItem('stSidebarCollapsed-', 'false'); } catch (e) {}
            const sb = doc.querySelector('[data-testid="stSidebar"]');
            if (!sb) return;
            if (sb.getAttribute('aria-expanded') === 'false') {
                const btn = doc.querySelector(
                    '[data-testid="stExpandSidebarButton"] button,' +
                    '[data-testid="stSidebarCollapsedControl"] button'
                );
                if (btn) { btn.click(); return; }
            }
            sb.style.setProperty('transform', 'none', 'important');
            sb.style.setProperty('width', '300px', 'important');
            sb.style.setProperty('min-width', '300px', 'important');
            sb.style.setProperty('visibility', 'visible', 'important');
        }
        pin();
        setInterval(pin, 400);
        new MutationObserver(pin).observe(doc.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    height=0,
)


pg.run()
