import streamlit as st

#: Light palette. Pages that hardcoded dark panel colours read these variables
#: instead, so the whole app follows one source of truth.
LIGHT_CSS = """
<style>
    :root {
        --bg:            #FFFFFF;
        --bg-subtle:     #F8FAFC;
        --panel:         #FFFFFF;
        --panel-border:  #E2E8F0;
        --text:          #0F172A;
        --text-muted:    #64748B;
        --text-faint:    #94A3B8;
        --accent:        #0284C7;
        --accent-soft:   #E0F2FE;
        --good:          #059669;
        --warn:          #B45309;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background-color: var(--bg-subtle);
        border-right: 1px solid var(--panel-border);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    /* Panels: the dark glass look becomes a light card with a real border,
       since translucency over white reads as washed-out grey. */
    .panel, .glass-card, .glass-metric {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 12px rgba(15,23,42,0.04);
        backdrop-filter: none;
    }
    .glass-metric { text-align: center; padding: 16px; }

    h1, h2, h3 { color: var(--text); font-weight: 700; letter-spacing: -0.02em; }
    .panel-title { color: var(--text); }
    .page-sub, .meta-key { color: var(--text-muted); }
    .meta-val { color: var(--text); }

    /* The hero and gradient text were built for a dark backdrop; on white they
       need a light wash and a darker gradient or the title disappears. */
    .page-hero {
        background: linear-gradient(135deg, #F8FAFC 0%, #E0F2FE 100%) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 18px;
        padding: 36px 40px;
        margin-bottom: 26px;
    }
    .page-title, .gradient-text {
        background: linear-gradient(135deg, #0F172A 30%, #0284C7 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .status-bar {
        background: var(--bg-subtle) !important;
        border: 1px solid var(--panel-border) !important;
        color: var(--text-muted);
    }
    .status-left { color: var(--accent) !important; }
    .status-right { color: var(--text-muted) !important; }

    .interp-box {
        background: var(--accent-soft) !important;
        border-left: 3px solid var(--accent);
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        color: var(--text) !important;
    }
    .caveat-box {
        background: #FEF3C7 !important;
        border-left: 3px solid #D97706;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        color: #78350F !important;
    }

    .meta-table tr { border-bottom: 1px solid var(--panel-border); }
    .seg-bar-bg { background: #E2E8F0; }
    .conf-inner { background: var(--bg) !important; }
    .conf-num   { color: var(--good) !important; }
    .conf-label { color: var(--text-muted) !important; }
</style>
"""


def apply_custom_theme():
    """Apply the light clinical theme.

    Was a dark 'command center' theme; the app now renders on white. Panel and
    text colours come from the CSS variables above so pages stay consistent.
    """
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
