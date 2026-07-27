import streamlit as st

# No st.set_page_config() here: Home.py is the entry script and already sets it.
# Calling it again from the default page tears down the sidebar navigation.

# Hero
st.markdown("""
    <div class="hero-card">
        <div class="badge">BraTS-PEDs · Federated 3D Segmentation</div>
        <h1 class="gradient-title" style="font-size: 2.75rem; margin: 0 0 12px 0;">NeuroFed AI</h1>
        <p style="color: #94A3B8; font-size: 1rem; max-width: 720px; margin: 0 auto;">
            Privacy-preserving pediatric brain tumor segmentation across institutional
            boundaries — federated training, domain adaptation, and held-out validation
            without moving a single patient scan.
        </p>
    </div>
""", unsafe_allow_html=True)

# Cohort at a glance
c1, c2, c3, c4 = st.columns(4)
cohort = [
    (c1, "Hospital A", "53", "training subjects"),
    (c2, "Hospital B", "92", "training subjects"),
    (c3, "Held-out Site", "82", "validation subjects"),
    (c4, "Modalities", "4", "t1c · t1n · t2f · t2w"),
]
for col, label, value, caption in cohort:
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.7rem; color: #38BDF8; font-family: monospace; text-transform: uppercase;">{label}</div>
                <div style="font-size: 2rem; color: #FFFFFF; font-weight: 700; margin: 6px 0;">{value}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">{caption}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Where to go next
left, right = st.columns(2)

with left:
    st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.15rem; margin-bottom: 14px;">🩺 Clinical & Analysis</h3>
            <p style="color: #CBD5E1; font-size: 0.875rem; line-height: 1.8;">
                Inspect volumes slice by slice in the <b>MRI Analysis Studio</b>, review
                per-region tumor breakdowns in the <b>Segmentation Report</b>, and trace
                model reasoning in <b>Clinical Explainability</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
        <div class="content-card">
            <h3 style="font-size: 1.15rem; margin-bottom: 14px;">🌐 Federated & AI Core</h3>
            <p style="color: #CBD5E1; font-size: 0.875rem; line-height: 1.8;">
                Watch FedAvg convergence in the <b>Federated Observatory</b>, compare
                feature alignment in the <b>Domain Adaptation Lab</b>, and browse the
                ablation matrix under <b>Model Intelligence</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <div class="content-card">
        <h3 style="font-size: 1.15rem; margin-bottom: 12px;">⚙️ Platform Configuration</h3>
        <p style="color: #CBD5E1; font-size: 0.875rem; line-height: 1.8;">
            Volumes are resampled to <b>96³</b> at FP16 with batch size 1. Evaluation uses
            the official BraTS-PEDs regions — <b>ET</b> (enhancing tumor), <b>NC</b>
            (enhancing + cystic + necrosis complex) and <b>WT</b> (whole tumor) — scored on
            the held-out institution only.
        </p>
        <p style="color: #64748B; font-size: 0.8rem; margin-top: 14px; font-style: italic;">
            Note: figures shown across this demo are placeholders until the trained
            checkpoints from <code>01_model_federated</code> are wired in.
        </p>
    </div>
""", unsafe_allow_html=True)
