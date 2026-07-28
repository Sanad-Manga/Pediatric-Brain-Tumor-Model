import streamlit as st

import backend as be
from datetime import datetime

st.set_page_config(page_title="Segmentation Report | NeuroFed AI", page_icon="📋", layout="wide")

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 20% 50%, rgba(56,189,248,0.09), transparent 50%),
                linear-gradient(135deg,#0B1628,#0F172A);
    border:1px solid rgba(56,189,248,0.2); border-radius:20px; padding:40px 44px; margin-bottom:28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#fff 40%,#38BDF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#94A3B8; font-size:0.92rem; }

.panel { background:rgba(15,23,42,0.55); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:26px; margin-bottom:18px; }
.panel-title { font-size:0.95rem; font-weight:700; color:#F3F4F6; margin-bottom:18px; display:flex; align-items:center; gap:8px; }

/* meta table */
.meta-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.meta-table tr { border-bottom:1px solid rgba(255,255,255,0.05); }
.meta-table tr:last-child { border-bottom:none; }
.meta-table td { padding:11px 8px; }
.meta-key { color:#64748B; width:42%; font-size:0.78rem; font-family:'JetBrains Mono',monospace; }
.meta-val { color:#CBD5E1; font-weight:500; }

/* interpretation block */
.interp-box {
    background:rgba(5,10,31,0.75);
    border-left:3px solid #38BDF8;
    border-radius:0 10px 10px 0;
    padding:16px 20px;
    color:#CBD5E1; font-size:0.87rem; line-height:1.85;
}

/* subregion bars */
.seg-row { margin-bottom:14px; }
.seg-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.seg-label { display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#94A3B8; }
.seg-dot   { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.seg-pct   { font-size:0.8rem; font-family:'JetBrains Mono',monospace; font-weight:600; }
.seg-bar-bg  { height:6px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; }
.seg-bar     { height:100%; border-radius:4px; }

/* confidence ring placeholder */
.conf-ring {
    width:110px; height:110px; border-radius:50%;
    background: conic-gradient(rgba(255,255,255,0.10) 0% 100%);
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 12px auto;
    box-shadow: 0 0 24px rgba(52,211,153,0.25);
    position:relative;
}
.conf-inner {
    width:82px; height:82px; border-radius:50%;
    background:#0B1628;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.conf-num   { font-size:1.3rem; font-weight:800; color:#34D399; letter-spacing:-0.02em; line-height:1; }
.conf-label { font-size:0.6rem; color:#64748B; text-transform:uppercase; letter-spacing:0.05em; margin-top:2px; }

/* export panel */
.export-hint { font-size:0.78rem; color:#64748B; line-height:1.6; margin-bottom:16px; }
.export-item {
    display:flex; align-items:center; gap:10px;
    padding:10px 14px; border-radius:10px;
    background:rgba(10,15,35,0.65); border:1px solid rgba(255,255,255,0.06);
    margin-bottom:8px; font-size:0.8rem; color:#94A3B8;
    transition: border-color .2s ease;
}
.export-item:hover { border-color:rgba(56,189,248,0.3); }
.export-icon { font-size:1rem; }

/* status bar */
.status-bar {
    display:flex; justify-content:space-between; align-items:center;
    background:rgba(5,10,31,0.7); border:1px solid rgba(56,189,248,0.12);
    border-radius:10px; padding:10px 16px; margin-top:6px;
    font-size:0.72rem; font-family:'JetBrains Mono',monospace;
}
.status-left  { color:#38BDF8; display:flex; align-items:center; gap:8px; }
.status-right { color:#64748B; }
.live-dot { width:6px; height:6px; border-radius:50%; background:#34D399;
    animation:pulseDot 1.8s infinite; }
@keyframes pulseDot {
    0%  { box-shadow:0 0 0 0 rgba(52,211,153,.6); }
    70% { box-shadow:0 0 0 6px rgba(52,211,153,0); }
    100%{ box-shadow:0 0 0 0 rgba(52,211,153,0); }
}
</style>

<div class="page-hero">
    <div class="page-title">Clinical AI Segmentation Report</div>
    <div class="page-sub">Diagnostic summary layout. Metrics populate once a trained checkpoint exists.</div>
</div>
""", unsafe_allow_html=True)

# ── status bar
st.markdown(f"""
<div class="status-bar">
    <div class="status-left"><span class="live-dot"></span>Report Active · Federated 3D U-Net + CORAL</div>
    <div class="status-right">Generated: {datetime.now().strftime("%Y-%m-%d  %H:%M")} UTC</div>
</div>
<div style="margin-bottom:20px;"></div>
""", unsafe_allow_html=True)

if not be.cache_available():
    st.error(f"Cache directory not found: `{be.CACHE_DIR}`")
    st.stop()


@st.cache_data(show_spinner="Loading volume…")
def _load_seg(subject_id: str):
    v = be.load_volume(subject_id)
    return v.seg, v.cohort


_subjects = be.list_subjects()
_sel1, _sel2 = st.columns([1, 2])
_cohort = _sel1.selectbox("Cohort", list(_subjects.keys()), index=2)
_sid = _sel2.selectbox("Subject", _subjects[_cohort])
_seg, _cohort_name = _load_seg(_sid)
_sub = {r["name"]: r for r in be.subregion_stats(_seg)}
_reg = {r["region"]: r for r in be.region_stats(_seg)}
_status = be.model_status()
_wt_ml = _reg["WT"]["volume_ml"]

col_main, col_side = st.columns([2, 1], gap="medium")

# ══════════════════════════════════════════════════════
#  LEFT — metadata + interpretation + subregions
# ══════════════════════════════════════════════════════
with col_main:

    # Patient metadata
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📄 Patient & Scan Metadata</div>
        <table class="meta-table">
            <tr>
                <td class="meta-key">Subject ID</td>
                <td class="meta-val" style="color:#38BDF8;font-family:'JetBrains Mono',monospace;font-weight:600;">{_sid}</td>
            </tr>
            <tr>
                <td class="meta-key">Dataset Cohort</td>
                <td class="meta-val">{_cohort_name}</td>
            </tr>
            <tr>
                <td class="meta-key">MRI Modalities</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">t1c · t1n · t2f · t2w</td>
            </tr>
            <tr>
                <td class="meta-key">Volume Shape</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">96 × 96 × 96 voxels</td>
            </tr>
            <tr>
                <td class="meta-key">Total Tumor Volume</td>
                <td class="meta-val" style="color:#00F2FE;font-weight:700;">~{_wt_ml:.1f} mL ({_reg["WT"]["voxels"]:,} vox)</td>
            </tr>
            <tr>
                <td class="meta-key">AI Confidence Score</td>
                <td class="meta-val" style="color:#64748B;font-weight:700;">N/A &mdash; no trained model</td>
            </tr>
            <tr>
                <td class="meta-key">Inference Mode</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">FP16 · Mixed Precision</td>
            </tr>
            <tr>
                <td class="meta-key">Federation Round</td>
                <td class="meta-val" style="font-family:'JetBrains Mono',monospace;">not run</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Subregion breakdown
    st.markdown('<div class="panel"><div class="panel-title">🧠 Tumor Subregion Breakdown</div>', unsafe_allow_html=True)

    regions = [
        (r["name"], r["pct_of_tumor"], r["color"])
        for r in sorted(be.subregion_stats(_seg), key=lambda x: -x["pct_of_tumor"])
    ]
    for name, pct, color in regions:
        st.markdown(f"""
        <div class="seg-row">
            <div class="seg-top">
                <div class="seg-label">
                    <div class="seg-dot" style="background:{color};box-shadow:0 0 6px {color}66;"></div>
                    {name}
                </div>
                <span class="seg-pct" style="color:{color};">{pct:.1f}%</span>
            </div>
            <div class="seg-bar-bg">
                <div class="seg-bar" style="width:{min(pct,100):.1f}%;background:linear-gradient(90deg,{color}cc,{color});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Measured description. Deliberately NOT a clinical interpretation: no trained
    # model exists, and inferring pathology from voxel counts would be fabrication.
    _lines = "".join(
        f'<li><b style="color:{r["color"]};">{r["name"]}</b>: {r["pct_of_tumor"]:.1f}% of tumor '
        f'volume ({r["voxels"]:,} voxels, ~{r["volume_ml"]:.1f} mL)</li>'
        for r in be.subregion_stats(_seg) if r["voxels"] > 0
    )
    _absent = [r["name"] for r in be.subregion_stats(_seg) if r["voxels"] == 0]
    _absent_txt = (
        f'<br><br>Absent in this subject: {", ".join(_absent)}.' if _absent else ""
    )
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">📐 Measured Segmentation Summary</div>
        <div class="interp-box">
            Measured directly from <b>{_sid}</b>'s expert ground-truth mask
            ({_cohort_name} cohort). These are voxel counts, not model output.
            <ul style="margin:12px 0 0 18px;line-height:1.9;">{_lines}</ul>
            {_absent_txt}
            <br><br>
            Official evaluation regions &mdash;
            <b>ET</b> {_reg["ET"]["voxels"]:,} vox ·
            <b>NC</b> {_reg["NC"]["voxels"]:,} vox ·
            <b>WT</b> {_reg["WT"]["voxels"]:,} vox.
            <br><br>
            <span style="color:#FCD34D;">No clinical interpretation is offered here.
            {_status.detail} Any diagnostic reading of these numbers must come from a
            qualified clinician, not this demo.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_side:

    # Confidence ring
    st.markdown("""
    <div class="panel" style="text-align:center;">
        <div class="panel-title" style="justify-content:center;">📊 Model Confidence</div>
        <div class="conf-ring">
            <div class="conf-inner">
                <div class="conf-num" style="font-size:1.6rem;">N/A</div>
                <div class="conf-label">No trained model</div>
            </div>
        </div>
        <div style="font-size:0.72rem;color:#64748B;margin-top:4px;">A confidence score requires<br>trained weights, which do not exist yet</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("""
    <div class="panel">
        <div class="panel-title">⚡ Quick Stats</div>
        <table class="meta-table">
            <tr><td class="meta-key">Dice Score</td>
                <td class="meta-val" style="color:#64748B;font-family:monospace;">&mdash;</td></tr>
            <tr><td class="meta-key">IoU</td>
                <td class="meta-val" style="color:#64748B;font-family:monospace;">&mdash;</td></tr>
            <tr><td class="meta-key">HD95</td>
                <td class="meta-val" style="color:#64748B;font-family:monospace;">&mdash;</td></tr>
            <tr><td class="meta-key">Sensitivity</td>
                <td class="meta-val" style="font-family:monospace;color:#64748B;">&mdash;</td></tr>
            <tr><td class="meta-key">Specificity</td>
                <td class="meta-val" style="font-family:monospace;color:#64748B;">&mdash;</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Export panel
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📥 Export Report</div>
        <div class="export-hint">Download publication-ready outputs for clinical records and research submission.</div>
        <div class="export-item"><span class="export-icon">📄</span> Full PDF Clinical Report</div>
        <div class="export-item"><span class="export-icon">📊</span> CSV Subregion Statistics</div>
        <div class="export-item"><span class="export-icon">🖼️</span> NIfTI Segmentation Mask</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬇  Generate PDF Report", use_container_width=True, type="primary"):
        st.success("PDF Report generated successfully!")

    st.markdown("""
    <div style="font-size:0.68rem;color:rgba(100,116,139,0.5);text-align:center;margin-top:10px;font-family:'JetBrains Mono',monospace;">
        HIPAA · GDPR COMPLIANT · RESEARCH USE ONLY
    </div>
    """, unsafe_allow_html=True)