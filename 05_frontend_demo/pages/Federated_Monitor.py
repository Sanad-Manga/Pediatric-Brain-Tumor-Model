"""Federation status — what the cohort actually looks like, and what has not run.

This page previously displayed a federated training run that never happened: 50
FedAvg rounds, five hospital nodes (two of which do not exist), per-site Dice
scores, and a convergence curve generated from a closed-form expression rather
than measured. `config.yaml` has `use_federation: false`, and section 03 trains
centrally by design — FedAvg belongs to `01_model_federated` and has not been
run against this cohort.

What is shown instead is real: the federated *split* genuinely exists (the
manifests partition patients across two hospitals plus a held-out site, and no
held-out patient is ever trained on), and the metrics are those measured for
the centrally-trained checkpoint.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from components.theme import apply_custom_theme
from utils import loaders

st.set_page_config(page_title="Federation Status | NeuroFed AI", page_icon="🌐", layout="wide")
apply_custom_theme()

MANIFESTS = Path(__file__).resolve().parents[2] / "00_shared" / "manifests"


def manifest_count(name: str) -> int:
    path = MANIFESTS / f"{name}.json"
    if not path.exists():
        return 0
    blob = json.loads(path.read_text(encoding="utf-8"))
    subjects = blob["subjects"] if isinstance(blob, dict) else blob
    return len(subjects)


st.markdown("""
<style>
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:8px; }
.page-sub { font-size:0.92rem; }
.panel-title { font-size:1rem; font-weight:700; margin-bottom:18px; }
.fed-stats-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
.fed-stat { flex:1; min-width:150px; background:#fff; border:1px solid #E2E8F0;
            border-radius:14px; padding:18px; text-align:center; }
.fed-stat-val { font-size:1.7rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em; }
.fed-stat-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
                  color:#64748B; margin-top:4px; }
.node-grid { display:flex; gap:12px; flex-wrap:wrap; }
.node-card { flex:1; min-width:190px; border:1px solid #E2E8F0; border-radius:12px;
             padding:16px; background:#fff; }
.node-name { font-weight:700; color:#0F172A; margin-bottom:4px; }
.node-stat { font-size:0.78rem; color:#64748B; }
.node-role { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;
             color:#0284C7; margin-bottom:8px; font-weight:700; }
</style>
<div class="page-hero">
    <div class="page-title">Federation Status</div>
    <div class="page-sub">The cohort split that federated training would use — and the current state of that training.</div>
</div>
""", unsafe_allow_html=True)

a, b, held = manifest_count("hospitalA"), manifest_count("hospitalB"), manifest_count("heldout")
roc = loaders.load_metrics_cache()
status = loaders.load_checkpoint_status()

st.warning(
    "**Federated training has not been run on this cohort.** "
    "`config.yaml` sets `use_federation: false`; the checkpoint shown across this "
    "demo was trained **centrally**. FedAvg is implemented in `01_model_federated` "
    "and has not been executed against these manifests. No round counts, per-site "
    "scores, or convergence curves are shown here because none have been measured."
)

st.markdown(f"""
<div class="fed-stats-row">
    <div class="fed-stat"><div class="fed-stat-val">2</div>
        <div class="fed-stat-label">Training sites</div></div>
    <div class="fed-stat"><div class="fed-stat-val">{a + b}</div>
        <div class="fed-stat-label">Training patients</div></div>
    <div class="fed-stat"><div class="fed-stat-val">{held}</div>
        <div class="fed-stat-label">Held-out patients</div></div>
    <div class="fed-stat"><div class="fed-stat-val">0</div>
        <div class="fed-stat-label">FedAvg rounds run</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="panel">
    <div class="panel-title">🏥 Cohort split (from the manifests)</div>
    <div class="node-grid">
        <div class="node-card">
            <div class="node-role">Training site</div>
            <div class="node-name">Hospital A</div>
            <div class="node-stat">{a} patients</div>
        </div>
        <div class="node-card">
            <div class="node-role">Training site</div>
            <div class="node-name">Hospital B</div>
            <div class="node-stat">{b} patients</div>
        </div>
        <div class="node-card">
            <div class="node-role">Never trained on</div>
            <div class="node-name">Held-out</div>
            <div class="node-stat">{held} patients</div>
        </div>
    </div>
    <div style="font-size:0.75rem;color:#64748B;margin-top:14px;line-height:1.6;">
        Scanner vendor and field strength are not recorded anywhere in this dataset,
        so they are not shown. The split itself is enforced by the manifests: held-out
        patients are excluded from every plan and never augmented.
    </div>
</div>
""", unsafe_allow_html=True)

if roc:
    st.markdown('<div class="panel"><div class="panel-title">📊 Measured Dice by region — held-out patients</div>',
                unsafe_allow_html=True)
    regions = [r for r in ("ET", "TC", "WT") if roc["regions"].get(r)]
    fig = go.Figure(go.Bar(
        x=regions,
        y=[roc["regions"][r]["dice"] * 100 for r in regions],
        marker_color=["#DC2626", "#4F46E5", "#0284C7"][:len(regions)],
        text=[f"{roc['regions'][r]['dice']*100:.1f}%" for r in regions],
        textposition="outside", textfont=dict(color="#0F172A", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(gridcolor="#E2E8F0", color="#64748B", range=[0, 100], title="Dice (%)"),
        xaxis=dict(color="#64748B"), font=dict(color="#0F172A"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Centrally-trained {status['epochs_completed']}-epoch checkpoint, measured over "
        f"{roc['n_slices']} tumour-bearing slices from {roc['n_subjects']} held-out patients. "
        "These are not per-site federated scores — there are none."
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No measured metrics yet — run `python -m utils.build_metrics_cache`.")
