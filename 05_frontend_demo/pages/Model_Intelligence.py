import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from components.theme import apply_custom_theme
from utils.inference import DEFAULT_CHECKPOINT, load_model
from utils.loaders import cache_dir, load_metrics_cache, load_training_history

st.set_page_config(page_title="Model Intelligence | NeuroFed AI", page_icon="🧠", layout="wide")
apply_custom_theme()


def _load():
    """The trained model, for reading real architecture facts off it."""
    return load_model(DEFAULT_CHECKPOINT, str(cache_dir()))

st.markdown("""
<style>
.page-hero {
    background: radial-gradient(circle at 20% 50%, rgba(168,85,247,0.07), transparent 50%),
                linear-gradient(135deg,#F8FAFC,#EDE9FE);
    border:1px solid #E2E8F0; border-radius:20px; padding:40px 44px; margin-bottom:28px;
}
.page-title { font-size:2rem; font-weight:800; letter-spacing:-0.02em;
    background:linear-gradient(135deg,#0F172A 30%,#7E22CE 80%,#0284C7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
.page-sub { color:#64748B; font-size:0.92rem; }
.panel { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:28px; margin-bottom:20px; }
.panel-title { font-size:1rem; font-weight:700; color:#0F172A; margin-bottom:18px; }
.spec-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.spec-table tr { border-bottom:1px solid #E2E8F0; }
.spec-table tr:last-child { border-bottom:none; }
.spec-table td { padding:12px 8px; }
.spec-key { color:#64748B; width:42%; font-family:'JetBrains Mono',monospace; font-size:0.78rem; }
.spec-val { color:#334155; font-weight:500; }
.spec-val b { color:#0F172A; }
.spec-val .tag { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.7rem;
    font-family:'JetBrains Mono',monospace; background:rgba(56,189,248,0.1);
    color:#0284C7; border:1px solid rgba(56,189,248,0.2); margin-left:6px; }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px; }
.metric-box { background:#F8FAFC; border:1px solid #E2E8F0;
    border-radius:12px; padding:18px; text-align:center;
    transition: border-color .2s, transform .2s; }
.metric-box:hover { transform:translateY(-2px); border-color:rgba(168,85,247,0.3); }
.metric-val { font-size:1.9rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:4px; }
.metric-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:#64748B; margin-bottom:8px; }
.metric-bar-bg { height:3px; background:#E2E8F0; border-radius:3px; overflow:hidden; }
.metric-bar { height:100%; border-radius:3px; }
.layer-row { display:flex; align-items:center; gap:14px; margin-bottom:12px; }
.layer-name { font-size:0.78rem; font-family:'JetBrains Mono',monospace; color:#64748B; width:180px; flex-shrink:0; }
.layer-bar-bg { flex:1; height:8px; background:#E2E8F0; border-radius:4px; overflow:hidden; }
.layer-bar { height:100%; border-radius:4px; }
.layer-params { font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:#64748B; width:80px; text-align:right; }
</style>

<div class="page-hero">
    <div class="page-title">Model Intelligence & Architecture</div>
    <div class="page-sub">Architecture, parameter counts and validation benchmarks read directly from the trained checkpoint.</div>
</div>
""", unsafe_allow_html=True)

# ── Benchmark metrics -- measured, not illustrative.
# These were hardcoded (92.4% Dice / 4.8mm HD95 / 94.2% sens / 98.1% spec) and
# bore no relation to any model that was ever trained. They now come from
# data/roc_cache.json, computed by utils.build_metrics_cache over the held-out
# patients. Whole tumour is shown as the headline region because it is the one
# clinicians read first; the per-region table lives on Model Performance.
_roc = load_metrics_cache()

if _roc is None:
    st.warning(
        "No measured metrics available yet. Build them with "
        "`python -m utils.build_metrics_cache` — this page will not display "
        "placeholder benchmark numbers."
    )
else:
    _wt = _roc["regions"]["WT"]
    _cards = [
        (f"{_wt['dice']*100:.1f}%", "Mean Dice (WT)", _wt["dice"] * 100, "#0284C7", "#0891B2"),
        (f"{_wt['hd95_median_mm']:.1f}mm", "HD95 (WT, median)",
         max(0, 100 - _wt["hd95_median_mm"] * 10), "#0891B2", "#0284C7"),
        (f"{_wt['sensitivity']*100:.1f}%", "Sensitivity (WT)",
         _wt["sensitivity"] * 100, "#059669", "#0284C7"),
        (f"{_wt['specificity']*100:.2f}%", "Specificity (WT)",
         _wt["specificity"] * 100, "#7E22CE", "#4F46E5"),
    ]
    _boxes = "".join(
        f"""<div class="metric-box">
            <div class="metric-val" style="background:linear-gradient(135deg,#0F172A,{c1});
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{value}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-bar-bg"><div class="metric-bar"
                 style="width:{min(width,100):.1f}%;background:linear-gradient(90deg,{c1},{c2});"></div></div>
        </div>"""
        for value, label, width, c1, c2 in _cards
    )
    st.markdown(f'<div class="metric-grid">{_boxes}</div>', unsafe_allow_html=True)
    st.caption(
        f"Measured on {_roc['n_subjects']} held-out patients "
        f"({_roc['n_slices']} tumour-bearing {_roc['plane']} slices) using the "
        f"{_roc['checkpoint']['epochs_completed']}-epoch checkpoint. "
        "Specificity is high because most pixels are background — read it "
        "alongside Dice, never alone."
    )

col1, col2 = st.columns(2, gap="medium")

with col1:
    # Read the specs off the loaded model rather than restating them. The
    # previous table described a 31.2M-parameter federated 3D U-Net on
    # 96x96x96 volumes; the model in this repo is a 0.48M-parameter 2D U-Net
    # on 256x256 slices, trained centrally.
    st.markdown('<div class="panel"><div class="panel-title">⚙️ Architecture Specifications</div>',
                unsafe_allow_html=True)
    try:
        _model, _head, _cfg, _meta = _load()
        _total = sum(p.numel() for p in _model.parameters())
        _h, _w = _cfg.common_size
        rows = [
            ("Backbone", f"<b>2D U-Net</b> · depth {len(_model.encoders)} · instance norm"),
            ("Input shape", f"<b>{len(_cfg.modalities)} × {_h} × {_w}</b> "
                            f"<span class='tag'>float32</span>"),
            ("Modalities", " · ".join(_cfg.modalities)),
            ("Loss function", "DiceCELoss (MONAI), background excluded"
                              + (" + tumour-type CE" if _meta["has_type_head"] else "")),
            ("Optimizer", "AdamW <span class='tag'>lr=1e-3</span>"),
            ("Weight decay", "<b>1e-5</b>"),
            ("Federation", "not used — trained centrally "
                           "<span class='tag'>use_federation: false</span>"),
            ("Output classes", f"{_cfg.num_classes} (background + labels 1–4)"),
            ("Total parameters", f"<b>{_total:,}</b> ({_total/1e6:.2f}M)"),
            ("Epochs trained", f"<b>{_meta['epochs_completed']}</b>"),
        ]
        st.markdown(
            '<table class="spec-table">'
            + "".join(f'<tr><td class="spec-key">{k}</td><td class="spec-val">{v}</td></tr>'
                      for k, v in rows)
            + "</table>", unsafe_allow_html=True)
        st.caption("Section 01 owns the production architecture; this compact U-Net exists so "
                   "section 03 can train and score end to end.")
    except Exception as exc:                      # no checkpoint on this machine
        st.info(f"Architecture unavailable — no checkpoint loaded ({exc}).")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Parameter counts read off the real modules. The previous bars were
    # invented (1.2M/2.4M/9.6M for a model that has 0.48M parameters total).
    st.markdown('<div class="panel"><div class="panel-title">📊 Parameter distribution by module</div>',
                unsafe_allow_html=True)
    try:
        _model, _head, _cfg, _meta = _load()
        groups = [(name, sum(p.numel() for p in mod.parameters()))
                  for name, mod in _model.named_children()]
        groups = [(n, c) for n, c in groups if c]
        if _head is not None:
            groups.append(("tumour-type head", sum(p.numel() for p in _head.parameters())))
        biggest = max(c for _, c in groups)
        palette = ["#0284C7", "#7E22CE", "#4F46E5", "#059669", "#0891B2", "#B45309"]
        html = ""
        for i, (name, count) in enumerate(groups):
            html += f"""
            <div class="layer-row">
                <div class="layer-name">{name}</div>
                <div class="layer-bar-bg"><div class="layer-bar"
                     style="width:{count / biggest * 100:.0f}%;background:{palette[i % len(palette)]};"></div></div>
                <div class="layer-params">{count / 1e6:.3f}M</div>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Bars are relative to the largest module. The bottleneck dominates, "
                   "as expected for a U-Net.")
    except Exception as exc:
        st.info(f"Parameter breakdown unavailable — no checkpoint loaded ({exc}).")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Training curves — the real ones, from history.json
st.markdown('<div class="panel"><div class="panel-title">📈 Training loss & validation Dice</div>',
            unsafe_allow_html=True)

_history = load_training_history()
if not _history:
    st.info("No `history.json` for this run yet, so no curve is drawn. "
            "Train a model, or recover a curve from a log with "
            "`03_augmentation_eval/tools/history_from_log.py`.")
else:
    _epochs = [h["epoch"] for h in _history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_epochs, y=[h["loss"] for h in _history], name="Train loss",
                             line=dict(color="#0F172A", width=2)))
    fig.add_trace(go.Scatter(x=_epochs, y=[h.get("mean_dice") for h in _history],
                             name="Val mean Dice", yaxis="y2",
                             line=dict(color="#059669", width=2)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(color="#0F172A")),
        yaxis=dict(gridcolor="#E2E8F0", color="#64748B", title="loss"),
        yaxis2=dict(title="Dice", overlaying="y", side="right", range=[0, 1],
                    color="#64748B", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#E2E8F0", color="#64748B", title="Epoch"),
        font=dict(color="#0F172A"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(_history)} epochs recorded (epoch {_epochs[0]}–{_epochs[-1]}). "
               "Validation Dice is measured per held-out patient, unaugmented.")
st.markdown("</div>", unsafe_allow_html=True)
