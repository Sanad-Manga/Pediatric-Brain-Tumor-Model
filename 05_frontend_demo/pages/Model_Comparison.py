import streamlit as st
import numpy as np
import json
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# Directory setup
BASE_DIR = Path(__file__).resolve().parents[1]
COMPARISON_CACHE = BASE_DIR / "comparison_cache"
DEMO_CACHE = BASE_DIR / "demo_cache"

# Inject Custom CSS matching the Modern SaaS theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
* { font-family: 'Inter', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif; }

.page-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F9FF 60%, #E0F2FE 100%);
    border: 1px solid rgba(14, 165, 233, 0.15); border-radius: 20px;
    padding: 30px 40px; margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.02);
}
.page-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; font-family: 'Outfit', sans-serif;
    color: #0F172A; margin-bottom: 8px;
}
.page-sub { color: #475569; font-size: 0.95rem; line-height: 1.6; }
.badge-ok { background:#D1FAE5; color:#065F46; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
.badge-review { background:#FEF3C7; color:#92400E; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }

.review-alert {
    background-color: #FDF4FF;
    border-left: 4px solid #D946EF;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(217, 70, 239, 0.05);
}
.review-title {
    color: #86198F;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.review-desc {
    color: #A21CAF;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Make st.metric look elegant */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
</style>
""", unsafe_allow_html=True)

def load_patient_list():
    if not COMPARISON_CACHE.exists():
        return []
    return sorted([d.name for d in COMPARISON_CACHE.iterdir() if d.is_dir()])

def load_patient_data(patient_id):
    patient_dir = COMPARISON_CACHE / patient_id
    
    with open(patient_dir / "meta.json", "r") as f:
        meta = json.load(f)
    lbl_2d = np.load(patient_dir / "labels_2d.npz")["labels"]
    lbl_3d = np.load(patient_dir / "labels_3d.npz")["labels"]
    lbl_rstar = np.load(patient_dir / "labels_rstar.npz")["labels"]
    
    review_mask = None
    review_spots = []
    
    mask_path = patient_dir / "review_mask.npz"
    spots_path = patient_dir / "review_spots.json"
    
    if mask_path.exists() and spots_path.exists():
        review_mask = np.load(mask_path)["mask"]
        with open(spots_path, "r") as f:
            review_spots = json.load(f)
            
    # ─── 🌟 Bonus: Load Hi-Res R* if available (Issue #53) ───
    lbl_rstar_hires = None
    hires_path = patient_dir / "labels_rstar_hires.npz"
    if hires_path.exists():
        lbl_rstar_hires = np.load(hires_path)["labels"]
            
    return meta, lbl_2d, lbl_3d, lbl_rstar, review_mask, review_spots, lbl_rstar_hires

def get_background_slice(patient_id, slice_idx):
    bg_path = DEMO_CACHE / patient_id / "axial" / f"slice_{slice_idx:03d}.npz"
    if bg_path.exists():
        img_arr = np.load(bg_path)["image"][0]
        return np.rot90(img_arr, 2).copy()
    return None

def create_plotly_viewer(bg_img, mask_img, show_mask, review_img=None, show_review=False):
    fig = go.Figure()
    
    if bg_img is not None:
        bg_img = np.ascontiguousarray(bg_img.T.copy())
    
    mask_img = np.ascontiguousarray(mask_img.T.copy())
    
    if review_img is not None:
        review_img = np.ascontiguousarray(review_img.T.copy())

    if bg_img is not None:
        fig.add_trace(go.Heatmap(z=bg_img, colorscale='gray', showscale=False, hoverinfo='skip'))
    else:
        fig.add_trace(go.Heatmap(z=np.zeros_like(mask_img), colorscale='gray', zmin=0, zmax=1, showscale=False, hoverinfo='skip'))
        
    if show_mask:
        mask_display = np.where(mask_img == 0, np.nan, mask_img)
        colorscale = [
            [0.00, 'rgba(0,0,0,0)'],
            [0.25, 'rgba(239,68,68,0.65)'],
            [0.50, 'rgba(16,185,129,0.65)'],
            [0.75, 'rgba(59,130,246,0.65)'],
            [1.00, 'rgba(234,179,8,0.65)']
        ]
        fig.add_trace(go.Heatmap(z=mask_display, colorscale=colorscale, zmin=0, zmax=4, showscale=False, hoverinfo='skip'))

    if show_review and review_img is not None:
        review_display = np.where(review_img == 0, np.nan, 1)
        review_colorscale = [
            [0.0, 'rgba(0,0,0,0)'],
            [1.0, 'rgba(217,70,239,0.9)']
        ]
        fig.add_trace(go.Heatmap(z=review_display, colorscale=review_colorscale, zmin=0, zmax=1, showscale=False, hoverinfo='skip'))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange='reversed'),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='black',
        paper_bgcolor='rgba(0,0,0,0)',
        height=320
    )
    return fig

def main():
    st.markdown("""
    <div class="page-hero">
        <div class="page-title">🔍 Model Comparison Matrix</div>
        <div class="page-sub">Evaluate architectural variants side-by-side. Compare the 2D ensemble, 3D family, and the R* fusion approach against ground truth metrics.</div>
    </div>
    """, unsafe_allow_html=True)
    
    patients = load_patient_list()
    if not patients:
        st.error("Comparison cache directory not found or empty.")
        return
        
    selected_patient = st.selectbox("Select Patient Record", patients)
    
    if selected_patient:
        meta, lbl_2d, lbl_3d, lbl_rstar, review_mask, review_spots, lbl_rstar_hires = load_patient_data(selected_patient)
        
        status = meta.get("rstar_status", "unknown")
        warnings = meta.get("rstar_warnings", [])
        
        st.markdown(f"**R* Pipeline Status:** <span class='badge-{'ok' if status == 'ok' else 'review'}'>{status.upper()}</span>", unsafe_allow_html=True)
        for w in warnings:
            st.warning(f"⚠️ **Warning:** {w}")
            
        st.divider()
        
        # ─── DYNAMIC SLICE RANGE CALCULATION ───
        max_slice = lbl_rstar.shape[2] - 1
        has_review = len(review_spots) > 0
        has_hires = lbl_rstar_hires is not None
        
        demo_axial_dir = DEMO_CACHE / selected_patient / "axial"
        avail_bg_slices = []
        if demo_axial_dir.exists():
            for f in demo_axial_dir.glob("slice_*.npz"):
                try:
                    avail_bg_slices.append(int(f.stem.split('_')[1]))
                except ValueError:
                    pass
                    
        if avail_bg_slices:
            min_bg = min(avail_bg_slices)
            max_bg = max(avail_bg_slices)
            default_slice = (min_bg + max_bg) // 2
            bg_caption = f"ℹ️ Background MRI available for slices {min_bg}–{max_bg}. Other slices display on a black canvas."
        else:
            default_slice = max_slice // 2
            bg_caption = "ℹ️ Background MRI unavailable for this demo record. Slices display on a black canvas."
            
        # ─── DYNAMIC CONTROL PANEL LAYOUT ───
        layout = [2.5, 1]
        if has_review: layout.append(1)
        if has_hires: layout.append(1.2) # Give hi-res toggle a slightly wider column
        
        cols = st.columns(layout)
        
        with cols[0]:
            slice_idx = st.slider("Axial Slice Navigation", 0, max_slice, default_slice)
            st.caption(bg_caption)
            
        with cols[1]:
            st.write("") 
            show_mask = st.toggle("Overlay Segmentation", value=True)
            
        idx = 2
        show_review = False
        if has_review:
            with cols[idx]:
                st.write("")
                show_review = st.toggle("🟣 Flagged Spots", value=True)
            idx += 1
            
        use_hires = False
        if has_hires:
            with cols[idx]:
                st.write("")
                use_hires = st.toggle("✨ Hi-Res Mode", value=False)
                
        bg_img = get_background_slice(selected_patient, slice_idx)
            
        img_col1, img_col2, img_col3 = st.columns(3)
        rev_slice = review_mask[:, :, slice_idx] if review_mask is not None else None
        
        # Decide which R* label mask to display
        active_rstar_lbl = lbl_rstar_hires if use_hires else lbl_rstar
        rstar_title = "R* (Hi-Res)" if use_hires else "R* (Fusion)"
        
        with img_col1:
            st.markdown("<h4 style='font-family: Outfit, sans-serif; color: #0F172A;'>2D Architecture</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_plotly_viewer(bg_img, lbl_2d[:, :, slice_idx], show_mask, None, False), use_container_width=True, config={'displayModeBar': False}, key="viewer_2d")
        with img_col2:
            st.markdown("<h4 style='font-family: Outfit, sans-serif; color: #0F172A;'>3D Architecture</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_plotly_viewer(bg_img, lbl_3d[:, :, slice_idx], show_mask, None, False), use_container_width=True, config={'displayModeBar': False}, key="viewer_3d")
        with img_col3:
            st.markdown(f"<h4 style='font-family: Outfit, sans-serif; color: #0EA5E9;'>{rstar_title}</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_plotly_viewer(bg_img, active_rstar_lbl[:, :, slice_idx], show_mask, rev_slice, show_review), use_container_width=True, config={'displayModeBar': False}, key="viewer_rstar")

        if has_review:
            st.markdown(f"""
            <div class="review-alert">
                <div class="review-title">🟣 {len(review_spots)} spots flagged for review</div>
                <div class="review-desc">Flagged spots are areas the model is unsure about and are meant for expert review, not a diagnosis.</div>
            </div>
            """, unsafe_allow_html=True)
            
            df_spots = pd.DataFrame(review_spots)
            df_spots.rename(columns={
                "spot_id": "Spot ID",
                "voxels": "Volume (Voxels)",
                "mean_et_prob": "Confidence (Mean Prob)",
                "models_agree": "Models Agreement",
                "reason": "Reason Flagged"
            }, inplace=True)
            
            st.dataframe(
                df_spots,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Spot ID": st.column_config.NumberColumn(format="%d"),
                    "Volume (Voxels)": st.column_config.NumberColumn(format="%d"),
                    "Confidence (Mean Prob)": st.column_config.NumberColumn(format="%.3f"),
                    "Models Agreement": st.column_config.NumberColumn(format="%.3f"),
                    "Reason Flagged": st.column_config.TextColumn()
                }
            )

        st.divider()
        
        st.markdown("<h3 style='font-family: Outfit, sans-serif; color: #0F172A;'>📊 Multi-Model Performance Comparison</h3>", unsafe_allow_html=True)
        
        # Display the research note if Hi-Res is active
        if use_hires and "rstar_hires_note" in meta:
            st.caption(f"🔬 **Research Option Active:** {meta['rstar_hires_note']}")
        else:
            st.caption("Dice similarity coefficient per region across all architectures.")
        
        regions = ["ET", "NC", "WT"]
        metrics_2d = meta.get("regions", {}).get("2d", {})
        metrics_3d = meta.get("regions", {}).get("3d", {})
        
        # Choose which metrics to display in the 3rd column
        if use_hires:
            metrics_rstar = meta.get("regions", {}).get("rstar_hires", {})
            col_3_title = "<b style='color:#0EA5E9;'>R* (Hi-Res)</b>"
        else:
            metrics_rstar = meta.get("regions", {}).get("rstar", {})
            col_3_title = "<b style='color:#0EA5E9;'>R* Fusion</b>"
        
        met_col1, met_col2, met_col3 = st.columns(3)
        
        with met_col1:
            st.markdown("**2D Baseline**")
            for r in regions:
                val = metrics_2d.get(r, {}).get("dice", "N/A")
                st.metric(label=f"{r} Dice", value=f"{val:.4f}" if isinstance(val, float) else val)
                
        with met_col2:
            st.markdown("**3D Family**")
            for r in regions:
                val = metrics_3d.get(r, {}).get("dice", "N/A")
                st.metric(label=f"{r} Dice", value=f"{val:.4f}" if isinstance(val, float) else val)
                
        with met_col3:
            st.markdown(col_3_title, unsafe_allow_html=True)
            for r in regions:
                val_rstar = metrics_rstar.get(r, {}).get("dice", "N/A")
                val_2d = metrics_2d.get(r, {}).get("dice", 0.0)
                
                if isinstance(val_rstar, float) and isinstance(val_2d, float):
                    st.metric(label=f"{r} Dice", value=f"{val_rstar:.4f}", delta=f"{(val_rstar - val_2d):.4f} vs 2D")
                else:
                    st.metric(label=f"{r} Dice", value=val_rstar)

if __name__ == "__main__":
    main()