import streamlit as st
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from components.theme import apply_theme # Uncomment and use if a specific global theme is required

# Streamlit page configuration
st.set_page_config(page_title="Model Comparison", page_icon="🔍", layout="wide")

# Directory setup
BASE_DIR = Path(__file__).resolve().parents[1]
COMPARISON_CACHE = BASE_DIR / "comparison_cache"
DEMO_CACHE = BASE_DIR / "demo_cache"

# Segmentation Colors Map
# 0: Background (Transparent)
# 1: Enhancing Tumor - ET (Red)
# 2: Non-Enhancing Core - NETC (Blue)
# 3: Cystic Component - CC (Green)
# 4: Peritumoral Edema - ED (Yellow)
COLORS = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.6],
    [0.0, 0.0, 1.0, 0.6],
    [0.0, 1.0, 0.0, 0.6],
    [1.0, 1.0, 0.0, 0.6]
]
CMAP = ListedColormap(COLORS)

def load_patient_list():
    """Retrieve all patients available in the comparison cache."""
    if not COMPARISON_CACHE.exists():
        return []
    return [d.name for d in COMPARISON_CACHE.iterdir() if d.is_dir()]

def load_patient_data(patient_id):
    """Load metadata and label arrays for a specific patient."""
    patient_dir = COMPARISON_CACHE / patient_id
    
    with open(patient_dir / "meta.json", "r") as f:
        meta = json.load(f)
        
    lbl_2d = np.load(patient_dir / "labels_2d.npz")["labels"]
    lbl_3d = np.load(patient_dir / "labels_3d.npz")["labels"]
    lbl_rstar = np.load(patient_dir / "labels_rstar.npz")["labels"]
    
    return meta, lbl_2d, lbl_3d, lbl_rstar

def get_background_slice(patient_id, slice_idx):
    """Attempt to load the MRI background slice from the demo cache."""
    bg_path = DEMO_CACHE / patient_id / "axial" / f"slice_{slice_idx}.npz"
    if bg_path.exists():
        data = np.load(bg_path)
        # Fallback to the first available sequence if t1c is not directly accessible
        return data["t1c"] if "t1c" in data else data[data.files[0]]
    return None

def main():
    st.title("🔍 Multi-Model Segmentation Comparison")
    st.markdown("""
    Evaluate the **2D**, **3D**, and combined **R*** models side-by-side. 
    Select a patient from the dataset to visualize axial slices and review performance metrics.
    """)
    
    patients = load_patient_list()
    if not patients:
        st.error("Comparison cache is missing. Please verify the `comparison_cache/` directory.")
        return
        
    selected_patient = st.selectbox("Select Patient Record 👤", sorted(patients))
    
    if selected_patient:
        meta, lbl_2d, lbl_3d, lbl_rstar = load_patient_data(selected_patient)
        
        # Guardrails and Status Badges
        status = meta.get("rstar_status", "unknown")
        warnings = meta.get("rstar_warnings", [])
        
        if status == "ok" and not warnings:
            st.success("✅ **Quality Check:** Passed. The R* preprocessing pipeline encountered no issues.")
        else:
            st.warning(f"⚠️ **R* Status:** {status.upper()}")
            for w in warnings:
                st.error(f"🚨 **Analysis Warning:** {w}")
        
        st.divider()
        
        # Visualization Section
        st.subheader("🧠 Volumetric Slice Visualization")
        
        # Controls layout
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            max_slice = lbl_rstar.shape[2] - 1
            slice_idx = st.slider("Navigate Axial Slice (Z-Axis)", min_value=0, max_value=max_slice, value=max_slice // 2)
        with ctrl_col2:
            st.write("") # Vertical spacing alignment
            show_mask = st.toggle("Show Tumor Segmentation Overlay", value=True)
            
        bg_img = get_background_slice(selected_patient, slice_idx)
        
        if bg_img is None:
            st.info("ℹ️ **Notice:** Original MRI background scans are restricted to demo patients only. Displaying segmentation masks on a dark background.")
        
        # Render the 3 Models
        img_col1, img_col2, img_col3 = st.columns(3)
        model_configs = [
            ("2D Architecture", lbl_2d, img_col1),
            ("3D Architecture", lbl_3d, img_col2),
            ("R* (Combined Approach)", lbl_rstar, img_col3)
        ]
        
        for model_name, lbl_vol, col in model_configs:
            with col:
                st.markdown(f"#### {model_name}")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.axis('off')
                fig.patch.set_facecolor('black') 
                
                # Render Background
                if bg_img is not None:
                    ax.imshow(bg_img, cmap='gray')
                else:
                    # Provide a black canvas if no MRI background is available
                    ax.imshow(np.zeros((lbl_vol.shape[0], lbl_vol.shape[1])), cmap='gray')
                
                # Render Segmentation Overlay
                if show_mask:
                    slice_mask = lbl_vol[:, :, slice_idx]
                    masked_data = np.ma.masked_where(slice_mask == 0, slice_mask)
                    ax.imshow(masked_data, cmap=CMAP, interpolation='none', vmin=0, vmax=4)
                    
                st.pyplot(fig, transparent=True)
                plt.close(fig)

        st.divider()
        
        # Metrics Dashboard
        st.subheader("📊 Dynamic Performance Metrics")
        st.caption("Dice similarity coefficient per model for Enhancing Tumor (ET), Tumor Core (NC), and Whole Tumor (WT).")
        
        metrics_data = meta.get("regions", {})
        m_col1, m_col2, m_col3 = st.columns(3)
        
        metric_configs = [
            ("2d", m_col1, "2D Model Performance"),
            ("3d", m_col2, "3D Model Performance"),
            ("rstar", m_col3, "R* Model Performance")
        ]
        
        for dict_key, column, display_title in metric_configs:
            with column:
                st.markdown(f"**{display_title}**")
                model_metrics = metrics_data.get(dict_key, {})
                
                for region in ["ET", "NC", "WT"]:
                    region_metrics = model_metrics.get(region, {})
                    dice_score = region_metrics.get("dice", "N/A")
                    
                    # Formatting numerical layout elegantly
                    if isinstance(dice_score, float):
                        st.metric(label=f"{region} Dice Score", value=f"{dice_score:.4f}")
                    else:
                        st.metric(label=f"{region} Dice Score", value=dice_score)

if __name__ == "__main__":
    main()