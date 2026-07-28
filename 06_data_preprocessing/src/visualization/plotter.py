import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_patient_occupancy_curves(patient_report: Dict[str, Any], save_path: str = None):
    """Plots brain tissue and tumor occupancy curves across Axial, Coronal, and Sagittal planes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    patient_id = patient_report["patient_id"]

    views = [("axial_occupancy", "Axial View (Z-axis)"), 
             ("coronal_occupancy", "Coronal View (Y-axis)"), 
             ("sagittal_occupancy", "Sagittal View (X-axis)")]

    for idx, (key, title) in enumerate(views):
        ax = axes[idx]
        brain_occ = patient_report[key]["brain"]
        tumor_occ = patient_report[key]["tumor"]
        slices = np.arange(len(brain_occ))

        ax.plot(slices, brain_occ, label="Brain Tissue Area %", color="navy", linewidth=2)
        ax.plot(slices, tumor_occ, label="Tumor Area %", color="crimson", linewidth=2)
        ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.7, label="Brain Cutoff (5%)")
        
        ax.set_title(f"{patient_id} - {title}")
        ax.set_xlabel("Slice Index")
        ax.set_ylabel("Occupancy Ratio")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_triplanar_slice_with_overlay(t1c_vol: np.ndarray, seg_vol: np.ndarray, 
                                     axial_idx: int, coronal_idx: int, sagittal_idx: int,
                                     patient_id: str = ""):
    """Displays 2D slice with segmentation mask overlay across Axial, Coronal, and Sagittal views."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial
    axes[0].imshow(t1c_vol[:, :, axial_idx].T, cmap="gray", origin="lower")
    axes[0].imshow(np.ma.masked_where(seg_vol[:, :, axial_idx].T == 0, seg_vol[:, :, axial_idx].T), 
                   cmap="jet", alpha=0.5, origin="lower")
    axes[0].set_title(f"Axial (Slice {axial_idx})")
    axes[0].axis("off")

    # Coronal
    axes[1].imshow(t1c_vol[:, coronal_idx, :].T, cmap="gray", origin="lower")
    axes[1].imshow(np.ma.masked_where(seg_vol[:, coronal_idx, :].T == 0, seg_vol[:, coronal_idx, :].T), 
                   cmap="jet", alpha=0.5, origin="lower")
    axes[1].set_title(f"Coronal (Slice {coronal_idx})")
    axes[1].axis("off")

    # Sagittal
    axes[2].imshow(t1c_vol[sagittal_idx, :, :].T, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(seg_vol[sagittal_idx, :, :].T == 0, seg_vol[sagittal_idx, :, :].T), 
                   cmap="jet", alpha=0.5, origin="lower")
    axes[2].set_title(f"Sagittal (Slice {sagittal_idx})")
    axes[2].axis("off")

    plt.suptitle(f"Triplanar Inspection: {patient_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()