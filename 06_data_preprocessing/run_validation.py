import os
import yaml
import numpy as np
from tqdm import tqdm


def validate_processed_dataset(config_path: str = "config/pipeline_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    proc_dir = cfg["dataset"]["processed_dir"]
    print(f"=== Quality Control & Dataset Validation ===")
    print(f"Validating dataset at: {proc_dir}\n")

    patients = [f for f in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, f))]
    
    total_axial_slices = 0
    total_coronal_slices = 0
    errors = []

    for p_id in tqdm(patients, desc="Validating Patients"):
        p_path = os.path.join(proc_dir, p_id)
        
        for view in ["axial", "coronal"]:
            view_dir = os.path.join(p_path, view)
            if not os.path.exists(view_dir):
                errors.append(f"{p_id}: Missing {view} folder")
                continue

            slices = [f for f in os.listdir(view_dir) if f.endswith(".npz")]
            if len(slices) == 0:
                errors.append(f"{p_id}: Zero slices extracted in {view}")
                continue

            if view == "axial":
                total_axial_slices += len(slices)
            else:
                total_coronal_slices += len(slices)

            # Sample test the first slice in each directory
            sample_file = os.path.join(view_dir, slices[0])
            data = np.load(sample_file)
            img = data["image"]  # Expected shape: [4, H, W]
            mask = data["mask"]  # Expected shape: [1, H, W]

            if img.ndim != 3 or img.shape[0] != 4:
                errors.append(f"{p_id} ({view}/{slices[0]}): Invalid image shape {img.shape}")
            if mask.ndim != 3 or mask.shape[0] != 1:
                errors.append(f"{p_id} ({view}/{slices[0]}): Invalid mask shape {mask.shape}")
            if np.isnan(img).any() or np.isnan(mask).any():
                errors.append(f"{p_id} ({view}/{slices[0]}): Contains NaN values")

    print("\n=== VALIDATION REPORT SUMMARY ===")
    print(f"Total Patients Verified: {len(patients)}")
    print(f"Total Axial 2D Slices:   {total_axial_slices}")
    print(f"Total Coronal 2D Slices: {total_coronal_slices}")
    print(f"Total Combined 2D Slices:{total_axial_slices + total_coronal_slices}")
    print(f"Integrity Errors Found:  {len(errors)}")
    
    if errors:
        print("\nERRORS DETECTED:")
        for err in errors[:10]:
            print(f"- {err}")
    else:
        print("\nSTATUS: ALL CHECKS PASSED PERFECTLY! DATASET IS READY FOR MODEL TRAINING.")


if __name__ == "__main__":
    validate_processed_dataset()