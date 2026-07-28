import os
import yaml
import json
import argparse
from tqdm import tqdm
from src.preprocessing.monai_pipeline import MONAIBraTSPreprocessor


def is_patient_already_processed(patient_out_dir: str) -> bool:
    """Checks if a patient folder already contains processed axial and coronal slices."""
    axial_dir = os.path.join(patient_out_dir, "axial")
    coronal_dir = os.path.join(patient_out_dir, "coronal")
    
    if os.path.exists(axial_dir) and os.path.exists(coronal_dir):
        has_axial = len([f for f in os.listdir(axial_dir) if f.endswith(".npz")]) > 0
        has_coronal = len([f for f in os.listdir(coronal_dir) if f.endswith(".npz")]) > 0
        return has_axial and has_coronal
    return False


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Run 2D MONAI Preprocessing Pipeline (With Resume Support)")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["dataset"]["raw_dir"]
    output_dir = cfg["dataset"]["processed_dir"]

    os.makedirs(output_dir, exist_ok=True)

    patient_folders = [
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, f)) and not f.startswith(".")
    ]

    print(f"=== Phase 4: Executing MONAI 2D Preprocessing Pipeline ===")
    print(f"Input Dataset:  {raw_dir}")
    print(f"Output Dataset: {output_dir}")
    print(f"Total Patients: {len(patient_folders)}\n")

    preprocessor = MONAIBraTSPreprocessor(cfg)
    summary_logs = []

    for p_dir in tqdm(patient_folders, desc="Processing Patients"):
        p_id = os.path.basename(p_dir)
        patient_out_dir = os.path.join(output_dir, p_id)

        # Smart Resume Check: Skip if already extracted
        if is_patient_already_processed(patient_out_dir):
            continue

        res = preprocessor.process_patient(p_dir, output_dir)
        summary_logs.append(res)

    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_logs, f, indent=2)

    print("\nPreprocessing Completed Successfully!")
    print(f"Summary log saved at: {summary_file}")


if __name__ == "__main__":
    main()