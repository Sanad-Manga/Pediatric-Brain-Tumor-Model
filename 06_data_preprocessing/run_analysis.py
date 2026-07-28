import os
import json
import yaml
import argparse
import pandas as pd
from src.data.dataset_scanner import BraTSDatasetScanner


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pediatric BraTS Dataset Analysis")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["dataset"]["raw_dir"]
    modalities = cfg["dataset"]["modalities"]
    mask_key = cfg["dataset"]["mask_key"]

    os.makedirs("./reports", exist_ok=True)

    print(f"=== Phase 1: Scanning Dataset at {raw_dir} ===")
    scanner = BraTSDatasetScanner(raw_data_dir=raw_dir, modalities=modalities, mask_key=mask_key)
    reports, df_summary = scanner.run_full_scan()

    # Save outputs
    json_path = cfg["analysis"]["output_report_path"]
    csv_path = cfg["analysis"]["summary_csv_path"]

    with open(json_path, "w") as f:
        json.dump(reports, f, indent=2)

    df_summary.to_csv(csv_path, index=False)

    print(f"\nAnalysis complete!")
    print(f"Detailed JSON Report: {json_path}")
    print(f"Summary CSV Report:  {csv_path}\n")

    # Display High-Level Summary Statistics
    print("=== SUMMARY METRICS ===")
    print(f"Total Patients Scanned:     {len(df_summary)}")
    print(f"Valid Patient Scans:       {len(df_summary[df_summary['status'] == 'VALID'])}")
    print(f"Incomplete/Corrupted Scans: {len(df_summary[df_summary['status'] != 'VALID'])}")
    print(f"Unique Shapes Encountered:   {df_summary['shape'].nunique()}")
    print(f"Unique Orientations Found:  {df_summary['orientation'].unique()}")
    print(f"Identical Modality Align:   {df_summary['modalities_aligned'].all()}")
    print("=======================\n")


if __name__ == "__main__":
    main()