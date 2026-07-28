import os
import glob
import numpy as np
import torch
import nibabel as nib
from typing import Dict, List, Any
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
)


class MONAIBraTSPreprocessor:
    """Production 2D Preprocessing Pipeline for Pediatric BraTS MRI using MONAI.

    Performs:
    1. Canonical reorientation to RAS
    2. Non-zero Z-score intensity normalization per modality
    3. Occupancy-based 2D slice extraction for Axial and Coronal views
    4. Multi-modal packaging into [4, H, W] npz arrays
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modalities = config["dataset"]["modalities"]
        self.mask_key = config["dataset"]["mask_key"]
        
        # MONAI Transformation Chain for 3D Volume Standardization
        self.transforms = Compose([
            LoadImaged(keys=self.modalities + [self.mask_key]),
            EnsureChannelFirstd(keys=self.modalities + [self.mask_key]),
            Orientationd(keys=self.modalities + [self.mask_key], axcodes="RAS"),
            NormalizeIntensityd(
                keys=self.modalities, 
                nonzero=True, 
                channel_wise=True
            ),
        ])

    def _get_patient_files(self, patient_dir: str) -> Dict[str, str]:
        patient_id = os.path.basename(patient_dir)
        files = {}
        for k in self.modalities + [self.mask_key]:
            p1 = os.path.join(patient_dir, f"{k}.nii.gz")
            p2 = os.path.join(patient_dir, f"{patient_id}-{k}.nii.gz")
            if os.path.exists(p1):
                files[k] = p1
            elif os.path.exists(p2):
                files[k] = p2
            else:
                matches = glob.glob(os.path.join(patient_dir, f"*{k}.nii.gz"))
                if matches:
                    files[k] = matches[0]
        return files

    def process_patient(self, patient_dir: str, output_base_dir: str) -> Dict[str, Any]:
        patient_id = os.path.basename(patient_dir)
        files_dict = self._get_patient_files(patient_dir)

        # Apply MONAI Standardization Transforms
        data_dict = self.transforms(files_dict)

        # Stack Modalities into Single Tensor [4, D, H, W]
        mod_tensors = [data_dict[m] for m in self.modalities]
        image_4d = torch.cat(mod_tensors, dim=0).numpy() # Shape: [4, X, Y, Z]
        mask_3d = data_dict[self.mask_key].numpy()[0]    # Shape: [X, Y, Z]

        # Setup Patient Output Directories
        patient_out_dir = os.path.join(output_base_dir, patient_id)
        axial_dir = os.path.join(patient_out_dir, "axial")
        coronal_dir = os.path.join(patient_out_dir, "coronal")
        os.makedirs(axial_dir, exist_ok=True)
        os.makedirs(coronal_dir, exist_ok=True)

        extracted_counts = {"axial": 0, "coronal": 0}
        min_brain_occ = self.config["preprocessing"]["occupancy_thresholds"]["min_brain_occupancy"]

        # ------------------- AXIAL SLICE EXTRACTION (Axis Z = 2) -------------------
        num_axial = mask_3d.shape[2]
        for z in range(num_axial):
            img_slice = image_4d[:, :, :, z]  # [4, H, W]
            mask_slice = mask_3d[:, :, z]     # [H, W]

            brain_voxels = np.sum(img_slice[0] != 0)
            tumor_voxels = np.sum(mask_slice > 0)
            total_area = mask_slice.size
            brain_occ = brain_voxels / total_area

            # Slicing condition: Brain >= 5% OR Has Tumor
            if brain_occ >= min_brain_occ or tumor_voxels > 0:
                save_path = os.path.join(axial_dir, f"slice_{z:03d}.npz")
                np.savez_compressed(
                    save_path,
                    image=img_slice.astype(np.float32),
                    mask=mask_slice.astype(np.uint8)[None, ...] # [1, H, W]
                )
                extracted_counts["axial"] += 1

        # ------------------- CORONAL SLICE EXTRACTION (Axis Y = 1) -------------------
        num_coronal = mask_3d.shape[1]
        for y in range(num_coronal):
            img_slice = image_4d[:, :, y, :]  # [4, H, W]
            mask_slice = mask_3d[:, y, :]     # [H, W]

            brain_voxels = np.sum(img_slice[0] != 0)
            tumor_voxels = np.sum(mask_slice > 0)
            total_area = mask_slice.size
            brain_occ = brain_voxels / total_area

            if brain_occ >= min_brain_occ or tumor_voxels > 0:
                save_path = os.path.join(coronal_dir, f"slice_{y:03d}.npz")
                np.savez_compressed(
                    save_path,
                    image=img_slice.astype(np.float32),
                    mask=mask_slice.astype(np.uint8)[None, ...] # [1, H, W]
                )
                extracted_counts["coronal"] += 1

        return {
            "patient_id": patient_id,
            "axial_slices": extracted_counts["axial"],
            "coronal_slices": extracted_counts["coronal"]
        }