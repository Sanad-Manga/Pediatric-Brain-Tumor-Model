"""The optional nnU-Net member (SPEC Addendum 4): hash-verified loading and axis handling.

`nnunetv2` is imported only here and only when the option is on, so `import rstar` and the default recipe do not need it.
The network is rebuilt with nnU-Net's standard trainer class rather than the class named in the checkpoint: the model was
trained with `nnUNetTrainer_save5` (the default trainer saving more often), which exists only in the training environment
and does not change the architecture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import RStarConfig, load_manifest
from .models import _resolve

SPACING = [1.0, 1.0, 1.0]  # BraTS grid; the contract guarantees 1 mm isotropic


def load_nnunet(cfg: RStarConfig):
    """Returns (predictor, info) for the manifest's 'nnunet' entry. The checkpoint hash is verified before it is opened."""
    manifest = load_manifest(cfg.manifest_path)
    if "nnunet" not in manifest:
        raise ValueError("nnunet=True but the models manifest has no 'nnunet' entry")
    entry = manifest["nnunet"]
    items = ([dict(c, name=f"{entry['name']}[{Path(c['path']).parent.name}]") for c in entry["checkpoints"]]
             if "checkpoints" in entry else [entry])                  # several folds (SPEC Addendum 4c) or one
    ckpt_paths = [_resolve(item, cfg) for item in items]  # every SHA-256 checked (unless verify_hashes=False) before any load
    try:
        import torch
        from batchgenerators.utilities.file_and_folder_operations import load_json
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    except ImportError as exc:
        raise ImportError("nnunet=True needs the nnunetv2 package: pip install nnunetv2==2.8.1") from exc

    model_dir = Path(ckpt_paths[0]).parent.parent
    dataset_json = load_json(str(model_dir / "dataset.json"))
    plans_manager = PlansManager(load_json(str(model_dir / "plans.json")))
    checkpoints = [torch.load(p, map_location="cpu", weights_only=False) for p in ckpt_paths]
    checkpoint = checkpoints[0]
    configuration_manager = plans_manager.get_configuration(checkpoint["init_args"]["configuration"])
    network = nnUNetTrainer.build_network_architecture(
        plans_manager, configuration_manager,
        determine_num_input_channels(plans_manager, configuration_manager, dataset_json),
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False)
    device = torch.device(cfg.resolved_device())
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, device=device,
                                perform_everything_on_device=device.type == "cuda", verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans_manager, configuration_manager, [c["network_weights"] for c in checkpoints],
                                    dataset_json, checkpoint["trainer_name"],
                                    checkpoint.get("inference_allowed_mirroring_axes"))
    return predictor, [dict(item, resolved_path=str(p)) for item, p in zip(items, ckpt_paths)]


def nnunet_probs(predictor, volume: np.ndarray, present) -> np.ndarray:
    """(4, X, Y, Z) raw intensities -> (5, X, Y, Z) float32 probabilities. Absent channels are passed as zeros.
    nnU-Net works on arrays in the order its NIfTI reader produces, (C, Z, Y, X), so the axes are reversed in and out."""
    x = np.array(volume, dtype=np.float32, copy=True)
    for c, is_present in enumerate(present):
        if not is_present:
            x[c] = 0.0
    image = np.ascontiguousarray(x.transpose(0, 3, 2, 1))
    _, probs = predictor.predict_single_npy_array(image, {"spacing": list(SPACING)}, None, None, True)
    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape[1:] != image.shape[1:]:
        raise RuntimeError(f"nnU-Net returned probabilities of shape {probs.shape} for an input of shape {image.shape}")
    return np.ascontiguousarray(probs.transpose(0, 3, 2, 1))
