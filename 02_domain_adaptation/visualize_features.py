"""Create before/after PCA and LDA plots of model bottleneck embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

MODEL_SECTION = Path(__file__).resolve().parents[1] / "01_model_federated"
sys.path.insert(0, str(MODEL_SECTION))

from src.data import build_dataset  # noqa: E402
from src.model import build_model  # noqa: E402


def extract_features(model, manifest: str, cache_path: str, device: torch.device) -> np.ndarray:
    dataset = build_dataset(manifest, data_mode="real", cache_path=cache_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _, features = model(x.to(device))
            rows.append(features.float().cpu().numpy()[0])
    return np.asarray(rows)


def load_checkpoint_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model = build_model().to(device)
    model.load_state_dict(state)
    return model


def project(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
    lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(scaled, labels)
    return pca, lda


def plot_projection(ax, points, labels, title):
    colors = {"Hospital A": "#2563eb", "Hospital B": "#dc2626", "Held-out": "#16a34a"}
    for label, color in colors.items():
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.7, label=label, c=color)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--hospital-a", required=True)
    parser.add_argument("--hospital-b", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--before-checkpoint", required=True)
    parser.add_argument("--after-checkpoint", required=True)
    parser.add_argument("--output", default="feature_domains.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = [args.hospital_a, args.hospital_b, args.heldout]
    domain_names = ["Hospital A", "Hospital B", "Held-out"]
    counts = [len(json.loads(Path(path).read_text())) for path in manifests]
    labels = np.concatenate([np.repeat(name, count) for name, count in zip(domain_names, counts)])

    projections = {}
    for stage, checkpoint in (
        ("Before adaptation", args.before_checkpoint),
        ("After adaptation", args.after_checkpoint),
    ):
        model = load_checkpoint_model(checkpoint, device)
        features = np.concatenate(
            [extract_features(model, manifest, args.cache_path, device) for manifest in manifests]
        )
        projections[stage] = project(features, labels)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for column, stage in enumerate(("Before adaptation", "After adaptation")):
        pca, lda = projections[stage]
        plot_projection(axes[0, column], pca, labels, f"PCA — {stage}")
        plot_projection(axes[1, column], lda, labels, f"LDA — {stage}")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=3)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

