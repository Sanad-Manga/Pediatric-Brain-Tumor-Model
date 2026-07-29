"""Training loop for the 2D segmentation model.

Section 01 owns the production model and the federated loop. This is the
centralised trainer section 03 needs to produce a checkpoint its own ablation
runner can score -- it deliberately does not implement FedAvg.

Three things it does take seriously, because getting them wrong invalidates the
ablation:

* **The validation split is by patient**, taken from the training hospitals only.
  Splitting slices would put adjacent, nearly identical slices of one patient on
  both sides and report a meaningless Dice.
* **Validation is never augmented**, and is scored per patient from restacked
  volumes -- the same code path `eval` uses, so the numbers are comparable.
* **A checkpoint is written every epoch** to ``checkpoints/<run_id>/``, and
  training resumes from it. Colab disconnects (CONTRACTS.md).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader

from .config import Config
from .dataset import EvalSubjectDataset, PatientBalancedBatchSampler, SliceDataset
from .evaluate import evaluate_subject
from .metrics import REGION_ORDER, aggregate
from .model import TumorTypeHead, build_model
from .plan import build_plans
from .tumor_type import build_type_index, label_index, read_type_index, write_type_index

LOGGER = logging.getLogger(__name__)


def split_by_patient(entries, val_fraction: float, seed: int):
    """Hold out whole patients for validation -- never individual slices."""
    subjects = sorted({e["source_subject_id"] for e in entries})
    rng = np.random.default_rng(seed)
    order = [subjects[i] for i in rng.permutation(len(subjects))]

    n_val = max(1, int(round(val_fraction * len(subjects))))
    if n_val >= len(subjects):
        raise ValueError(
            f"val_fraction {val_fraction} leaves no training subjects "
            f"({len(subjects)} available)"
        )
    val = set(order[:n_val])

    train_entries = [e for e in entries if e["source_subject_id"] not in val]
    val_subjects = sorted(val)
    assert not ({e["source_subject_id"] for e in train_entries} & val), "patient leaked into val"
    return train_entries, val_subjects


def collate(batch):
    return {
        "image": torch.from_numpy(np.stack([b["image"] for b in batch])),
        "label": torch.from_numpy(np.stack([b["label"] for b in batch])),
        "plane": [b["plane"] for b in batch],
        "hospital": [b["hospital"] for b in batch],
        "source_subject_id": [b["source_subject_id"] for b in batch],
    }


def build_type_lookup(cfg: Config, cache_dir, subjects, plane: str = "axial") -> dict[str, int]:
    """``{subject_id: type index}`` for the tumour-type auxiliary head.

    Computed once, from the geometric proxy in `tumor_type.py` -- reconstructing
    each subject's mask volume from its axial slices, which has been verified
    exact against the real cache. Cached to ``<cache_dir>/tumor_type.json`` so
    resuming or restarting training (e.g. after a Colab disconnect) doesn't
    re-read every subject's slices again -- that reconstruction is slow over a
    mounted Drive filesystem.
    """
    try:
        index = read_type_index(cache_dir)
        if set(index["subjects"]) >= set(subjects):
            return {sid: label_index(v["stratum"]) for sid, v in index["subjects"].items()
                    if sid in subjects}
    except FileNotFoundError:
        pass
    index = build_type_index(cache_dir, subjects, cfg, plane=plane, progress_every=0)
    write_type_index(index, cache_dir)
    return {sid: label_index(v["stratum"]) for sid, v in index["subjects"].items()}


@torch.no_grad()
def validate(model, subjects, cfg: Config, cache_dir, device="cpu",
            type_head=None, type_lookup=None) -> dict:
    """Per-patient Dice on held-out *patients*, unaugmented -- the eval path.

    When a tumour-type head is given, also reports its accuracy against the
    geometric proxy on the same validation patients. This is diagnostic only --
    it never enters `results/ablation_results.csv`, whose schema is fixed by
    CONTRACTS.md.
    """
    model.eval()
    dataset = EvalSubjectDataset(subjects, cfg, cache_dir=cache_dir)

    by_subject: dict[str, list] = {}
    for i in range(len(dataset)):
        try:
            item = dataset[i]
        except FileNotFoundError:
            continue
        by_subject.setdefault(item["subject_id"], []).append(item)

    scores = []
    type_correct = type_total = 0
    for subject_id, items in by_subject.items():
        s, _true, _planes = evaluate_subject(model, items, cfg, device)
        scores.append(s)

        if type_head is not None and type_lookup is not None and subject_id in type_lookup:
            image = torch.as_tensor(items[0]["images"], dtype=torch.float32).to(device)
            _logits, features = model(image)
            type_logits = type_head(features).mean(dim=0, keepdim=True)
            pred = int(type_logits.argmax(dim=1).item())
            type_correct += int(pred == type_lookup[subject_id])
            type_total += 1

    out = aggregate(scores) if scores else {f"dice_{r}": 0.0 for r in REGION_ORDER}
    if type_total:
        out["type_accuracy"] = type_correct / type_total
    return out


def save_checkpoint(path: Path, model, optimizer, epoch: int, best: float, cfg: Config,
                    type_head=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": "unet",
        "epoch": epoch,
        "best_mean_dice": best,
        "spatial_dims": getattr(model, "spatial_dims", 2),
        # Record the geometry: a checkpoint trained at one width cannot be
        # rebuilt from a config that has since changed to another.
        "width": cfg.model_width,
        "depth": cfg.model_depth,
        "use_augmentation": cfg.use_augmentation,
        "use_mixup": cfg.use_mixup,
    }
    if type_head is not None:
        payload["type_head_state_dict"] = type_head.state_dict()
    torch.save(payload, path)
    return path


def run(
    cfg: Config,
    run_id: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_fraction: float = 0.25,
    max_steps_per_epoch: int | None = None,
    cache_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    device: str = "cpu",
    seed: int = 1337,
    resume: bool = True,
) -> dict:
    """Train, validating per patient each epoch. Returns the run summary."""
    torch.manual_seed(seed)
    cache_dir = Path(cache_dir) if cache_dir else cfg.resolve("cache_2d")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (cfg.root / "checkpoints" / run_id)

    # Hospitals are pooled here on purpose: this is the centralised baseline.
    # Keeping them apart is federation, which is section 01's.
    plans = build_plans(cfg, cache_dir)
    entries = [e for p in plans.values() for e in p["entries"]]
    train_entries, val_subjects = split_by_patient(entries, val_fraction, seed)

    LOGGER.info("train: %d entries over %d patients | val: %d patients",
                len(train_entries),
                len({e["source_subject_id"] for e in train_entries}),
                len(val_subjects))

    train_ds = SliceDataset(train_entries, cfg, cache_dir=cache_dir)
    sampler = PatientBalancedBatchSampler(train_entries, batch_size, cfg=cfg, seed=seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)

    model = build_model(cfg, spatial_dims=2).to(device)
    # Class-weighted CE. ET is the smallest and worst-scoring region, so a
    # uniform loss lets the network trade it away for the easier bulk regions.
    # Weights apply to the CE term only; the Dice term is already region-wise.
    _w = cfg.class_weights
    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, include_background=False,
        weight=torch.tensor(_w, dtype=torch.float32, device=device) if _w else None,
    )
    if _w:
        LOGGER.info("class-weighted CE: %s", _w)

    # Tumour-type auxiliary head: reads `features` from the shared model
    # interface without changing model(x)'s signature. Trained against the
    # geometric dmg_like/astrocytoma_like proxy so the network has to actually
    # extract location/enhancement information, not be handed the label.
    type_head = type_lookup = type_loss_fn = None
    all_subjects = sorted({e["source_subject_id"] for e in entries})
    if cfg.tumor_type_enabled:
        LOGGER.info("classifying %d subjects for the tumour-type proxy", len(all_subjects))
        type_lookup = build_type_lookup(cfg, cache_dir, all_subjects)
        example = train_ds[0]["image"]
        with torch.no_grad():
            _, feat = model(torch.as_tensor(example, dtype=torch.float32).unsqueeze(0).to(device))
        type_head = TumorTypeHead(in_features=feat.shape[1]).to(device)
        type_loss_fn = torch.nn.CrossEntropyLoss()

    params = list(model.parameters()) + (list(type_head.parameters()) if type_head else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    # Fixed lr made validation Dice oscillate late in the run; decay it so the
    # last epochs can settle instead of stepping over the minimum.
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=cfg.min_lr)
        LOGGER.info("cosine LR decay: %.1e -> %.1e over %d epochs", lr, cfg.min_lr, epochs)

    start_epoch, best, best_mean = 0, -1.0, -1.0
    last_path = ckpt_dir / "last.pt"
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if type_head is not None and "type_head_state_dict" in payload:
            type_head.load_state_dict(payload["type_head_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best = float(payload.get("best_mean_dice", -1.0))
        LOGGER.info("resumed from %s at epoch %d", last_path, start_epoch)
        for _ in range(start_epoch):        # keep the LR curve aligned on resume
            if scheduler is not None:
                scheduler.step()

    # Carry forward any history from a previous run of this run_id, so resuming
    # after a disconnect doesn't silently truncate the curve to post-resume
    # epochs only.
    history_path = ckpt_dir / "history.json"
    history = []
    if resume and history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as fh:
                history = [r for r in json.load(fh).get("history", [])
                           if int(r.get("epoch", -1)) < start_epoch]
        except (json.JSONDecodeError, OSError):
            LOGGER.warning("could not read %s; starting a fresh history", history_path)

    def write_history(done: bool) -> None:
        """Persist the curve after every epoch.

        Writing only at the end loses the entire history when a run is stopped
        early -- which is the normal case on a preemptible GPU or a Colab
        session, not an edge case.
        """
        payload = {"run_id": run_id, "epochs": epochs,
                   "epochs_completed": len(history), "completed": done,
                   "best_mean_dice": round(best, 4), "val_subjects": val_subjects,
                   "train_entries": len(train_entries), "history": history,
                   "checkpoint_dir": str(ckpt_dir)}
        tmp = history_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(history_path)   # atomic: never leave a half-written curve

    for epoch in range(start_epoch, epochs):
        model.train()
        if type_head is not None:
            type_head.train()
        losses, type_losses, t0 = [], [], time.time()

        for step, batch in enumerate(loader):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits, features = model(image)
            loss = loss_fn(logits, label)

            if type_head is not None:
                # Every slice in a batch inherits its patient's proxy label --
                # the label is per-patient, the head is trained per-slice.
                known = [i for i, sid in enumerate(batch["source_subject_id"])
                        if sid in type_lookup]
                if known:
                    targets = torch.tensor(
                        [type_lookup[batch["source_subject_id"][i]] for i in known],
                        dtype=torch.long, device=device,
                    )
                    type_logits = type_head(features[known])
                    type_loss = type_loss_fn(type_logits, targets)
                    loss = loss + cfg.tumor_type_loss_weight * type_loss
                    type_losses.append(float(type_loss.detach()))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

            if step and step % 20 == 0:
                LOGGER.info("  epoch %d step %d loss %.4f (%.1fs/step)",
                            epoch, step, float(np.mean(losses[-20:])),
                            (time.time() - t0) / (step + 1))

        val = validate(model, val_subjects, cfg, cache_dir, device,
                       type_head=type_head, type_lookup=type_lookup)
        mean_dice = float(np.mean([val[f"dice_{r}"] for r in REGION_ORDER]))
        record = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
                  "seconds": round(time.time() - t0, 1), "mean_dice": round(mean_dice, 4),
                  **{k: round(v, 4) for k, v in val.items()}}
        if type_losses:
            record["type_loss"] = round(float(np.mean(type_losses)), 4)
        history.append(record)

        line = (f"epoch {epoch}: loss={record['loss']:.4f}  val "
               + "  ".join(f"{r}={val[f'dice_{r}']:.3f}" for r in REGION_ORDER)
               + f"  mean={mean_dice:.3f}")
        if "type_accuracy" in val:
            line += f"  type_acc={val['type_accuracy']:.3f}"
        print(line + f"  ({record['seconds']:.0f}s)")

        # Two selection criteria, both recorded. `mean` can be carried by one
        # strong region while another collapses -- that is exactly how a
        # checkpoint 0.165 worse at ET was once promoted over a better one.
        # `min_region` scores the weakest region and cannot be gamed that way.
        min_region = min(val[f"dice_{r}"] for r in REGION_ORDER)
        record["min_region_dice"] = round(min_region, 4)
        score = min_region if cfg.selection_metric == "min_region" else mean_dice

        save_checkpoint(last_path, model, optimizer, epoch, max(best, score), cfg, type_head)
        if score > best:
            best = score
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best, cfg, type_head)
            LOGGER.info("  new best by %s: %.4f (epoch %d)", cfg.selection_metric, best, epoch)
        if mean_dice > best_mean:
            best_mean = mean_dice
            save_checkpoint(ckpt_dir / "best_mean.pt", model, optimizer, epoch, best_mean,
                            cfg, type_head)
        if scheduler is not None:
            scheduler.step()
            record["lr"] = round(optimizer.param_groups[0]["lr"], 8)
        write_history(done=False)

    write_history(done=True)
    with open(history_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
