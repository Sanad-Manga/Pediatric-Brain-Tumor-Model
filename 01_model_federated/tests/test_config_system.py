import copy

import pytest
import torch

import run
from src.config import load_config
from src.model import build_model
from src.train_single import _build_loss


def test_default_model_parameter_count_matches_documented_size():
    model = build_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 4_800_000 <= parameter_count <= 5_000_000


def test_wide_model_has_about_sixteen_times_the_parameters():
    default_count = sum(parameter.numel() for parameter in build_model().parameters())
    wide_count = sum(parameter.numel() for parameter in build_model(width=64, depth=5).parameters())
    assert wide_count > default_count * 12
    assert wide_count < default_count * 20


def test_small_model_forward_shapes_and_deepcopy_architecture():
    model = build_model(width=4, depth=3)
    logits, features = model(torch.randn(1, 4, 16, 16, 16))
    assert logits.shape == (1, 5, 16, 16, 16)
    assert features.shape == (1, 256)

    copied = copy.deepcopy(model)
    assert copied._init_args[-2:] == (4, 3)
    assert sum(parameter.numel() for parameter in copied.parameters()) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_class_weights_change_loss_and_region_loss_rejects_weights():
    torch.manual_seed(7)
    logits = torch.randn(1, 5, 4, 4, 4)
    labels = torch.randint(0, 5, (1, 1, 4, 4, 4))
    plain_loss = _build_loss("dice_ce")(logits, labels)
    weighted_loss = _build_loss("dice_ce", [0.2, 3.0, 1.0, 1.0, 1.0])(logits, labels)
    assert not torch.isclose(plain_loss, weighted_loss)

    with pytest.raises(ValueError, match="class_weights"):
        _build_loss("region_dice_bce", [0.2, 3.0, 1.0, 1.0, 1.0])


def test_resolve_settings_defaults_config_and_explicit_cli_precedence():
    parser = run.build_arg_parser()
    no_config_args = parser.parse_args([])
    defaults = run.resolve_settings(no_config_args, None)
    assert defaults["lr"] == 1e-3
    assert defaults["loss"] == "dice_ce"
    assert defaults["lr_horizon"] is None
    assert defaults["modality_dropout"] == 0.0
    assert defaults["sequence_shift"] == 0.0
    assert defaults["sequence_shift_max_voxels"] == 1.2

    cfg = {
        "loss": {"kind": "dice_focal", "class_weights": None},
        "schedule": {"kind": "cosine", "min_lr": 1e-5},
        "augmentation": {"modality_dropout_prob": 0.4},
    }
    absent_flag = run.resolve_settings(parser.parse_args([]), cfg)
    assert absent_flag["loss"] == "dice_focal"
    assert absent_flag["modality_dropout"] == 0.4

    explicit_default = run.resolve_settings(parser.parse_args(["--loss", "dice_ce"]), cfg)
    assert explicit_default["loss"] == "dice_ce"


def test_config_loader_rejects_invalid_values(tmp_path):
    cases = [
        ("loss:\n  kind: unknown\n", "loss.kind"),
        ("model:\n  depth: 1\n", "model.depth"),
        ("loss:\n  class_weights: [1, 2, 3, 4]\n", "class_weights"),
    ]
    for index, (contents, message) in enumerate(cases):
        path = tmp_path / f"bad_{index}.yaml"
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_config(path)


def test_config_file_has_expected_defaults():
    cfg = load_config(run.__file__.replace("run.py", "config.yaml"))
    assert cfg["model"] == {"width": 16, "depth": 5}
    assert cfg["loss"]["class_weights"] is None
    assert cfg["schedule"]["kind"] == "none"
    assert "seed" not in cfg["augmentation"]


def test_parser_exposes_legacy_defaults_and_tracks_explicit_default_flag():
    parser = run.build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.loss == "dice_ce"
    assert defaults.lr == 1e-3
    assert defaults.modality_dropout == 0.0
    assert defaults.sequence_shift == 0.0
    assert defaults.sequence_shift_max_voxels == 1.2

    explicit = parser.parse_args(["--loss", "dice_ce"])
    assert "loss" in explicit._explicit_cli


def test_checkpoint_stores_non_default_model_dimensions(tmp_path):
    from src.checkpoint import load_checkpoint, save_checkpoint

    save_checkpoint(
        str(tmp_path), "wide", epoch=0, model_state={}, optimizer_state={},
        model_width=64, model_depth=5,
    )
    checkpoint = load_checkpoint(str(tmp_path), "wide")
    assert checkpoint["model_config"] == {"width": 64, "depth": 5}