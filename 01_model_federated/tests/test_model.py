import torch

from src.model import FederatedUNet3D, build_model


def test_model_output_shapes():
    model = build_model()
    x = torch.randn(1, 4, 96, 96, 96)
    seg_logits, features = model(x)
    assert seg_logits.shape == (1, 5, 96, 96, 96)
    assert features.dim() == 2
    assert features.shape[0] == 1


def test_model_has_no_pretrained_loading_path():
    # FederatedUNet3D exposes no method/argument for loading pretrained weights;
    # its only weight source is random init at construction time.
    model = build_model()
    assert not hasattr(model, "load_pretrained")
    assert isinstance(model, FederatedUNet3D)
