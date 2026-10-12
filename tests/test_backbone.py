import pytest
import torch

from manifoldflow import Frozen224Backbone


def test_frozen_backbone_extracts_expected_feature_width() -> None:
    backbone = Frozen224Backbone(projected_dim=32, pretrained=False)
    image = torch.rand(4, 3, 224, 224)

    features = backbone.extract_features(image)

    assert features.shape == (4, 512)


def test_frozen_backbone_is_frozen() -> None:
    backbone = Frozen224Backbone(projected_dim=32, pretrained=False)

    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_frozen_backbone_rejects_invalid_projected_dim() -> None:
    with pytest.raises(ValueError):
        Frozen224Backbone(projected_dim=0, pretrained=False)
    with pytest.raises(ValueError):
        Frozen224Backbone(projected_dim=513, pretrained=False)


def test_encode_requires_fit_projection_first() -> None:
    backbone = Frozen224Backbone(projected_dim=16, pretrained=False)
    image = torch.rand(2, 3, 224, 224)

    with pytest.raises(RuntimeError):
        backbone.encode(image)


def test_fit_projection_then_encode_produces_projected_dimension() -> None:
    backbone = Frozen224Backbone(projected_dim=16, pretrained=False)
    calibration_features = torch.randn(64, 512)
    backbone.fit_projection(calibration_features)

    image = torch.rand(3, 3, 224, 224)
    encoded = backbone.encode(image)

    assert encoded.shape == (3, 16)
    assert bool(backbone.is_fitted)


def test_fit_projection_rejects_undersized_calibration_sample() -> None:
    backbone = Frozen224Backbone(projected_dim=16, pretrained=False)

    with pytest.raises(ValueError):
        backbone.fit_projection(torch.randn(8, 512))


def test_extract_features_rejects_wrong_image_shape() -> None:
    backbone = Frozen224Backbone(projected_dim=16, pretrained=False)

    with pytest.raises(ValueError):
        backbone.extract_features(torch.rand(2, 3, 32, 32))
