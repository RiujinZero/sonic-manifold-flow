import pytest
import torch

from manifoldflow.world_identity import TemporalLabelEvidence


def test_temporal_label_resists_single_conflicting_frame() -> None:
    evidence = TemporalLabelEvidence(("amber", "violet"))
    for _ in range(4):
        label, _ = evidence.update(torch.tensor([0.9, 0.1]), 1.0)

    label, _ = evidence.update(torch.tensor([0.05, 0.95]), 1.0)

    assert label == "amber"


def test_temporal_label_can_change_after_sustained_counterevidence() -> None:
    evidence = TemporalLabelEvidence(("amber", "violet"), decay=0.5)
    evidence.update(torch.tensor([0.9, 0.1]), 1.0)

    for _ in range(8):
        label, _ = evidence.update(torch.tensor([0.05, 0.95]), 1.0)

    assert label == "violet"


def test_temporal_label_abstains_on_ambiguous_evidence() -> None:
    evidence = TemporalLabelEvidence(("amber", "violet"))

    label, confidence = evidence.update(torch.tensor([0.51, 0.49]), 1.0)

    assert label == ""
    assert confidence == pytest.approx(0.51)
