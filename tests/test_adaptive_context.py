import pytest
import torch

from manifoldflow.adaptive_context import AdaptiveBasinContext
from manifoldflow.index import ExactIndex


def _context(**overrides: float) -> AdaptiveBasinContext:
    centers = torch.eye(4)
    defaults: dict[str, float] = {
        "alpha_base": 0.5,
        "decay": 0.9,
        "rbf_bandwidth": 1.0,
        "novelty_threshold": 0.5,
    }
    defaults.update(overrides)
    return AdaptiveBasinContext(centers, ExactIndex(centers), **defaults)


def test_first_call_forces_alpha_one_and_routes_by_raw_observation_alone() -> None:
    context = _context()

    decision = context.route(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    assert decision.routed_id == 0
    assert decision.alpha_used == 1.0
    assert decision.novel is False
    assert torch.equal(context.context, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_agreement_between_raw_and_context_keeps_alpha_at_base() -> None:
    context = _context(alpha_base=0.5)
    context.route(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # establishes context at basin 0

    decision = context.route(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # same basin again

    assert decision.agreement is True
    assert decision.routed_id == 0
    assert decision.alpha_used == pytest.approx(0.5)


def test_confident_conflict_raises_alpha_toward_one() -> None:
    context = _context(alpha_base=0.5)
    context.route(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # context now anchored at basin 0

    # A confident (distance-zero), conflicting observation at basin 1.
    decision = context.route(torch.tensor([0.0, 1.0, 0.0, 0.0]))

    assert decision.agreement is False
    assert decision.raw_confidence == pytest.approx(1.0)
    assert decision.alpha_used == pytest.approx(1.0)
    assert decision.routed_id == 1  # raw observation wins outright


def test_novel_observation_forces_alpha_one_and_fresh_context() -> None:
    context = _context(novelty_threshold=0.8)
    context.route(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    # Equidistant from every basin - low confidence under any of them.
    ambiguous = torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 1.0, 1.0]), dim=0)
    decision = context.route(ambiguous)

    assert decision.novel is True
    assert decision.alpha_used == 1.0
    # Context is overwritten, not blended with the stale prior anchor.
    assert torch.equal(context.context, context.centers[decision.routed_id])


def test_reset_clears_context() -> None:
    context = _context()
    context.route(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    context.reset()

    assert not context._has_context
    assert torch.equal(context.context, torch.zeros(4))


def test_route_rejects_wrong_shape() -> None:
    context = _context()
    with pytest.raises(ValueError):
        context.route(torch.zeros(3))


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha_base": -0.1},
        {"alpha_base": 1.1},
        {"decay": -0.1},
        {"decay": 1.1},
        {"rbf_bandwidth": 0.0},
        {"novelty_threshold": -0.1},
        {"novelty_threshold": 1.1},
    ],
)
def test_constructor_rejects_invalid_parameters(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _context(**overrides)


def test_constructor_rejects_empty_centers() -> None:
    with pytest.raises(ValueError):
        AdaptiveBasinContext(
            torch.zeros(0, 4),
            ExactIndex(torch.eye(4)),
            alpha_base=0.5,
            decay=0.9,
            rbf_bandwidth=1.0,
            novelty_threshold=0.5,
        )
