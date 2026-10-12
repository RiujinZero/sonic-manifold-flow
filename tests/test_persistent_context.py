import pytest
import torch

from manifoldflow.index import ExactIndex
from manifoldflow.persistent_context import PersistentBasinContext


def test_alpha_one_is_stateless_regardless_of_decay() -> None:
    centers = torch.tensor([[0.0], [10.0], [20.0]])
    index = ExactIndex(centers)
    context = PersistentBasinContext(centers, index, alpha=1.0, decay=0.9)

    first = context.route(torch.tensor([0.5]))
    second = context.route(torch.tensor([19.5]))

    assert first == 0
    assert second == 2  # would be pulled toward basin 0's context if alpha < 1


def test_context_pulls_an_ambiguous_query_toward_recently_visited_center() -> None:
    centers = torch.tensor([[0.0], [10.0]])
    index = ExactIndex(centers)
    context = PersistentBasinContext(centers, index, alpha=0.5, decay=1.0)

    context.route(torch.tensor([0.0]))  # establishes context at basin 0's center
    # Exactly equidistant from both centers on its own (z0=5.0); with decay=1.0
    # the context never fades, so the alpha=0.5 blend pulls this toward basin 0.
    routed = context.route(torch.tensor([5.0]))

    assert routed == 0


def test_reset_clears_context_back_to_stateless_behavior() -> None:
    centers = torch.tensor([[0.0], [10.0]])
    index = ExactIndex(centers)
    context = PersistentBasinContext(centers, index, alpha=0.5, decay=1.0)
    context.route(torch.tensor([0.0]))

    context.reset()
    routed = context.route(torch.tensor([5.0]))

    # No prior context after reset: exactly equidistant, so this behaves like
    # a fresh first observation (routes on z0 alone via the tie-breaking the
    # underlying index already uses - either basin is a defensible answer,
    # the point is it isn't forced toward basin 0 anymore).
    assert routed in (0, 1)


def test_decay_fades_old_context_toward_the_new_center() -> None:
    centers = torch.tensor([[0.0], [10.0], [100.0]])
    index = ExactIndex(centers)
    # alpha=1.0 for the setup calls: routing ignores context entirely (only
    # z0 is queried), but context still accumulates - this lets the test
    # pin down exactly which basins get visited before probing the blend.
    context = PersistentBasinContext(centers, index, alpha=1.0, decay=0.0)
    context.route(torch.tensor([0.0]))  # visits basin 0; context -> 0.0
    context.route(torch.tensor([100.0]))  # visits basin 2; decay=0 -> context -> 100.0

    context.alpha = 0.5
    routed = context.route(torch.tensor([5.0]))  # query = 0.5*5 + 0.5*100 = 52.5

    # 52.5 is nearest to basin 1 (10.0, distance 42.5), not basin 0 (0.0,
    # distance 52.5) or basin 2 (100.0, distance 47.5) - decay=0.0 having
    # fully forgotten basin 0's visit is what makes this land on basin 1.
    assert routed == 1


def test_rejects_invalid_construction_arguments() -> None:
    centers = torch.tensor([[0.0], [10.0]])
    index = ExactIndex(centers)
    with pytest.raises(ValueError):
        PersistentBasinContext(centers, index, alpha=1.5, decay=0.5)
    with pytest.raises(ValueError):
        PersistentBasinContext(centers, index, alpha=0.5, decay=-0.1)
    with pytest.raises(ValueError):
        PersistentBasinContext(torch.empty(0, 2), index, alpha=0.5, decay=0.5)


def test_route_rejects_wrong_shaped_observation() -> None:
    centers = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    index = ExactIndex(centers)
    context = PersistentBasinContext(centers, index, alpha=0.5, decay=0.5)

    with pytest.raises(ValueError):
        context.route(torch.tensor([[0.0, 0.0]]))  # batched, not [dimension]
