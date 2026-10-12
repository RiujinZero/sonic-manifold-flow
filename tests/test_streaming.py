import torch

from manifoldflow import (
    BasinState,
    ExactMainDeltaIndex,
    NoveltyStager,
    OnlineBasinMemory,
    calibrate_novelty_threshold,
)


def test_main_delta_index_overrides_stale_centers_and_adds_new_ids() -> None:
    index = ExactMainDeltaIndex(torch.tensor([[0.0], [10.0]]))
    index.upsert(torch.tensor([0, 2]), torch.tensor([[20.0], [5.0]]))

    ids = index.query(torch.tensor([[4.0]]), k=3)

    assert ids.tolist() == [[2, 1, 0]]
    assert index.main_size == 2
    assert index.delta_size == 2
    assert index.size == 3


def test_main_delta_rebuild_clears_overrides() -> None:
    index = ExactMainDeltaIndex(torch.tensor([[0.0], [10.0]]))
    index.upsert(torch.tensor([0]), torch.tensor([[2.0]]))

    index.rebuild(torch.tensor([[2.0], [10.0]]))

    assert index.delta_size == 0
    assert index.query(torch.tensor([[1.9]]), 1).item() == 0


def test_main_delta_index_serializable_round_trips_query_behavior() -> None:
    index = ExactMainDeltaIndex(torch.tensor([[0.0], [10.0]]))
    index.upsert(torch.tensor([0, 2]), torch.tensor([[20.0], [5.0]]))

    restored = ExactMainDeltaIndex.from_serialized(index.serializable())

    assert restored.main_size == index.main_size
    assert restored.delta_size == index.delta_size
    assert restored.size == index.size
    query = torch.tensor([[4.0], [19.0], [9.0]])
    torch.testing.assert_close(
        restored.query(query, k=2), index.query(query, k=2)
    )


def test_non_iid_updates_touch_only_the_assigned_basin_and_delta() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
    untouched_center = bank.mu[1].clone()
    memory = OnlineBasinMemory(
        bank,
        novelty_threshold=4.0,
        novelty_support_radius=0.5,
        max_basins=4,
    )

    result = memory.update(torch.tensor([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]))

    assert result.basin_ids.tolist() == [0, 0, 0]
    assert not result.novel.any()
    assert bank.count.tolist() == [3, 0]
    torch.testing.assert_close(bank.mu[0], torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(bank.mu[1], untouched_center)
    assert memory.index.delta_size == 1


def test_persistent_novel_cluster_creates_one_valued_basin() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0, 0.0]]))
    memory = OnlineBasinMemory(
        bank,
        novelty_threshold=4.0,
        novelty_persistence=3,
        novelty_support_radius=1.0,
        max_basins=2,
        class_count=torch.tensor([[5, 0]]),
    )

    result = memory.update(
        torch.tensor([[10.0, 0.0], [10.2, 0.0], [9.8, 0.0]]),
        torch.tensor([1, 1, 1]),
    )

    assert result.basin_ids.tolist() == [-1, -1, 1]
    assert result.created.tolist() == [False, False, True]
    assert bank.mu.shape[0] == 2
    assert bank.count.tolist() == [0, 3]
    torch.testing.assert_close(bank.mu[1], torch.tensor([10.0, 0.0]))
    assert memory.class_count is not None
    assert memory.class_count.tolist() == [[5, 0], [0, 3]]
    assert memory.index.query(torch.tensor([[10.1, 0.0]]), 1).item() == 1
    assert memory.predict_nearest_value(torch.tensor([[10.1, 0.0]])).item() == 1
    assert (
        memory.predict_rbf_value(
            torch.tensor([[10.1, 0.0]]), top_k=2, temperature=0.1
        ).item()
        == 1
    )

    memory.rebuild_main_index()
    assert memory.index.delta_size == 0
    assert memory.predict_nearest_value(torch.tensor([[10.1, 0.0]])).item() == 1


def test_single_novel_outlier_stays_staged() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0]]))
    memory = OnlineBasinMemory(
        bank,
        novelty_threshold=1.0,
        novelty_persistence=3,
        novelty_support_radius=0.5,
        max_basins=2,
    )

    result = memory.update(torch.tensor([[8.0]]))

    assert result.basin_ids.item() == -1
    assert result.novel.item()
    assert not result.created.item()
    assert memory.stager.pending_count == 1
    assert bank.mu.shape[0] == 1


def test_capacity_guard_rejects_persistent_novel_cluster() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0]]))
    memory = OnlineBasinMemory(
        bank,
        novelty_threshold=1.0,
        novelty_persistence=2,
        novelty_support_radius=0.5,
        max_basins=1,
    )

    result = memory.update(torch.tensor([[8.0], [8.1]]))

    assert result.capacity_rejected.tolist() == [False, True]
    assert bank.mu.shape[0] == 1


def test_class_aware_staging_does_not_merge_different_labels() -> None:
    stager = NoveltyStager(
        dimension=1,
        persistence=2,
        support_radius=10.0,
        classes=2,
    )

    assert stager.observe(torch.tensor([4.0]), label=0) is None
    assert stager.observe(torch.tensor([4.1]), label=1) is None

    assert stager.pending_count == 2
    proposal = stager.observe(torch.tensor([4.2]), label=0)
    assert proposal is not None
    assert proposal.class_count is not None
    assert proposal.class_count.tolist() == [2, 0]


def _make_memory() -> OnlineBasinMemory:
    bank = BasinState.from_centers(torch.tensor([[0.0, 0.0]]))
    return OnlineBasinMemory(
        bank,
        novelty_threshold=4.0,
        novelty_persistence=3,
        novelty_support_radius=1.0,
        max_basins=3,
        class_count=torch.tensor([[5, 0]]),
    )


def test_online_basin_memory_serializable_round_trips_state() -> None:
    memory = _make_memory()
    memory.update(
        torch.tensor([[0.2, 0.0], [10.0, 0.0], [10.1, 0.0]]),
        torch.tensor([0, 1, 1]),
    )

    restored = OnlineBasinMemory.from_serialized(memory.serializable())

    torch.testing.assert_close(restored.bank.mu, memory.bank.mu)
    torch.testing.assert_close(restored.bank.raw_rate, memory.bank.raw_rate)
    torch.testing.assert_close(restored.bank.m2, memory.bank.m2)
    assert restored.bank.count.tolist() == memory.bank.count.tolist()
    assert restored.class_count.tolist() == memory.class_count.tolist()
    assert restored.stager.pending_count == memory.stager.pending_count
    query = torch.tensor([[0.1, 0.0], [10.0, 0.0]])
    assert (
        restored.index.query(query, 1).tolist() == memory.index.query(query, 1).tolist()
    )


def test_restart_after_checkpoint_matches_an_uninterrupted_run() -> None:
    # The real question a "restart verification" gate has to answer: does
    # continuing from a reloaded checkpoint produce the same future as never
    # having stopped? This stream exercises all three kinds of state a
    # checkpoint has to capture - winner updates, still-pending novelty, and
    # a novelty proposal promoted to a new basin after reload - so a gap in
    # any one of them (bank, delta index, stager) would show up as drift.
    first_half = torch.tensor([[0.2, 0.0], [10.0, 0.0], [10.1, 0.0]])
    first_labels = torch.tensor([0, 1, 1])
    second_half = torch.tensor([[9.9, 0.0], [0.4, 0.0], [10.0, 0.0]])
    second_labels = torch.tensor([1, 0, 1])

    uninterrupted = _make_memory()
    uninterrupted.update(first_half, first_labels)
    uninterrupted.update(second_half, second_labels)

    interrupted = _make_memory()
    interrupted.update(first_half, first_labels)
    restored = OnlineBasinMemory.from_serialized(interrupted.serializable())
    restored.update(second_half, second_labels)

    torch.testing.assert_close(restored.bank.mu, uninterrupted.bank.mu)
    torch.testing.assert_close(restored.bank.raw_rate, uninterrupted.bank.raw_rate)
    torch.testing.assert_close(restored.bank.m2, uninterrupted.bank.m2)
    assert restored.bank.count.tolist() == uninterrupted.bank.count.tolist()
    assert restored.class_count.tolist() == uninterrupted.class_count.tolist()
    assert restored.stager.pending_count == uninterrupted.stager.pending_count == 0
    assert restored.bank.mu.shape[0] == 2  # the staged basin was created either way
    query = torch.tensor([[0.3, 0.0], [10.0, 0.0]])
    assert (
        restored.index.query(query, 1).tolist()
        == uninterrupted.index.query(query, 1).tolist()
    )


def test_novelty_threshold_uses_known_latent_quantile() -> None:
    latent = torch.tensor([[0.0], [1.0], [2.0], [9.0]])
    centers = torch.tensor([[0.0], [10.0]])

    threshold = calibrate_novelty_threshold(
        latent, centers, quantile=0.75, chunk_size=2
    )

    torch.testing.assert_close(torch.tensor(threshold), torch.tensor(1.75))
