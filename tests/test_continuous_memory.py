import time

import pytest
import torch

from manifoldflow.continuous_memory import BasinMemory, Source


def _unit(vector: list[float]) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.tensor(vector), dim=-1)


def test_first_write_seeds_a_single_basin() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)

    basin_id = memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    assert basin_id == "0"
    assert memory.basin_count == 1


def test_novel_key_creates_a_new_basin_one_shot() -> None:
    # A single far observation must create a basin immediately - not be
    # staged pending repeated confirmation (SS5.1's one-shot acquisition).
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    basin_id = memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))

    assert basin_id == "1"
    assert memory.basin_count == 2


def test_near_duplicate_key_reuses_the_existing_basin() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    basin_id = memory.write(
        _unit([0.99, 0.01, 0.0, 0.0]), "fact one again", Source("user")
    )

    assert basin_id == "0"
    assert memory.basin_count == 1


def test_read_before_any_write_returns_no_hits() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)

    assert memory.read(_unit([1.0, 0.0, 0.0, 0.0])) == []


def test_read_returns_the_payload_of_the_nearest_basin() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))

    hits = memory.read(_unit([0.9, 0.1, 0.0, 0.0]), top_k=1)

    assert len(hits) == 1
    assert hits[0].payloads == ["fact one"]


def test_read_accumulates_multiple_payloads_on_the_same_basin() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    memory.write(_unit([0.99, 0.01, 0.0, 0.0]), "fact one confirmed", Source("user"))

    hits = memory.read(_unit([1.0, 0.0, 0.0, 0.0]), top_k=1)

    assert hits[0].payloads == ["fact one", "fact one confirmed"]


def test_write_rejects_wrong_shape() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    with pytest.raises(ValueError):
        memory.write(torch.zeros(3), "fact", Source("user"))


def test_write_rejects_empty_payload() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    with pytest.raises(ValueError):
        memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "", Source("user"))


def test_write_rejects_negative_novelty_threshold() -> None:
    with pytest.raises(ValueError):
        BasinMemory(dimension=4, novelty_threshold=-0.1)


def test_write_raises_once_max_basins_capacity_is_reached() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.01, max_basins=1)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    with pytest.raises(RuntimeError):
        memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))


def test_consolidate_returns_bank_version() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    assert memory.consolidate() == 0
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    assert memory.consolidate() == 0


def test_centers_and_index_require_at_least_one_write() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    with pytest.raises(RuntimeError):
        _ = memory.centers
    with pytest.raises(RuntimeError):
        _ = memory.index


def test_centers_and_index_are_usable_after_a_write() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    assert memory.centers.shape == (1, 4)
    assert memory.index.query(_unit([1.0, 0.0, 0.0, 0.0])[None, :], 1).item() == 0


def test_payloads_for_returns_stored_payloads_by_basin_id() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))

    assert memory.payloads_for(0) == ["fact one"]
    assert memory.payloads_for(1) == ["fact two"]
    assert memory.payloads_for(99) == []


def test_advance_and_reset_context_are_harmless_no_ops() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    memory.advance_context(_unit([1.0, 0.0, 0.0, 0.0]), session_id="s1")
    memory.reset_context("s1")

    assert memory.basin_count == 1


def test_write_rejects_a_non_durable_trust_tier() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    with pytest.raises(ValueError):
        memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("generated"))


@pytest.mark.parametrize(
    "trust_tier", ["user", "corrected", "imported", "verified", "confirmed"]
)
def test_write_accepts_every_durable_trust_tier(trust_tier: str) -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    basin_id = memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source(trust_tier))
    assert basin_id == "0"


def test_new_basin_starts_as_a_recent_microbasin() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5, promotion_threshold=2)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    assert memory.tier(0) == "recent"


def test_repeated_reinforcement_promotes_to_consolidated() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5, promotion_threshold=2)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    memory.write(_unit([0.99, 0.01, 0.0, 0.0]), "fact one confirmed", Source("user"))

    assert memory.tier(0) == "consolidated"


def test_supersede_marks_the_old_record_inactive() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.01)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "Mira is in Box 4", Source("user"))

    memory.write(
        _unit([0.0, 1.0, 0.0, 0.0]),
        "Mira is in Box 9",
        Source("corrected"),
        supersedes=(0, 0),
    )

    assert memory.payloads_for(0) == []  # old record no longer active
    assert memory.payloads_for(1) == ["Mira is in Box 9"]


def test_read_excludes_superseded_records() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.01)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "Mira is in Box 4", Source("user"))
    memory.write(
        _unit([0.0, 1.0, 0.0, 0.0]),
        "Mira is in Box 9",
        Source("corrected"),
        supersedes=(0, 0),
    )

    hits = memory.read(_unit([1.0, 0.0, 0.0, 0.0]), top_k=2)

    assert all("Mira is in Box 4" not in hit.payloads for hit in hits)


def test_write_rejects_an_invalid_supersedes_target() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    with pytest.raises(ValueError):
        memory.write(
            _unit([0.0, 1.0, 0.0, 0.0]),
            "fact two",
            Source("corrected"),
            supersedes=(0, 5),  # no such record index
        )


def test_active_records_excludes_superseded_entries() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.01)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "Mira is in Box 4", Source("user"))
    memory.write(
        _unit([0.0, 1.0, 0.0, 0.0]),
        "Mira is in Box 9",
        Source("corrected"),
        supersedes=(0, 0),
    )

    assert memory.active_records(0) == []
    remaining = memory.active_records(1)
    assert len(remaining) == 1
    assert remaining[0][1].text == "Mira is in Box 9"


def test_expire_stale_microbasins_only_affects_never_promoted_basins() -> None:
    memory = BasinMemory(
        dimension=4,
        novelty_threshold=0.01,
        promotion_threshold=2,
        microbasin_max_age=1,
    )
    memory.write(
        _unit([1.0, 0.0, 0.0, 0.0]), "recent, never reinforced", Source("user")
    )
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "consolidated fact", Source("user"))
    memory.write(
        _unit([0.0, 0.99, 0.01, 0.0]), "consolidated fact confirmed", Source("user")
    )
    # Age both basins past microbasin_max_age with unrelated writes.
    memory.write(_unit([0.0, 0.0, 1.0, 0.0]), "distractor a", Source("user"))
    memory.write(_unit([0.0, 0.0, 0.0, 1.0]), "distractor b", Source("user"))

    expired = memory.expire_stale_microbasins()

    assert 0 in expired  # never reinforced, still "recent" -> expired
    assert memory.payloads_for(0) == []
    assert memory.tier(1) == "consolidated"
    assert memory.payloads_for(1) != []  # promoted basins are not expired


def test_expire_stale_microbasins_is_a_no_op_when_nothing_is_stale() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5, microbasin_max_age=50)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    assert memory.expire_stale_microbasins() == []
    assert memory.payloads_for(0) == ["fact one"]


def test_prune_expired_basins_removes_and_renumbers() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.01, microbasin_max_age=1)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "stale", Source("user"))
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "other one", Source("user"))
    memory.write(_unit([0.0, 0.0, 1.0, 0.0]), "other two", Source("user"))
    memory.expire_stale_microbasins()

    mapping = memory.prune_expired_basins()

    assert memory.basin_count == 2
    assert mapping == {1: 0, 2: 1}
    assert memory.payloads_for(mapping[1]) == ["other one"]
    assert memory.payloads_for(mapping[2]) == ["other two"]


def test_prune_expired_basins_is_a_no_op_when_nothing_expired() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    assert memory.prune_expired_basins() == {}
    assert memory.basin_count == 1


def test_prune_expired_basins_handles_multiple_stale_basins() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5, microbasin_max_age=1)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))
    # Two more writes advance the step counter enough that both prior
    # basins are strictly past microbasin_max_age (but not the newest one).
    memory.write(_unit([0.0, 0.0, 1.0, 0.0]), "fact three", Source("user"))
    memory.write(_unit([0.0, 0.0, 0.0, 1.0]), "fact four", Source("user"))
    memory.expire_stale_microbasins()

    memory.prune_expired_basins()

    assert memory.basin_count == 2  # only facts three and four are recent enough


def test_merge_compatible_recent_basins_combines_close_microbasins() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.0)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact a", Source("user"))
    memory.write(_unit([0.99, 0.02, 0.0, 0.0]), "fact a near-dup", Source("user"))
    memory.write(_unit([0.0, 0.0, 1.0, 0.0]), "fact b unrelated", Source("user"))

    mapping = memory.merge_compatible_recent_basins(0.01)

    assert memory.basin_count == 2
    merged_id = mapping[0]
    assert mapping[1] == merged_id
    assert memory.payloads_for(merged_id) == ["fact a", "fact a near-dup"]
    assert memory.tier(merged_id) == "consolidated"  # 2 active records now


def test_merge_never_touches_consolidated_basins() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5, promotion_threshold=1)
    # promotion_threshold=1 means every basin below is immediately consolidated.
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact a", Source("user"))
    memory.write(_unit([0.0, 0.0, 1.0, 0.0]), "fact b", Source("user"))

    assert memory.tier(0) == "consolidated"
    assert memory.tier(1) == "consolidated"
    assert memory.merge_compatible_recent_basins(4.0) == {}
    assert memory.basin_count == 2


def test_merge_is_a_no_op_when_nothing_is_close_enough() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.0)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact a", Source("user"))
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact b", Source("user"))

    assert memory.merge_compatible_recent_basins(0.001) == {}
    assert memory.basin_count == 2


def test_merge_rejects_negative_threshold() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))
    with pytest.raises(ValueError):
        memory.merge_compatible_recent_basins(-0.1)


def test_consolidate_prunes_and_merges_then_bumps_version() -> None:
    memory = BasinMemory(
        dimension=4,
        novelty_threshold=0.0,
        merge_threshold=0.01,
        microbasin_max_age=1,
    )
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "stale", Source("user"))
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact a", Source("user"))
    memory.write(_unit([0.0, 0.99, 0.02, 0.0]), "fact a near-dup", Source("user"))
    memory.expire_stale_microbasins()

    version = memory.consolidate()

    # Two real rebuilds happened (prune, then merge), so the version
    # advances by 2, not 1 - each structural operation is its own
    # atomic snapshot-and-swap, not a single combined step.
    assert version == 2
    assert memory.basin_count == 1  # stale pruned, fact a + near-dup merged
    assert memory.payloads_for(0) == ["fact a", "fact a near-dup"]


def test_consolidate_returns_the_current_version_when_nothing_to_do() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    assert memory.consolidate() == 0
    assert memory.basin_count == 1


def test_from_bulk_constructs_one_basin_per_row() -> None:
    centers = torch.nn.functional.normalize(torch.eye(3), dim=-1)
    payloads = ["fact a", "fact b", "fact c"]

    memory = BasinMemory.from_bulk(
        centers, payloads, Source("imported"), novelty_threshold=0.5
    )

    assert memory.basin_count == 3
    for basin_id, payload in enumerate(payloads):
        assert memory.payloads_for(basin_id) == [payload]
    hits = memory.read(centers[1], top_k=1)
    assert hits[0].payloads == ["fact b"]


def test_from_bulk_rejects_mismatched_payload_count() -> None:
    centers = torch.nn.functional.normalize(torch.eye(3), dim=-1)
    with pytest.raises(ValueError):
        BasinMemory.from_bulk(centers, ["only one"], Source("imported"))


def test_from_bulk_rejects_a_non_durable_trust_tier() -> None:
    centers = torch.nn.functional.normalize(torch.eye(3), dim=-1)
    with pytest.raises(ValueError):
        BasinMemory.from_bulk(centers, ["a", "b", "c"], Source("generated"))


def test_write_raises_while_consolidation_lock_is_held() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    memory._consolidation_lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))
    finally:
        memory._consolidation_lock.release()

    # The lock is released again afterward, so writes work normally.
    memory.write(_unit([0.0, 1.0, 0.0, 0.0]), "fact two", Source("user"))
    assert memory.basin_count == 2


def test_consolidate_in_background_returns_a_thread_that_finishes() -> None:
    memory = BasinMemory(dimension=4, novelty_threshold=0.5)
    memory.write(_unit([1.0, 0.0, 0.0, 0.0]), "fact one", Source("user"))

    thread = memory.consolidate_in_background()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert memory.basin_count == 1


def test_concurrent_reads_succeed_while_background_consolidation_runs() -> None:
    # Enough close-together "recent" basins that merge takes long enough
    # (order of a second) to reliably overlap with concurrent reads issued
    # from the main thread - a real concurrency test, not just a check
    # that nothing crashes when called before/after.
    generator = torch.Generator().manual_seed(0)
    base = torch.nn.functional.normalize(torch.randn(8, generator=generator), dim=0)
    centers = torch.nn.functional.normalize(
        base + 0.01 * torch.randn(1500, 8, generator=generator), dim=-1
    )
    payloads = [f"fact {i}" for i in range(1500)]
    memory = BasinMemory.from_bulk(
        centers,
        payloads,
        Source("imported"),
        novelty_threshold=0.0,
        merge_threshold=1.0,
        promotion_threshold=1_000_000,  # keep every basin "recent" this pass
    )

    thread = memory.consolidate_in_background()
    read_count = 0
    read_errors: list[str] = []
    started = time.perf_counter()
    while thread.is_alive() and time.perf_counter() - started < 10.0:
        try:
            hits = memory.read(centers[0], top_k=1)
            if not hits:
                read_errors.append("empty result")
        except Exception as exc:  # pragma: no cover - failure path only
            read_errors.append(str(exc))
        else:
            read_count += 1
    thread.join(timeout=10)

    assert not thread.is_alive(), "consolidation did not finish in time"
    assert read_errors == []
    assert read_count > 0, "no read overlapped with the background consolidation"
