import pytest
import torch

from manifoldflow import (
    BasinState,
    OnlineBasinMemory,
    latest_version,
    load_checkpoint,
    publish_checkpoint,
    rollback,
)


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


def test_latest_version_is_none_before_any_publish(tmp_path) -> None:
    assert latest_version(tmp_path) is None


def test_publish_checkpoint_starts_at_version_zero_and_increments(tmp_path) -> None:
    memory = _make_memory()

    first = publish_checkpoint(memory, tmp_path)
    second = publish_checkpoint(memory, tmp_path)

    assert first == 0
    assert second == 1
    assert latest_version(tmp_path) == 1
    assert (tmp_path / "v000000.pt").exists()
    assert (tmp_path / "v000001.pt").exists()


def test_load_checkpoint_reconstructs_equivalent_memory(tmp_path) -> None:
    memory = _make_memory()
    memory.update(torch.tensor([[0.2, 0.0], [0.4, 0.0]]), torch.tensor([0, 0]))
    publish_checkpoint(memory, tmp_path)

    loaded = load_checkpoint(tmp_path)

    torch.testing.assert_close(loaded.bank.mu, memory.bank.mu)
    assert loaded.class_count.tolist() == memory.class_count.tolist()


def test_load_checkpoint_without_a_publish_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path)


def test_rollback_repoints_latest_without_touching_version_files(tmp_path) -> None:
    memory = _make_memory()
    publish_checkpoint(memory, tmp_path)  # v0: 1 basin
    memory.update(
        torch.tensor([[10.0, 0.0], [10.1, 0.0], [9.9, 0.0]]),
        torch.tensor([1, 1, 1]),
    )
    publish_checkpoint(memory, tmp_path)  # v1: 2 basins (novelty promoted)
    assert latest_version(tmp_path) == 1
    assert load_checkpoint(tmp_path).bank.mu.shape[0] == 2

    rollback(tmp_path, 0)

    assert latest_version(tmp_path) == 0
    rolled_back = load_checkpoint(tmp_path)
    assert rolled_back.bank.mu.shape[0] == 1
    # The v1 file is untouched, still loadable directly by version.
    assert load_checkpoint(tmp_path, version=1).bank.mu.shape[0] == 2


def test_rollback_to_missing_version_raises(tmp_path) -> None:
    memory = _make_memory()
    publish_checkpoint(memory, tmp_path)

    with pytest.raises(FileNotFoundError):
        rollback(tmp_path, 7)


def test_publish_after_process_restart_continues_the_version_sequence(
    tmp_path,
) -> None:
    # A checkpoint directory is meant to outlive any single process: a new
    # OnlineBasinMemory reloaded from the latest version and then republished
    # must continue numbering from where the prior process left off, not
    # restart at 0 and silently overwrite v0.
    memory = _make_memory()
    publish_checkpoint(memory, tmp_path)  # v0

    reloaded = load_checkpoint(tmp_path)
    reloaded.update(torch.tensor([[0.2, 0.0]]), torch.tensor([0]))
    next_version = publish_checkpoint(reloaded, tmp_path)

    assert next_version == 1
    assert load_checkpoint(tmp_path, version=0).bank.count.tolist() == [0]
