import json

import pytest
import torch

from manifoldflow import extract_sharded_latents, load_sharded_latents
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.training import batches_from_tensors, collect_latents


def _fixture(rows: int = 17, batch_size: int = 4):
    generator = torch.Generator().manual_seed(11)
    image = torch.rand(rows, 1, 28, 28, generator=generator)
    label = torch.arange(rows) % 3
    model = CompactImageAutoencoder(input_channels=1, input_size=28, latent_dim=6)
    model.freeze()
    batches = batches_from_tensors(image, label, batch_size=batch_size)
    return model, batches, image, label


def test_extract_sharded_latents_matches_monolithic_collection(tmp_path) -> None:
    model, batches, _, _ = _fixture()
    expected = collect_latents(model, batches, device="cpu")

    manifest = extract_sharded_latents(
        model, batches, output_dir=tmp_path, shard_size=5, device="cpu"
    )

    assert manifest["total_rows"] == expected.latent.shape[0]
    latent, label = load_sharded_latents(tmp_path)
    torch.testing.assert_close(latent, expected.latent)
    torch.testing.assert_close(label, expected.label)


def test_shard_files_hold_at_most_shard_size_rows(tmp_path) -> None:
    model, batches, _, _ = _fixture(rows=17, batch_size=4)

    manifest = extract_sharded_latents(
        model, batches, output_dir=tmp_path, shard_size=5, device="cpu"
    )

    for index in range(manifest["shard_count"]):
        payload = torch.load(
            tmp_path / f"shard-{index:06d}.pt", weights_only=True
        )
        assert payload["latent"].shape[0] <= 5


def test_extract_sharded_latents_resumes_without_recomputing_finished_shards(
    tmp_path,
) -> None:
    model, batches, _, _ = _fixture(rows=20, batch_size=4)
    expected = collect_latents(model, batches, device="cpu")

    interrupted = batches[:2]  # simulate a run killed after two batches (8 rows)
    first_manifest = extract_sharded_latents(
        model, interrupted, output_dir=tmp_path, shard_size=5, device="cpu"
    )

    resumed_manifest = extract_sharded_latents(
        model, batches, output_dir=tmp_path, shard_size=5, device="cpu"
    )

    # 8 rows in: one full 5-row shard plus one trailing 3-row shard.
    assert first_manifest["total_rows"] == 8
    assert resumed_manifest["total_rows"] == expected.latent.shape[0]
    latent, label = load_sharded_latents(tmp_path)
    torch.testing.assert_close(latent, expected.latent)
    torch.testing.assert_close(label, expected.label)


def test_resume_with_a_different_batch_size_trims_the_straddling_batch(
    tmp_path,
) -> None:
    # Same underlying rows, but the resumed call rechunks them into a
    # different batch size, so the skip point falls mid-batch rather than on
    # a batch boundary - this is what actually exercises the offset-trim
    # path (resuming with the same batch size always lands on a boundary).
    model, batches_of_4, image, label = _fixture(rows=20, batch_size=4)
    expected = collect_latents(model, batches_of_4, device="cpu")

    extract_sharded_latents(
        model, batches_of_4[:2], output_dir=tmp_path, shard_size=5, device="cpu"
    )
    batches_of_6 = batches_from_tensors(image, label, batch_size=6)

    resumed_manifest = extract_sharded_latents(
        model, batches_of_6, output_dir=tmp_path, shard_size=5, device="cpu"
    )

    assert resumed_manifest["total_rows"] == expected.latent.shape[0]
    latent, label_out = load_sharded_latents(tmp_path)
    torch.testing.assert_close(latent, expected.latent)
    torch.testing.assert_close(label_out, expected.label)


def test_manifest_stays_in_sync_with_shards_after_a_mid_run_crash(tmp_path) -> None:
    # Simulate a real process kill (not a truncated input iterable): the
    # encoder itself raises partway through, after some shards have already
    # been flushed to disk. The manifest must reflect exactly those
    # completed shards, not be stale or missing, so a resume knows the true
    # row count instead of silently redoing already-finished work.
    model, batches, _, _ = _fixture(rows=20, batch_size=4)
    expected = collect_latents(model, batches, device="cpu")

    real_encode = model.encode
    calls = 0

    def flaky_encode(image):
        nonlocal calls
        calls += 1
        if calls == 3:  # crash partway through the third batch
            raise RuntimeError("simulated kill")
        return real_encode(image)

    model.encode = flaky_encode
    with pytest.raises(RuntimeError, match="simulated kill"):
        extract_sharded_latents(
            model, batches, output_dir=tmp_path, shard_size=5, device="cpu"
        )
    model.encode = real_encode

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    on_disk_shards = len(list(tmp_path.glob("shard-*.pt")))
    assert manifest["shard_count"] == on_disk_shards
    assert manifest["shard_count"] > 0  # some progress really was saved

    resumed_manifest = extract_sharded_latents(
        model, batches, output_dir=tmp_path, shard_size=5, device="cpu"
    )

    assert resumed_manifest["total_rows"] == expected.latent.shape[0]
    latent, label = load_sharded_latents(tmp_path)
    torch.testing.assert_close(latent, expected.latent)
    torch.testing.assert_close(label, expected.label)


def test_load_sharded_latents_requires_a_manifest(tmp_path) -> None:
    try:
        load_sharded_latents(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing manifest should raise")
