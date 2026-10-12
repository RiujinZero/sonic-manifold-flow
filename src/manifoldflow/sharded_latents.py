"""Sharded, restartable latent extraction for large image streams.

`training.collect_latents` accumulates one monolithic in-memory tensor, which
is fine for the small MVP datasets but not for an ImageNet-shaped stream: a
run that is interrupted partway loses all progress, and the whole latent set
has to fit in RAM at once. This writes fixed-size shards to disk as they are
produced and, on a fresh call against an existing output directory, skips the
rows already covered by completed shards - so a killed run can resume rather
than recompute finished work, and no single monolithic artifact is required.

`batches` must be a deterministically-ordered iterable (a non-shuffled
loader) for resume to skip the same rows it originally encoded; a shuffled
training loader would resume onto a different subset than it left off on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn


def _write_manifest(manifest_path: Path, manifest: dict[str, int]) -> None:
    """Write the manifest atomically so a kill mid-write can't corrupt it."""
    tmp_path = manifest_path.parent / (manifest_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, manifest_path)


def extract_sharded_latents(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    output_dir: str | Path,
    shard_size: int,
    device: torch.device | str,
) -> dict[str, int]:
    """Encode a batched image stream into resumable, fixed-size latent shards.

    Each shard is written to `output_dir/shard-{index:06d}.pt` holding
    `{"latent": Tensor, "label": Tensor}`. `output_dir/manifest.json` records
    the completed shard count, shard size, and total row count; a rerun of
    this function against the same `output_dir` reads that manifest and skips
    the rows already covered by finished shards before it starts encoding.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    completed_shards = 0
    total_rows = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed_shards = int(manifest["shard_count"])
        total_rows = int(manifest["total_rows"])

    model.to(device)
    model.eval()
    # Skip by rows already written, not `completed_shards * shard_size`: the
    # final shard of a prior call can be smaller than shard_size (a trailing
    # partial flush), so shard count alone would under- or over-skip.
    skip_rows = total_rows
    seen_rows = 0
    shard_index = completed_shards
    new_total_rows = total_rows
    latent_buffer: list[Tensor] = []
    label_buffer: list[Tensor] = []
    buffered_rows = 0

    def flush(latent: Tensor, label: Tensor) -> None:
        nonlocal shard_index, new_total_rows
        torch.save(
            {"latent": latent.contiguous(), "label": label.contiguous()},
            output_dir / f"shard-{shard_index:06d}.pt",
        )
        new_total_rows += latent.shape[0]
        shard_index += 1
        # Update the manifest after every shard, not just at the end: if this
        # process is killed mid-run, shard files with no matching manifest
        # entry would otherwise be silently redone from scratch on resume.
        _write_manifest(
            manifest_path,
            {
                "shard_count": shard_index,
                "shard_size": shard_size,
                "total_rows": new_total_rows,
            },
        )

    with torch.no_grad():
        for image, label in batches:
            batch_rows = image.shape[0]
            if seen_rows + batch_rows <= skip_rows:
                seen_rows += batch_rows
                continue
            if seen_rows < skip_rows:
                offset = skip_rows - seen_rows
                image = image[offset:]
                label = label[offset:]
            seen_rows += batch_rows

            latent = model.encode(image.to(device)).cpu()
            latent_buffer.append(latent)
            label_buffer.append(label.to(dtype=torch.long, device="cpu"))
            buffered_rows += latent.shape[0]

            while buffered_rows >= shard_size:
                shard_latent = torch.cat(latent_buffer)
                shard_label = torch.cat(label_buffer)
                flush(shard_latent[:shard_size], shard_label[:shard_size])
                remainder_latent = shard_latent[shard_size:]
                remainder_label = shard_label[shard_size:]
                buffered_rows = remainder_latent.shape[0]
                latent_buffer = [remainder_latent] if buffered_rows else []
                label_buffer = [remainder_label] if buffered_rows else []

    if buffered_rows > 0:
        flush(torch.cat(latent_buffer), torch.cat(label_buffer))

    return {
        "shard_count": shard_index,
        "shard_size": shard_size,
        "total_rows": new_total_rows,
    }


def load_sharded_latents(output_dir: str | Path) -> tuple[Tensor, Tensor]:
    """Load and concatenate every shard written by `extract_sharded_latents`."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latent_parts = []
    label_parts = []
    for index in range(int(manifest["shard_count"])):
        payload = torch.load(
            output_dir / f"shard-{index:06d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        latent_parts.append(payload["latent"])
        label_parts.append(payload["label"])
    if not latent_parts:
        raise ValueError("manifest has no shards")
    return torch.cat(latent_parts).contiguous(), torch.cat(label_parts).contiguous()
