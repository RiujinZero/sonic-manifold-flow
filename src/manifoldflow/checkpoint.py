"""Versioned bank/value/delta checkpointing with atomic publish and rollback.

Design §5 separates structural consolidation (merge/split/prune/quantize -
still Phase D, not implemented here) from the persistence substrate a
consolidator publishes through: snapshot the current version, write the next
version to a temporary path, and atomically publish it only once fully
written, keeping every prior version on disk so a bad publish can roll back
to the last good one. This module provides that substrate for
`OnlineBasinMemory` - not the merge/split/prune decision logic itself.

Each version is one immutable file (`v{N:06d}.pt`, bundling bank, value,
and delta-index state via `OnlineBasinMemory.serializable()`); `latest.json`
is a small pointer file swapped atomically to publish or roll back. Prior
version files are never modified or deleted by this module, so rollback is
just repointing `latest.json` at a version file that is still there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from manifoldflow.streaming import OnlineBasinMemory


def _version_path(checkpoint_dir: Path, version: int) -> Path:
    return checkpoint_dir / f"v{version:06d}.pt"


def _write_pointer(checkpoint_dir: Path, version: int) -> None:
    pointer_path = checkpoint_dir / "latest.json"
    tmp_path = checkpoint_dir / "latest.json.tmp"
    tmp_path.write_text(
        json.dumps({"version": version}, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, pointer_path)


def latest_version(checkpoint_dir: str | Path) -> int | None:
    """Return the currently published version, or None if none exists yet."""
    pointer_path = Path(checkpoint_dir) / "latest.json"
    if not pointer_path.exists():
        return None
    return int(json.loads(pointer_path.read_text(encoding="utf-8"))["version"])


def publish_checkpoint(memory: OnlineBasinMemory, checkpoint_dir: str | Path) -> int:
    """Write and atomically publish the next version; prior versions remain.

    The version file is written to a `.tmp` path and moved into place with
    `os.replace` (atomic on POSIX and Windows), so a version file only ever
    exists fully written. `latest.json` is only repointed after that move
    succeeds, so a reader never observes a version the writer considers
    incomplete.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous = latest_version(checkpoint_dir)
    version = 0 if previous is None else previous + 1

    payload = memory.serializable()
    version_path = _version_path(checkpoint_dir, version)
    tmp_path = version_path.parent / (version_path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, version_path)
    _write_pointer(checkpoint_dir, version)
    return version


def load_checkpoint(
    checkpoint_dir: str | Path, *, version: int | None = None
) -> OnlineBasinMemory:
    """Load a specific version, or the currently published one if omitted."""
    checkpoint_dir = Path(checkpoint_dir)
    target = version if version is not None else latest_version(checkpoint_dir)
    if target is None:
        raise FileNotFoundError(f"no published checkpoint in {checkpoint_dir}")
    path = _version_path(checkpoint_dir, target)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint version {target} not found at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return OnlineBasinMemory.from_serialized(payload)


def rollback(checkpoint_dir: str | Path, version: int) -> None:
    """Atomically repoint `latest` at an earlier, still-present version."""
    checkpoint_dir = Path(checkpoint_dir)
    if not _version_path(checkpoint_dir, version).exists():
        raise FileNotFoundError(f"cannot roll back to missing version {version}")
    _write_pointer(checkpoint_dir, version)
