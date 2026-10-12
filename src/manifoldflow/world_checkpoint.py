"""Atomic bundles for world latents, learned adapters, and deployment metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from manifoldflow.world_memory import WorldMemory


def publish_world_bundle(
    directory: str | Path,
    *,
    memory: WorldMemory,
    session_id: str,
    modules: dict[str, nn.Module],
    manifest: dict[str, Any],
) -> Path:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    version = _latest(target_dir) + 1
    payload = {
        "format_version": 1,
        "memory": memory.serializable_session(session_id),
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "manifest": manifest,
    }
    target = target_dir / f"world-v{version:06d}.pt"
    temporary = target.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    pointer = target_dir / "world-latest.json"
    pointer_tmp = pointer.with_suffix(".json.tmp")
    pointer_tmp.write_text(json.dumps({"version": version}) + "\n", encoding="utf-8")
    os.replace(pointer_tmp, pointer)
    return target


def load_world_bundle(
    directory: str | Path,
    *,
    modules: dict[str, nn.Module] | None = None,
    session_id: str | None = None,
) -> tuple[WorldMemory, dict[str, Any]]:
    root = Path(directory)
    version = _latest(root)
    if version < 0:
        raise FileNotFoundError(f"no world bundle in {root}")
    payload = torch.load(
        root / f"world-v{version:06d}.pt", map_location="cpu", weights_only=True
    )
    if payload.get("format_version") != 1:
        raise ValueError("unsupported world bundle format")
    if modules is not None:
        for name, module in modules.items():
            module.load_state_dict(payload["modules"][name])
    memory = WorldMemory.from_serializable_session(
        payload["memory"], session_id=session_id
    )
    return memory, dict(payload["manifest"])


def _latest(directory: Path) -> int:
    pointer = directory / "world-latest.json"
    if not pointer.exists():
        return -1
    return int(json.loads(pointer.read_text(encoding="utf-8"))["version"])
