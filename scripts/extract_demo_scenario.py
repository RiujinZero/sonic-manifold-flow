"""One-off extraction: pull a real rendered episode out of the synthetic
corpus into a small, self-contained asset bundle the demo can ship with
(images + a compact manifest), instead of the flat-color canvas the
hand-authored curated scenarios use. Run once per scenario; not part of
the runtime app.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract(dataset_dir: Path, episode_id: str, output_dir: Path) -> None:
    rows = []
    with (dataset_dir / "world_frames.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["episode_id"] == episode_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"no frames found for {episode_id} in {dataset_dir}")
    rows.sort(key=lambda row: row["frame_index"])

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    manifest_frames = []
    for row in rows:
        source_image = dataset_dir / "images" / row["image"]
        dest_name = f"{row['frame_index']:04d}.jpg"
        shutil.copyfile(source_image, images_dir / dest_name)
        manifest_frames.append(
            {
                "image": dest_name,
                "entities": [
                    {
                        "entity_id": entity["entity_id"],
                        "tracker_id": entity["tracker_id"],
                        "class_name": entity["class_name"],
                        "box_xyxy": entity["box_xyxy"],
                        "visibility": entity["visibility"],
                        "zone": entity["zone"],
                        "label": entity["label"],
                        "appearance_key": entity["appearance_key"],
                    }
                    for entity in row["entities"]
                ],
            }
        )

    manifest = {
        "episode_id": episode_id,
        "scenario": rows[0]["scenario"],
        "frame_count": len(rows),
        "frames": manifest_frames,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    print(f"wrote {len(rows)} frames to {output_dir}")


def main() -> None:
    dataset_dir = _PROJECT_ROOT / "artifacts" / "world-sim-9zone" / "train"
    static_scenarios = (
        _PROJECT_ROOT / "src" / "manifoldflow" / "static" / "scenarios"
    )
    extract(dataset_dir, "episode-000870", static_scenarios / "car-occlusion")
    extract(dataset_dir, "episode-000171", static_scenarios / "cup-shuffle")


if __name__ == "__main__":
    main()
