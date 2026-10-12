"""Download and prepare licensed external video corpora for World-Gemma."""

from __future__ import annotations

import argparse
import json
from typing import Any

from manifoldflow.world_external_datasets import (
    ASSETS,
    asset_catalog,
    download_asset,
    extract_archive,
    prepare_perception_test,
    prepare_physion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="show allow-listed assets and licenses")

    download = commands.add_parser("download", help="download one official asset")
    download.add_argument("asset", choices=sorted(ASSETS))
    download.add_argument("--raw-root", default="data/world-external")
    download.add_argument("--no-extract", action="store_true")

    extract = commands.add_parser("extract", help="safely extract a local archive")
    extract.add_argument("archive")
    extract.add_argument("output")
    extract.add_argument("--workers", type=int, default=1)

    perception = commands.add_parser(
        "prepare-perception-test", help="normalize tracks and questions"
    )
    perception.add_argument("--raw-root", required=True)
    perception.add_argument("--output", required=True)
    perception.add_argument(
        "--annotation-file",
        help="path under raw-root; required when sample and train coexist",
    )
    perception.add_argument(
        "--video-root", help="separate MP4 root, such as WSL-native storage"
    )
    perception.add_argument("--all-videos", action="store_true")
    perception.add_argument("--extract-frames", action="store_true")
    perception.add_argument("--annotation-stride", type=int, default=1)
    perception.add_argument("--validation-fraction", type=float, default=0.15)
    perception.add_argument("--split-seed", type=int, default=17)

    physion = commands.add_parser("prepare-physion", help="index Physion MP4 files")
    physion.add_argument("--raw-root", required=True)
    physion.add_argument("--output", required=True)
    physion.add_argument("--validation-fraction", type=float, default=0.15)
    physion.add_argument("--split-seed", type=int, default=17)
    return parser


def run(args: argparse.Namespace) -> Any:
    if args.command == "catalog":
        return asset_catalog()
    if args.command == "download":
        return download_asset(
            args.asset, args.raw_root, extract=not args.no_extract
        )
    if args.command == "extract":
        extract_archive(args.archive, args.output, workers=args.workers)
        return {
            "archive": args.archive,
            "extracted_to": args.output,
            "workers": args.workers,
        }
    if args.command == "prepare-perception-test":
        return prepare_perception_test(
            args.raw_root,
            args.output,
            annotation_file=args.annotation_file,
            video_root=args.video_root,
            cup_games_only=not args.all_videos,
            extract_frames=args.extract_frames,
            annotation_stride=args.annotation_stride,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
        )
    if args.command == "prepare-physion":
        return prepare_physion(
            args.raw_root,
            args.output,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
        )
    raise ValueError(f"unknown command: {args.command}")


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
