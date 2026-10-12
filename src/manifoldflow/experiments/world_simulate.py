"""CLI for generating the synthetic tabletop training corpus."""

from __future__ import annotations

import argparse

from manifoldflow.world_dataset import generate_synthetic_dataset
from manifoldflow.world_renderer import RenderConfig
from manifoldflow.world_simulator import SimulatorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/world-sim")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--no-domain-randomization", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_synthetic_dataset(
        args.output,
        episode_count=args.episodes,
        seed=args.seed,
        simulator_config=SimulatorConfig(frames_per_episode=args.frames),
        render_config=RenderConfig(
            width=args.width,
            height=args.height,
            domain_randomization=not args.no_domain_randomization,
        ),
    )
    print(
        f"wrote {result.episodes} episodes, {result.frames} frames, and "
        f"{result.questions} questions to {result.output_dir}"
    )


if __name__ == "__main__":
    main()
