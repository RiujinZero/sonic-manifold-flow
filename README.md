# sonic-manifold-flow

Sonic + ManifoldFlow: a pretrained recurrent state-space language model (Sonic SSM) coupled to a
continuous, sparsely routed external memory (ManifoldFlow).

> **Unmaintained.** This is a research prototype released as-is. It is no longer developed or
> supported; issues and pull requests are not monitored.

## What it is

ManifoldFlow began as a geometry-first sparse attractor memory: an encoder maps an input to a
normalized latent, sparse retrieval selects a bounded set of candidate basins, and a local flow
moves the latent through them. Only retrieved basin records take part in a query or an online
update, so the bank can grow while per-query compute stays roughly constant.

This repository is the stage of that work where the memory is wired into the forward pass of the
Sonic SSM language model through a small trainable adapter. The combined system is trained to keep
doing ordinary next-token prediction while writing facts met during a session into the memory bank
and retrieving them later, instead of carrying them in the recurrent state.

It also contains a separate tabletop "world demo" application that uses the original image-latent
memory design to track persistent visual entities and answer questions about them.

The later byte-level continual learner that grew out of this codebase is published separately as
[ALM](https://github.com/RiujinZero/ALM).

## Layout

| Path | Contents |
|---|---|
| `src/manifoldflow/` | Core library: basin bank, flow, routing, indexes, continual updates, Sonic and Qwen adapters, world-demo application |
| `src/manifoldflow/experiments/` | Experiment and training entry points (exposed as `manifoldflow-*` console scripts) |
| `src/manifoldflow/dataset/` | Session and vocabulary generators for the memory training data |
| `scripts/` | Data pre-generation, sample generation and training launch scripts |
| `configs/` | Experiment and world-run configuration |
| `deploy/world-demo/` | Container setup for the world demo |
| `resources/tokenizers/` | The 50k English tokenizer used by Sonic |
| `vendor/GTR-sonic-ssm/` | Vendored Sonic SSM training code, kept for reference |
| `tests/` | Unit tests |
| `docs/` | Architecture and design documents |

## Documentation

- [Sonic-MF Architecture v1](docs/Sonic-MF%20Architecture%20v1.md): the system as implemented
- [Sonic-MF Training Design v0.1](docs/Sonic-MF%20Training%20Design%20v0.1.md)
- [Language world model design v0.4](docs/ManifoldFlow_Sonic_Language_World_Model_Design_v0.4.md)
- [ManifoldFlow design v0.2](docs/design-v0.2.md) and [v0.3](docs/design-v0.3.md): the original attractor memory
- [World demo](docs/world-demo.md) and its [model card](docs/world-model-card.md)
- [Third-party notices](docs/THIRD_PARTY_WORLD_NOTICES.md)

[README_ORIGINAL.md](README_ORIGINAL.md) is the working README from development. It has detailed
command-line usage and status notes, and may refer to result files that are not part of this
release.

## Getting started

Requires Python 3.10+ and PyTorch 2.2+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Optional extras are defined in `pyproject.toml` (for example `vision`, `stream`, `ann`, `world`).

## What is not included

Model checkpoints, training data, run outputs and experiment result logs are not part of this
repository. Scripts that expect them under `artifacts/` or `checkpoints/` need you to produce or
supply those files.

## License

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use, modify and
share this code for personal, research and other noncommercial purposes. Commercial use is not
permitted. Third-party components keep their own licenses; see
[the third-party notices](docs/THIRD_PARTY_WORLD_NOTICES.md).

Required Notice: Copyright (c) 2026 RiujinZero (https://github.com/RiujinZero/sonic-manifold-flow)
