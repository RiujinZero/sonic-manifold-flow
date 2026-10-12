# ManifoldFlow tabletop world demo

The gated local training and evaluation schedule, including WSL dependencies,
configs, launch scripts, and promotion criteria, is documented in
[`world-gemma-next-runs.md`](world-gemma-next-runs.md).

This is a separate application path from the Sonic experiments. It retains the
original ManifoldFlow image-latent design: every persistent visual entity is a
row in `BasinState`, updates use its Welford statistics, and queries route through
`ExactMainDeltaIndex`. Typed state and relations sit beside those latent basins
so the public UI can explain what was remembered.

## What works without downloaded models

The curated car and cup scenarios, event engine, persistent entity memory,
relationship tracking, evidence-based answers, chat API, session erasure, and
reviewer UI run without external weights:

```bash
source ~/gtr_venv/bin/activate
pip install -e '.[world]'
manifoldflow-world-demo
```

Open `http://127.0.0.1:8000`. The health endpoint reports live inference as
`unconfigured` until the vision checkpoints are supplied. Curated mode remains
available during a live-service outage.

## Enabling live perception

Accept the Gemma terms before downloading `google/gemma-3-270m-it`. Train a
four-class YOLOX Nano checkpoint using the official Apache-2.0 YOLOX code. The
required class order is:

1. `toy_car`
2. `occluder_box`
3. `cup`
4. `ball`

YOLOX is installed from its official source rather than the stale PyPI
dependency, whose pinned ONNX exporter does not build on the current WSL Python:

```bash
git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git artifacts/YOLOX
pip install --no-build-isolation --no-deps -e artifacts/YOLOX
pip install loguru thop ninja tabulate tensorboard pycocotools \
  opencv-python-headless
```

The ONNX simplifier is intentionally omitted because it is not used for local
training or PyTorch inference.

Provide these environment variables before starting the server:

```bash
export MANIFOLDFLOW_YOLOX_CHECKPOINT=/models/yolox_nano_tabletop.pth
export MANIFOLDFLOW_PROJECTOR_CHECKPOINT=/models/siglip_key_projector.pt
export MANIFOLDFLOW_GEMMA_CHECKPOINT=/models/oracle-schema-final.pt
export MANIFOLDFLOW_HOST=127.0.0.1
export MANIFOLDFLOW_PORT=8000
manifoldflow-world-demo
```

The Gemma variable is optional. With the YOLOX and SigLIP projector weights,
live perception works while chat uses the deterministic evidence fallback. With
the bundled checkpoint, the server restores the latent port, Gemma query
projector, and upper-layer LoRA together. Chat then switches to constrained
latent-memory Gemma answers. Set `MANIFOLDFLOW_GEMMA_FREE_GENERATION=1` only for
experiments; unrestricted generation is not the reviewer-facing mode. The old
separate port/query environment variables remain supported for legacy adapter
checkpoints.

## Training order

Generate the first curriculum locally before downloading any model weights:

```bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow"
source ~/gtr_venv/bin/activate
manifoldflow-world-simulate --output artifacts/world-sim --episodes 1000
```

This produces deterministic car-occlusion and cup-shuffle episodes, rendered
JPEG frames, YOLOX-compatible COCO boxes, typed world-state logs, oracle
128-dimensional identity latents, relations, events, and grounded questions.
Entire episodes are assigned to 70/15/15 splits, and validation/test questions
use language templates that never occur in training. The oracle keys combine
seeded class/color structure with smaller instance-specific variation; they are
not claimed SigLIP features. The vision-integration stage replaces them with
projected SigLIP2 crop embeddings.

`load_oracle_examples` joins each question with its exact frame state.
`collate_oracle_examples` pads the two-entity car and four-entity cup examples
into tensors shaped for `LatentMemoryPort`: `[batch, memory, 128]` keys,
`[batch, memory, 256]` values, memory masks, and evidence masks.

Synthetic state uses the same live encoder layout: normalized center at `0:2`,
velocity at `2:4`, box at `4:8`, confidence/visibility/counters at `8:12`,
relation fields at `12` and `16:22`, zone one-hot at `24:27`, and class one-hot
at `28:32`. A regression test checks the layout and prevents renderer dimensions
from being applied to already-normalized simulator boxes.

Training then proceeds in this order:

1. Train the Gemma query projector and latent port on the generated oracle state.
2. Fine-tune YOLOX Nano on the same rendered frames, then on recorded tabletop data.
3. Train `CrossModalKeyProjector`, `WorldStateEncoder` replacement, and
   `LatentMemoryPort` using real SigLIP2 features.
4. Replace oracle retrieval with `WorldMemory.query` and its image-latent bank.
5. Integrate recorded detector/tracker events.
6. Enable rank-8 LoRA only when `should_enable_lora` returns true.

The frozen SigLIP2 bridge is reproducible with:

```bash
manifoldflow-world-train-projector \
  --dataset artifacts/world-sim \
  --output checkpoints/world/siglip-key-projector.pt \
  --cache-dir artifacts/world-siglip-cache \
  --steps 2000 --batch-size 64 --feature-batch-size 64
```

It trains one shared linear projection for visible image crops and text
prototypes. Validation reports target cosine, within-frame retrieval, temporal
re-identification against earlier crops, and text-prototype retrieval.

YOLOX training uses `world_yolox_nano.py` with the official source checkout and
COCO-pretrained Nano weights. Export the best training checkpoint to a
model-only runtime file with `manifoldflow-world-export-yolox`; this keeps
optimizer and RNG objects out of public inference loading.

The first oracle adapter run and held-out validation are reproducible with:

```bash
manifoldflow-world-train-oracle \
  --dataset artifacts/world-sim \
  --output checkpoints/world/oracle-port.pt \
  --steps 1000 --batch-size 8

manifoldflow-world-evaluate-oracle \
  --dataset artifacts/world-sim \
  --checkpoint checkpoints/world/oracle-port.pt \
  --output checkpoints/world/oracle-validation.json \
  --split validation --max-examples 512
```

After the LoRA gate is met, preserve solved retrieval geometry with
`--freeze-query-projector` while refining generated answers.

For the controlled tabletop demo, reliable answers can use constrained Gemma
likelihood ranking over the 24 training-world facts. Candidate types are selected
from question intent, and Gemma ranks the remaining facts using latent memory.
This mode must be labeled as closed-world fact decoding; unrestricted generation
remains experimental.

Gemma stays frozen in the adapter-only stage. After the measured gate, rank-8
LoRA is allowed only on attention projections in its upper six layers. The
evaluator measures held-out retrieval and teacher-forced answer loss/accuracy
with correct, shuffled, and disabled memory; the test split remains untouched
until model-selection decisions are complete.

The reusable objectives and acceptance gate live in `world_training.py` and
`world_evaluation.py`. `world_checkpoint.py` atomically saves the image-latent
bank, typed records, learned modules, and deployment manifest.

## Public deployment behavior

- Browser camera frames are capped at 1 MB and five FPS.
- Visual crops are embedded before association, allowing the tracker to combine
  appearance, motion, and overlap rather than relying on box IoU alone.
- Cup labels accumulate as confidence-weighted entity evidence with hysteresis;
  ambiguous crops do not overwrite a stable color on a single frame.
- Re-identification requires both an absolute appearance threshold and separation
  from the second-best identity. Ambiguous observations create a new uncertain
  entity instead of silently merging two physical objects.
- Occlusion and containment confirmation uses elapsed source time. Competing
  relation candidates remain explicit evidence and only become a hard relation
  when confidence and winner margin gates both pass.
- Live ingestion is latest-frame-only. Each connection has one pending slot;
  newer frames supersede unprocessed frames instead of entering a FIFO backlog.
- Responses expose a cumulative `superseded_frames` count for latency auditing.
- Frames and crops are processed in memory and are not persisted.
- Sessions use random server-generated IDs and expire after ten minutes.
- Each session has its own tracker, ManifoldFlow bank, events, and relations.

## Prerecorded footage

Before live-camera evaluation, use `manifoldflow-world-replay-video` to build a
repeatable real-video benchmark. It supports deterministic frame-complete replay
and real-time latest-frame replay, separate timestamped annotations, question
checkpoints, JSON timelines, and a local HTML report. See
`docs/world-video-evaluation.md` for the recording protocol and commands.

For licensed external MP4 acquisition and normalization, see
`docs/world-external-video-datasets.md`. The local catalog currently includes
Physion and DeepMind Perception Test assets, records download receipts and
licenses, and converts Perception Test cup games into identity-linked tracks,
object-permanence questions, and optional YOLOX-compatible frames.
- The UI provides an explicit **End & erase** action.
- Questions are capped at thirty per session.
- Deploy behind TLS, rate limiting, and bot protection. Bind the application to
  localhost when a reverse proxy is present.

The application intentionally performs no face identification, person tracking,
microphone capture, arbitrary uploads, or cross-session learning.

## 5080 rig as server, laptop as camera client

The laptop must load the page over HTTPS. Browser webcam APIs are unavailable to
a normal `http://<rig-ip>` page even on a trusted home network. The included
Caddy service terminates local TLS and proxies both HTTP and WebSocket traffic to
the WSL inference service.

1. Give the 5080 rig a stable LAN address or DHCP reservation.
2. On the laptop, map `manifoldflow.test` to that address in the hosts file or
   local DNS.
3. Start the included Compose stack from `deploy/world-demo`. It exposes only
   Caddy on the LAN; port 8000 remains bound to the rig's loopback interface.
4. Export Caddy's local root certificate from the `caddy_data` volume and install
   it as a trusted root on the laptop. This is required once per client device.
5. Open `https://manifoldflow.test` on the laptop and approve camera access.

With no checkpoint variables, the stack starts in curated-only mode. For live
mode, place the four trained files under `checkpoints/world` and set their
container paths (for example `/models/yolox_nano_tabletop.pth`) in a local
`deploy/world-demo/.env` file.

The frontend uses same-origin `wss://` automatically, so it needs no CORS or
hard-coded rig address. Caddy includes the required camera permissions header.
Limit Windows Firewall access to the private network profile and, preferably,
the laptop's address. Do not expose ports 80, 443, or 8000 through the router.

The certificate can be copied after the first start with:

```bash
docker compose cp \
  caddy:/data/caddy/pki/authorities/local/root.crt ./manifoldflow-local-root.crt
```

Transfer only that public certificate—not Caddy's private key—to the laptop.

## Verification

Run the focused suite in WSL:

```bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow"
~/gtr_venv/bin/python -m pytest \
  tests/test_world_memory.py tests/test_world_events.py \
  tests/test_gemma_world.py tests/test_world_demo.py \
  tests/test_world_checkpoint.py tests/test_world_training.py \
  tests/test_public_demo.py -q
```

The full legacy suite additionally requires its existing `vision` extra.
