# World-Gemma local runbook

Date established: 2026-09-01

This is the gated local schedule for the next World-Gemma experiments on the
RTX 5080 rig. A run completing is not the same as a checkpoint being promoted.
Every promotion requires the dependent gates below, and results belong under
Git-ignored `artifacts/` or `checkpoints/` paths.

## Environment

Run all launchers from WSL2, not Windows Python. The known working environment
is `~/gtr_venv` on the Ryzen 9 9950X / RTX 5080 machine, with PyTorch
2.11.0+cu128. The default may be overridden with `WORLD_VENV`.

```bash
cd /mnt/c/Users/user/Documents/projects/sonic-manifold-flow
source ~/gtr_venv/bin/activate
pip install -e '.[world,dev,ann]'
pip install -e artifacts/YOLOX
pip install loguru thop ninja tabulate tensorboard pycocotools
```

Required system/runtime pieces are WSL2, an NVIDIA WSL driver with CUDA visible
to PyTorch, Git, Python 3.10 or newer, and FFmpeg-compatible OpenCV video
decoding. Model access requires accepted Gemma terms and a Hugging Face token in
the environment. Use a standard token variable such as `HF_TOKEN`; never put a
token in a config file or commit it.

The authoritative defaults are in `configs/world-runs/next-runs.env`. Export a
value before launching to override it. Promotion thresholds also have a
machine-readable copy in `configs/world-runs/gates.json`.

## Inputs that must already exist

- YOLOX source checkout: `artifacts/YOLOX`
- Perception Test cup corpus:
  `artifacts/world-corpora/perception-test-train-cups`
- Initial full YOLOX checkpoint:
  `checkpoints/world/yolox-runs/world_yolox_real_dense/best_ckpt.pth`
- SigLIP projector: `checkpoints/world/siglip-key-projector.pt`
- Gemma adapter: `checkpoints/world/oracle-schema-final.pt`
- Synthetic corpus: `artifacts/world-sim`
- Local validation footage: `artifacts/world-video-footage`
- Versioned local annotations: `configs/world-video-annotations`

The Perception Test manifest points to the WSL-native MP4 directory
`~/world-gemma-data/perception-test-train-videos`. The prepared JPEG/COCO corpus
must remain intact even if the original MP4s are later moved.

## Run order and gates

| ID | Earliest start | Launcher | Purpose | Promotion gate |
| --- | --- | --- | --- | --- |
| R0 | Now | `bash scripts/world_runs/00_preflight.sh` | Validate dependencies, CUDA, corpus isolation, files, and focused tests. | All checks pass. Do not start training on a partial corpus. |
| R1 | After R0 | `bash scripts/world_runs/01_train_yolox_perception.sh` | Two-epoch real-domain adaptation smoke run. | Finite loss, evaluation completes, no OOM, and a usable checkpoint exists. Accuracy is report-only at this stage. |
| R2 | After R1 | command below | Full 40-epoch Perception Test cup/ball adaptation. | Real validation mAP50:95 at least 0.45, cup AP at least 0.55, ball AP at least 0.30. |
| R3 | After R2 | evaluation/replay commands below | Check synthetic retention and the two whole-clip local validation scenes. | No supported synthetic class loses more than 0.05 AP50:95; physical observation precision at least 0.70, recall at least 0.75, identity consistency at least 0.90, and label-switch rate at most 0.05. |
| R4 | After R3 and real identity-loader implementation | Not yet launchable | Train the SigLIP/ManifoldFlow identity projector on real temporal positive/negative pairs. | Temporal re-ID top-1 at least 0.85; cup/ball re-ID at least 0.80; ambiguous-match false accepts and label switches at most 0.05. |
| R5 | After R4 | `bash scripts/world_runs/03_replay_validation.sh ...` | Calibrate state/event uncertainty on training-only scenes, then evaluate frozen settings. | Relation precision at least 0.75, relation recall at least 0.65, expected-event recall at least 0.70, and ambiguous evidence abstains. |
| R6 | After R5 supplies reliable visual state | `bash scripts/world_runs/04_evaluate_gemma_gate.sh` | Re-evaluate whether Gemma is using retrieved world state. | Retrieval top-1 at least 0.95, ranked answer accuracy at least 0.75, correct-minus-shuffled margin at least 0.40, disabled accuracy at most 0.15. |
| R7 | Independent 5080 slot | `bash scripts/world_runs/05_repeat_scale_benchmark.sh` | Repeat the 100k/1M systems benchmark in three fresh launches/seeds. | 1M CPU p95 below 1 ms and resident-GPU batch-256 speedup at least 2x in every repeat. |

### R1 — detector smoke

The default launcher is intentionally short:

```bash
bash scripts/world_runs/01_train_yolox_perception.sh
bash scripts/world_runs/02_evaluate_yolox.sh
```

If batch 32 does not fit, set `YOLOX_BATCH_SIZE=16` and record the override. Do
not lower image size or change the split to rescue a run without treating it as
a new experiment.

### R2 — full detector adaptation

```bash
RUN_NAME=world_yolox_perception_full \
RUN_EPOCHS=40 \
RUN_EVAL_INTERVAL=2 \
bash scripts/world_runs/01_train_yolox_perception.sh

RUN_NAME=world_yolox_perception_full \
bash scripts/world_runs/02_evaluate_yolox.sh
```

The corpus has 9,422 cup, 3,407 ball, and only 99 toy-car source observations;
it has no box class. Toy-car performance is report-only on real validation and
box performance cannot be inferred from it. This is why R3 is mandatory.

### R3 — retention and physical replay

Evaluate the same full checkpoint on the untouched synthetic validation split:

```bash
RUN_NAME=world_yolox_perception_full \
EVAL_DATASET=artifacts/world-sim \
EVAL_LABEL=synthetic-regression \
bash scripts/world_runs/02_evaluate_yolox.sh

bash scripts/world_runs/03_replay_validation.sh \
  checkpoints/world/yolox-runs/world_yolox_perception_full/best_ckpt.pth
```

`Video_1` and `Video_3` remain whole-clip validation. The two clips previously
used for perception training are excluded. Because the original four clips
have already informed development choices, they are diagnostic validation, not
a pristine final test. Freeze settings and record fresh physical scenes before
making a public accuracy claim.

If real AP rises but synthetic retention fails, do not promote the checkpoint.
The next implementation task is a source-balanced mixed-corpus builder and
sampler, followed by a new R2. Reusing validation frames as training data is not
an acceptable fix.

### R4 — currently blocked, by design

`manifoldflow-world-train-projector` currently consumes synthetic
`world_frames.jsonl` oracle keys. It does not yet construct real temporal
positive pairs, hard negatives, or occlusion-aware triplets from Perception
Test tracks. Running it against the real corpus would therefore not be the
experiment described by R4. Implement and test that general loader/objective,
then add its launcher here before spending a full GPU run.

### R6 — Gemma interpretation

The current checkpoint already shows retrieval top-1 0.957 and constrained
ranked-answer accuracy 0.650, versus 0.129 with shuffled memory and 0.068 with
memory disabled. Free generation is only 0.055 accurate and remains diagnostic.
Use constrained ranking for the proof of concept until generation crosses a
separate gate; do not summarize ranked accuracy as free-form language accuracy.

Only start another LoRA run if the code-level training gate calls for it after
R5. A perception or state-construction failure is not evidence that Gemma needs
more parameters.

### R7 — scale interpretation

The launcher runs seeds 17, 18, and 19 at 100k and 1M basins. Each benchmark is
a fresh Python process, so banks and HNSW indexes are rebuilt independently. A
stricter cold-machine study can repeat the launcher from fresh WSL sessions,
but those extra repeats must be reported separately. This benchmark measures
lookup, rerank, and analytic flow speed over synthetic basin banks. It does not
measure semantic recall or application-derived latent quality.

## Recording a run

For every run, append to `docs/World-Gemma-Dev-Journal.md`:

- commit hash and whether the tree was clean;
- launcher plus every overridden environment value;
- GPU, PyTorch/CUDA, dataset manifest, split counts, and random seed;
- start/end time, wall time, peak VRAM/RAM, and output checkpoint;
- all gate metrics, including failures and abstentions;
- decision: promote, hold for diagnosis, or reject.

Never overwrite an accepted checkpoint. Export the selected full YOLOX
checkpoint to a runtime-only checkpoint with `manifoldflow-world-export-yolox`,
then update the runtime config in a separate reviewed commit.
