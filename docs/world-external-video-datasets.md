# External MP4 corpora

World-Gemma uses external video only when the source, license, and clip split are
recorded. Raw archives, extracted MP4s, and prepared frames belong under `data/`
or `artifacts/`; both are ignored by Git.

## Recommended acquisition order

Start with Physion's compact training set, then validate the Perception Test
mapping on its sample files. The full Perception Test training video archive is
26.5 GB, so it is intentionally a separate download.

```bash
cd "/mnt/c/Users/user/Documents/projects/sonic-manifold-flow"
source ~/gtr_venv/bin/activate
pip install -e '.[world]'

manifoldflow-world-datasets catalog
manifoldflow-world-datasets download physion-train-videos
manifoldflow-world-datasets prepare-physion \
  --raw-root data/world-external/physion \
  --output artifacts/world-corpora/physion

manifoldflow-world-datasets download perception-test-sample-annotations
manifoldflow-world-datasets download perception-test-sample-videos
manifoldflow-world-datasets prepare-perception-test \
  --raw-root data/world-external/perception-test \
  --annotation-file extracted/perception-test-sample-annotations/sample.json \
  --output artifacts/world-corpora/perception-test-sample \
  --extract-frames
```

On WSL, large archives can remain in the repository while their many MP4 files
are extracted to the native Linux filesystem. This avoids `/mnt/c` per-file
overhead:

```bash
manifoldflow-world-datasets extract \
  data/world-external/perception-test/archives/train_videos.zip \
  ~/world-gemma-data/perception-test-train-videos --workers 8

manifoldflow-world-datasets prepare-perception-test \
  --raw-root data/world-external/perception-test \
  --annotation-file \
    extracted/perception-test-train-annotations/all_train.json \
  --video-root ~/world-gemma-data/perception-test-train-videos \
  --output artifacts/world-corpora/perception-test-train-cups \
  --extract-frames
```

For the real cup-game training corpus:

```bash
manifoldflow-world-datasets download perception-test-train-annotations
manifoldflow-world-datasets download perception-test-train-cut-frames
manifoldflow-world-datasets download perception-test-train-videos --no-extract
manifoldflow-world-datasets prepare-perception-test \
  --raw-root data/world-external/perception-test \
  --annotation-file \
    extracted/perception-test-train-annotations/all_train.json \
  --output artifacts/world-corpora/perception-test-train-cups \
  --extract-frames
```

Downloads use an allow-listed catalog and keep an adjacent receipt containing
the source URL, license, local byte count, and extraction location. Interrupted
downloads resume from the `.part` file when the server supports HTTP ranges.
Archive extraction rejects paths that escape the destination directory.

## Normalized outputs

The Perception Test importer defaults to cup games and writes:

- `clips.jsonl`: source clip metadata and local MP4 location;
- `track_frames.jsonl`: sparse normalized boxes, persistent entity IDs,
  visibility, source labels, and occluder flags;
- `questions.jsonl`: object-permanence questions, answers, alternatives, and
  evidence entity IDs;
- `coco.json` and `images/`: visible annotated frames for YOLOX adaptation when
  `--extract-frames` is supplied, under clip-separated `train/` and
  `validation/` directories;
- `manifest.json`: counts, source, license, and preparation settings.

Grounded questions determine which track is the hidden object, which track is
the occluder, and which tracks are the interchangeable cups. This is more robust
than a clip-specific filename or color rule. Masked observations remain in the
world-state timeline as `occluded` but are excluded from YOLOX ground truth.
When the official cup-game cut mapping is present, questions are attached before
the reveal frame so the answer cannot leak into the model input.

Physion preparation does not duplicate its MP4s. It writes a clip manifest with
the scenario inferred from the official directory layout. A later dynamics
stage can window those videos while retaining the source-level split.

The importer assigns whole clips to deterministic train and validation splits;
it never combines frames from one source clip across them. Keep the four locally
recorded scenes and future fresh recordings outside these training directories
so they remain meaningful domain-transfer evaluations.

## Attribution and exclusions

- DeepMind Perception Test non-code materials: CC BY 4.0.
- Physion code and data: MIT.
- IntPhys 2 is not in the download catalog. Its additional terms limit it to
  evaluation and prohibit generative-AI training.

The public demo should use locally recorded footage. Dataset attribution belongs
in the model card and release notices even when dataset clips are not displayed.
