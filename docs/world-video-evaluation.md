# Prerecorded World-Gemma evaluation

Prerecorded tabletop clips are the first real-camera benchmark. They keep the
physical scene fixed while detector, projector, tracker, memory, and language
checkpoints change. Public webcam mode still retains no frames; this tool is an
explicit local experiment that reads footage you provide.

## Record clips

Use a fixed camera at 1080p and 30 FPS when possible. Keep clips short, preserve
the untouched originals, and avoid drawing boxes or labels into the video. The
suggested set is in `configs/world-video-scenarios.json`. Twenty to sixty seconds
per scene is enough for the first pass.

## First replay

Run without annotations to create a timeline, summary, HTML report, and a
starter annotation file:

```bash
manifoldflow-world-replay-video \
  checkpoints/world/yolox_nano_tabletop.pth \
  checkpoints/world/siglip-key-projector.pt \
  footage/cup-swap.mp4 \
  --output-dir artifacts/world-video-runs/cup-swap
```

The output directory contains:

- `timeline.jsonl`: every analyzed source timestamp and predicted world state.
- `summary.json`: throughput, latency, drop, and optional accuracy metrics.
- `report.html`: a local, self-contained review page without embedded footage.
- `annotations.template.json`: a separate ground-truth starter file.

The default five-FPS sampling matches the browser capture budget. Change it with
`--send-fps`; this does not alter the source video.
Detector confidence defaults to the public runtime's `0.30`. Evaluation sweeps
may set `--detector-confidence`, and the chosen value should be included whenever
results are reported.
The tracker has its own `--track-confidence` gate, defaulting to `0.35`; detector
threshold sweeps should lower both gates explicitly.
Weak classes can be swept without flooding reliable classes by repeating
`--class-detector-confidence CLASS=VALUE` and
`--class-track-confidence CLASS=VALUE`. The global gates remain the fallback for
classes without an override.

The four initial real-clip annotations are versioned under
`configs/world-video-annotations`. Their `split` fields reserve `Video` and
`Video_2` for perception training and hold out the entirety of `Video_1` and
`Video_3` for validation. The source videos remain local and ignored.

Export the annotated frames for detector adaptation with:

```bash
manifoldflow-world-export-video-dataset \
  artifacts/world-video-footage \
  configs/world-video-annotations \
  artifacts/world-real-coco
```

The split stored in each annotation is authoritative, so validation clips cannot
accidentally leak into training when the export is repeated.

## Two replay modes

Deterministic mode is the default. It analyzes every selected frame in order and
injects the original video timestamp into world memory. No selected frame is
dropped, making checkpoint comparisons reproducible.

Real-time mode emits frames at their recorded cadence into a one-frame mailbox:

```bash
manifoldflow-world-replay-video \
  checkpoints/world/yolox_nano_tabletop.pth \
  checkpoints/world/siglip-key-projector.pt \
  footage/fast-motion.mp4 \
  --annotations footage/fast-motion.annotations.json \
  --mode realtime --playback-rate 1 \
  --output-dir artifacts/world-video-runs/fast-motion-realtime
```

If inference is busy, newer frames supersede the pending frame. The final source
frame is always drained before replay ends. The report records offered,
processed, and superseded frame counts plus processing and queue-delay latency.

## Annotation format

Annotations are JSON and never modify the original video. Checkpoints are sorted
by `timestamp_s`. Boxes are optional normalized `[x1, y1, x2, y2]` coordinates.
Stable ground-truth IDs, such as `red-cup`, are independent of tracker IDs.
For annotated video runs, the closed set of cup-color text prototypes is derived
from the cup labels present in the annotation checkpoints. This exposes the
allowed vocabulary without revealing which label belongs to a detected cup.

```json
{
  "schema_version": 1,
  "video": "cup-swap.mp4",
  "scenario": "cup-swap",
  "checkpoints": [
    {
      "timestamp_s": 2.0,
      "entities": [
        {
          "id": "red-cup",
          "class": "cup",
          "label": "red",
          "zone": "left",
          "visibility": "visible",
          "box": [0.08, 0.22, 0.27, 0.78]
        }
      ],
      "relations": []
    }
  ],
  "expected_events": [
    {
      "type": "zone_changed",
      "entity": "red-cup",
      "start_s": 3.0,
      "end_s": 4.5
    }
  ],
  "questions": [
    {
      "id": "red-cup-location",
      "timestamp_s": 5.0,
      "question": "Where is the red cup?",
      "expected_answers": ["The red cup is in the right zone."]
    }
  ]
}
```

Checkpoint scoring reports entity precision/recall, box IoU, label, zone,
visibility, and active-relation accuracy. Event windows report expected-event
recall rather than precision because annotations may intentionally list only
important events. Timestamped questions are asked immediately after the first
processed frame at or after their requested time. The initial CLI uses the
deterministic evidence answerer; the response mode is preserved in the timeline
so it is never confused with a Gemma result. To run the same timestamped
questions against the selected bundled checkpoint, add:

```bash
--gemma-checkpoint checkpoints/world/oracle-schema-final.pt
```

Bundled port, query-projector, and LoRA weights are restored together. Constrained
fact ranking remains the default; `--free-generation` is an explicit experimental
ablation.

Across checkpoints, reports also measure physical-ID consistency, identity
switch rate, and non-empty label switch rate. These expose trackers that score
individual boxes correctly while silently changing entity identity or color.

Use `--max-frames` for a short smoke test. A deterministic run and a real-time
run should use separate output directories so their reports remain directly
comparable.

## Initial real-footage result

The synthetic-only detector produced zero matched current-frame observations on
all four clips. A 25-frame/61-box adaptation set was then created only from
`Video` and `Video_2`; `Video_1` and `Video_3` remained whole-clip held out with
11 frames/26 boxes. The best dense YOLOX-Nano checkpoint reached 42.36% COCO
AP50:95 and 67.4% AP50 on that small held-out set. Per-class AP50:95 was 3.81%
toy car, 82.71% box, 57.69% cup, and 25.25% ball.

At the public runtime gates (detector 0.30, tracker 0.35), held-out observation
precision/recall was 100%/83.3% on the pink-cup clip and 100%/62.5% on the
same-car clip. Those attractive precision values conceal missing ball/car
detections: relation recall and question accuracy were still zero. Lowering both
gates to 0.01 increased observation recall to 88.9% and 87.5%, and the same-car
question became correct, but precision collapsed to 41.0% and 31.8%. The cup
clip incorrectly bound the ball to the mint cup instead of the pink cup.

This is a useful perception baseline, not a world-model accuracy claim. The
validation set is only two clips and eleven annotated frames. It demonstrates
that the downstream memory can consume weak real-camera detections, while also
isolating the next bottleneck: more car/ball examples and calibrated class-wise
detection gates are required before binding and object-permanence behavior can
be evaluated cleanly.

An exploratory class-specific sweep kept cup/box at 0.30/0.35 while lowering
only ball/car detector and tracker gates to 0.01. On the same-car clip it reached
90.0% persistent-state recall, 100% relation recall, 100% expected-event recall,
and 100% question accuracy. Observation precision was only 41.2%, so this is not
a selected runtime configuration. On the cup clip, observation precision/recall
was 78.9%/83.3%, but relation recall and question accuracy remained zero. The
system answered that it could not determine which cup contained the ball. This
is safer than the globally permissive run's incorrect mint-cup claim, but it
still fails the intended color-binding test.

Because these settings were inspected on the two validation clips, the result is
diagnostic and those clips should no longer be described as pristine for future
threshold selection. A new physical clip set will be needed for a final locked
evaluation after thresholds are calibrated on training footage.

## Temporal identity and uncertainty upgrade

The live and replay pipelines now embed detection crops before tracking. The
association step combines projected appearance, motion prediction, and box
overlap. A clean appearance match wins through a crossing, while strong overlap
can preserve identity through one noisy embedding. Unit scenarios cover both
cases with colors and geometry unrelated to the recorded clips.

Cup color is no longer a per-frame argmax that overwrites entity metadata.
Confidence- and crop-quality-weighted evidence decays over time, establishes a
label only with a winner margin, and requires sustained counterevidence before a
label switch. ManifoldFlow re-identification similarly requires a nearest-vs-
second-nearest margin, rejects impossible same-timestamp merges, and uses a
spatial guard for currently visible entities.

Containment and occlusion confirmation now use source timestamps by default, so
the same transition has comparable timing at different sampling rates. Candidate
relations retain their evidence scores. A hard relation is written only when the
best candidate clears both confidence and ambiguity gates; otherwise the entity
becomes missing and the evidence answerer abstains.

On the existing diagnostic clips, current-frame detector precision/recall is
unchanged because the detector checkpoint is unchanged. The pink-cup clip's
persistent-state recall improved from 65.2% to 91.3%, state precision changed
from 55.6% to 50.0%, and label accuracy improved from 33.3% to 42.9%. Binding
still correctly remains unresolved rather than guessing a cup.

The same-car clip retains 90.0% state recall, but relation recall and Q&A fall to
zero under the uncertainty gate. Timeline inspection showed that the earlier
reported relation success was mediated by a low-confidence false car proposal
that happened to disappear near the box. The upgraded engine refuses to promote
that proposal into a confident world fact. State precision also falls because
simultaneous similar proposals are no longer merged into one convenient entity.
This is a stricter and more truthful baseline, although it makes the remaining
detector calibration work more visible.
