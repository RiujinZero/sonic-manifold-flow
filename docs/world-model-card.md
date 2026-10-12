# Model card: ManifoldFlow tabletop world memory POC

## Intended use

This non-commercial research prototype demonstrates persistent object identity,
visibility state, containment, and question answering in controlled tabletop
scenes. It is intended for grant review, research evaluation, and reproducible
demonstrations.

It is not a general world model. It does not simulate physics, identify people,
infer intent, or provide safety-critical conclusions.

## Components

- Gemma 3 270M IT, subject to the Gemma Terms of Use.
- SigLIP2 Base Patch16 224 for aligned image/text embeddings.
- Official YOLOX Nano, Apache-2.0, fine-tuned on four tabletop classes.
- ByteTrack, MIT, for temporary visual track identifiers.
- ManifoldFlow `BasinState` image-latent memory and exact main/delta routing.
- Trainable state, query, and Gemma latent-prefix adapters.

## Evaluation

Report detector mAP50, visible identity accuracy, post-occlusion re-ID,
car-location accuracy, cup-containment accuracy, unknown precision, retrieval
recall@4, stale-state replacement, shuffled-memory degradation, non-memory Gemma
regression, and latency. Do not publish success claims until held-out clip and
question-template results pass `WorldEvaluation.acceptance_failures()`.

## Known limitations

- The public MVP assumes one fixed camera and visibly distinct cups.
- Detection gaps can be confused with occlusion.
- Similar objects can collapse into one identity basin or create duplicates.
- Gemma 270M has limited language and reasoning capacity.
- Live-browser lighting, camera angle, and object appearance can differ sharply
  from the training set.
- The deterministic fallback is explicitly labeled as evidence mode and must not
  be presented as Gemma output.

## Privacy

Live frames are transient and are not written to disk or logs. Sessions expire
after ten minutes and are isolated by random identifiers. The demo requests no
microphone access and prohibits human identification or surveillance use.
