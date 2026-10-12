# Public Gemma–ManifoldFlow World Model Demo

## Summary

Build a public, non-commercial grant-review demo combining:

- `google/gemma-3-270m-it` as the language interface.
- Apache-2.0-licensed YOLOX Nano for object detection.
- MIT-licensed ByteTrack for frame-to-frame tracking.
- Frozen SigLIP2 embeddings for visual identity and semantic retrieval.
- ManifoldFlow for persistent entity, state, and relationship memory.
- A learned latent-prefix port that injects retrieved world state into Gemma.

The public site will provide two modes:

1. Reliable, curated car-occlusion and cup-shuffle scenes with an interactive timeline and unrestricted questions.
2. An experimental live-browser-camera mode with explicit consent, session isolation, resource limits, and no retained video.

Ultralytics will not be used. Its current terms say projects using its YOLO code or trained models must either release the complete project under AGPL-3.0 or obtain a commercial license. [Ultralytics licensing guidance](https://www.ultralytics.com/license). YOLOX preserves the requested YOLO-family detector while using the permissive [Apache-2.0-licensed official implementation](https://github.com/megvii-basedetection/yolox).

## Architecture

### Perception and tracking

- Fine-tune `yolox_nano` on `toy_car`, `occluder_box`, `cup`, and `ball`.
- Initialize from the official COCO checkpoint and pin the source commit and weight hash.
- Use bounding boxes rather than instance masks.
- Infer containment and occlusion from calibrated geometry:
  - Ball center entering an expanded cup box before disappearing.
  - Car trajectory entering the box’s occlusion corridor.
  - Confidence hysteresis across multiple frames.
- Use the official [ByteTrack implementation](https://github.com/FoundationVision/ByteTrack), licensed under MIT, for short-term tracking.
- Use frozen `google/siglip2-base-patch16-224` for crop identity and text-query embeddings. [SigLIP2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2).
- Refresh crop embeddings on track creation, large appearance change, low-confidence identity, and every two seconds—not every frame.

### World state and ManifoldFlow

Introduce typed world-model interfaces alongside the existing text-memory APIs:

- `ObservationEvent`: session, frame/time, tracker ID, class, box, confidence, position, velocity.
- `WorldRecord`: persistent entity ID, class, appearance key, latest state value, version, provenance.
- `RelationRecord`: subject, relation, target, confidence, start/end times.
- `VisibilityState`: `VISIBLE`, `OCCLUDED`, `CONTAINED`, `MISSING`, or `UNKNOWN`.
- `WorldSnapshot`: immutable query-time view of entities, relationships, and recent events.
- `WorldMemory`:
  - `upsert_observation(event)`
  - `set_visibility(entity_id, state)`
  - `set_relation(subject, relation, target)`
  - `query(question, k=4)`
  - `snapshot()`
  - `close_session(session_id)`

Give each persistent entity one ManifoldFlow basin:

- 128-dimensional identity key from normalized SigLIP2 appearance plus class embedding; exclude position and time.
- 256-dimensional state value encoding position/zone, velocity, visibility, confidence, age, and relationships.
- Store structured state, thumbnails, and provenance separately for auditability.
- Use exact RBF routing for the small entity bank.
- Treat ByteTrack IDs as temporary; use ManifoldFlow and SigLIP2 to recover identity after tracker loss.
- Enforce session identifiers in every lookup and update. No user can retrieve another reviewer’s state.

### Event behavior

Emit:

- `APPEARED`
- `MOVED`
- `OCCLUDED`
- `REAPPEARED`
- `DISAPPEARED`
- `CONTAINED_BY`
- `RELEASED`
- `ZONE_CHANGED`

Defaults:

- Require three confirming frames for a state transition.
- Mark a track lost after ten missing frames.
- Preserve a car as `OCCLUDED_BY(car, box)` when its trajectory ends inside the box corridor.
- Match a reappearing car to its persistent entity and update its new side.
- Record `CONTAINED_BY(ball, cup)` when the ball enters the cup region and disappears.
- Move that relation with the persistent cup during a shuffle.
- Clear containment only when the ball is confidently visible outside the cup.
- Return `unknown` when evidence is insufficient.
- Use a four-point table calibration and named zones for deterministic language.

Live mode will provide a short calibration wizard and require three visibly distinct cups—preferably red, blue, and yellow. Identical-cup tracking is not part of the public MVP.

### Gemma latent port

Use `google/gemma-3-270m-it`, subject to Google’s Gemma terms. Those terms explicitly cover remotely hosted access, so the site must display the required notices and acceptable-use information. [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and [model card](https://huggingface.co/google/gemma-3-270m-it).

Implement `GemmaWorldModel`:

- Project a pooled Gemma question state into the 128-dimensional ManifoldFlow key space.
- Encode the same question through SigLIP2’s text encoder.
- Union both candidate sets, rerank with exact RBF distance, and expand directly related entities.
- Retrieve up to four records.
- Convert each state through `Linear(256→64) → GELU → Linear(64→640)`.
- Add learned type, recency, confidence, and retrieval-weight embeddings.
- Insert the resulting continuous memory tokens after BOS and before user tokens.
- Extend the attention mask and mask memory-token labels with `-100`.
- Use Gemma’s supported `inputs_embeds` interface rather than changing transformer internals. [Gemma Transformers documentation](https://huggingface.co/docs/transformers/v4.50.0/model_doc/gemma3).

## Training and Evaluation

### Training stages

1. **YOLOX detector**
   - Annotate approximately 1,000 frames from at least 30 clips.
   - Include diverse ordinary toy cars, cups, balls, boxes, backgrounds, lighting, and browser-camera quality.
   - Split complete clips 70/15/15 to avoid adjacent-frame leakage.

2. **Oracle-state latent port**
   - Freeze Gemma, YOLOX, and SigLIP2.
   - Train the state encoder, query projection, fusion layers, and latent port against perfect state and retrieval.
   - Combine answer cross-entropy, retrieval InfoNCE, and shuffled-memory contrastive losses.

3. **ManifoldFlow retrieval**
   - Replace oracle retrieval with entity basins and exact RBF routing.
   - Train on synthetic events and recorded state logs.
   - Include identity changes, detection gaps, stale-state replacement, and insufficient-evidence cases.

4. **Vision integration**
   - Feed YOLOX and ByteTrack observations into the event engine.
   - Keep the detector and vision encoder frozen while refining memory adapters.
   - Evaluate only on unseen clips, trajectories, object instances, and question templates.

5. **Conditional LoRA**
   - Stay adapters-only unless retrieval recall@4 reaches 95% while memory-answer accuracy remains below 85%.
   - If triggered, add rank-8 LoRA to query/value/output attention projections in Gemma’s upper six layers at approximately `2e-5`.
   - Keep all original Gemma weights frozen. [Google Gemma tuning guidance](https://ai.google.dev/gemma/docs/tune).

### Benchmark

Create:

- 30 held-out car-occlusion trials.
- 30 held-out cup-shuffle trials.
- Approximately 20,000 training, 2,000 validation, and 2,000 test event/question examples.
- Questions covering present location, hidden container, history, movement, corrections, reappearance, and lack of evidence.

Compare:

- Gemma without memory.
- Gemma with serialized state text.
- ByteTrack-only state.
- Flat typed dictionary/k-nearest-neighbor memory.
- ManifoldFlow latent memory.
- Correct, disabled, shuffled, and oracle memory.
- Gemma-only, SigLIP-only, and union retrieval.
- Adapters-only versus LoRA if triggered.

Acceptance targets:

- Detector mAP50 of at least 0.90 on the controlled set.
- Visible-object identity accuracy of at least 95%.
- Re-identification after 1–5 seconds of occlusion of at least 90%.
- Correct car identity/location in at least 90% of trials.
- Correct cup after a 3–10 second shuffle in at least 85%.
- Unknown/no-evidence precision of at least 90%.
- Retrieval recall@4 of at least 95%.
- Held-out memory-answer accuracy of at least 85%.
- Shuffled-memory accuracy at least 40 points below correct-memory accuracy.
- Native latent memory within three points of the text-memory baseline while using fewer prompt tokens.
- Non-memory instruction accuracy degradation no greater than 5%.

## Public Demo and Operations

### Reviewer experience

Build a responsive FastAPI and vanilla HTML/JavaScript application:

- Landing page explaining the research claim, limitations, architecture, and privacy behavior.
- Curated demo player with play, pause, replay, event timeline, memory inspection, and chat.
- Live-camera tab using browser `getUserMedia`.
- Overlay showing temporary tracker ID, persistent entity ID, class, zone, visibility, relation, and confidence.
- Example questions plus unrestricted short natural-language questions.
- A visible “Why this answer?” panel showing retrieved entities and evidence.
- Clear labeling that the system is a research prototype, can make mistakes, and is not a general physical-world predictor.

### Live browser camera

- Request camera permission only after explicit user action.
- Send resized 640×480 JPEG frames over an authenticated WebSocket at a maximum of five FPS.
- Do not request microphone access.
- Process frames in memory and never write them, derived crops, or chat content to persistent storage.
- Delete the session memory immediately on disconnect or after five minutes of inactivity.
- Generate random, unguessable session IDs; never accept session IDs supplied directly by clients.
- Support two concurrent live sessions per GPU, with a visible FIFO queue beyond that.
- Limit live sessions to ten minutes and questions to thirty per session.
- Offer an immediate “End and erase session” control.
- Make curated scenes available even when the live GPU queue is full.

### Deployment

- Package the backend in a reproducible GPU container.
- Deploy on one NVIDIA L4-class instance with:
  - TLS termination.
  - Reverse proxy and WebSocket support.
  - Health/readiness endpoints.
  - Automatic restart.
  - Read-only application container filesystem.
  - Separate immutable model volume.
- Place the service behind CDN/DDoS protection and a privacy-preserving bot challenge.
- Apply per-IP and per-session rate limiting.
- Allow only JPEG/WebP camera frames under the fixed dimension and byte limits.
- Disable arbitrary file uploads and outbound network access from inference workers.
- Record only aggregate metrics: latency, error type, queue depth, and anonymous session duration.
- Never log frames, prompts, answers, IP addresses, or persistent identifiers.
- Provide a fallback prerecorded rendering if GPU inference is unavailable.
- Run a warmup request after deployment and continuously check curated-scene answers.

Public-demo performance targets:

- Curated mode available at least 99% during the announced review period.
- At least 10 processed FPS for curated video.
- At least five processed FPS per live session with two simultaneous sessions.
- Event-to-memory p95 below 250 ms.
- First answer token below 1.5 seconds and complete short answer below three seconds when not queued.
- No cross-session state leakage in automated concurrency tests.

## Release, Compliance, and Boundaries

Release artifacts must include:

- Dependency and model revision lock.
- Apache-2.0 and MIT notices for YOLOX and ByteTrack.
- Gemma notice and link to its terms.
- Model card describing training data, intended use, limitations, evaluation, and failure cases.
- Plain-language privacy notice stating that webcam frames are transient.
- Terms of use prohibiting sensitive surveillance, identification of people, or harmful use.
- Reproducible benchmark results and representative failure examples.
- Security checklist and incident shutdown procedure.
- A pre-launch license review; this plan is technical guidance, not legal advice.

Scope is limited to one camera, one isolated session, four tabletop object classes, and visibly distinct cups. Human recognition, facial analysis, recording, multi-camera tracking, identical-cup hard mode, physics forecasting, autonomous actions, audio, and large-scale ANN indexing are excluded.

The existing Sonic pipeline and unrelated working-tree changes remain untouched.
