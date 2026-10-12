import asyncio
import io
import json
import time

from manifoldflow.world_video import (
    AnnotatedReplayProcessor,
    ReplayRecord,
    ReplayRun,
    VideoFrame,
    annotation_template,
    evaluate_replay,
    iter_video_frames,
    load_annotations,
    replay_deterministic,
    replay_realtime,
    write_replay_artifacts,
)


def test_iter_video_frames_avoids_chroma_subsampling(tmp_path) -> None:
    cv2 = __import__("cv2")
    from PIL import Image, JpegImagePlugin

    numpy = __import__("numpy")
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32)
    )
    frame = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
    frame[:, :16] = (200, 20, 180)  # a subtle, near-confusable pastel patch
    frame[:, 16:] = (60, 200, 140)
    for _ in range(3):
        writer.write(frame)
    writer.release()

    frames = list(iter_video_frames(video_path, send_fps=5.0))
    assert frames
    with Image.open(io.BytesIO(frames[0].jpeg)) as image:
        image.load()
        # 0 == 4:4:4 in Pillow's JPEG sampling encoding; re-subsampling a
        # frame that came from an already chroma-subsampled source codec
        # compounds color loss and can flip subtle color-name classification
        # (see docs/World-Gemma-Dev-Journal.md, pink-vs-mint finding).
        assert JpegImagePlugin.get_sampling(image) == 0


class _Processor:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen = []

    def process_at(self, session_id, jpeg, *, timestamp):
        time.sleep(self.delay)
        self.seen.append((session_id, jpeg, timestamp))
        return {"frame": len(self.seen) - 1, "entities": [], "events": []}


class _Answerer:
    def answer(self, session_id, question):
        return {"answer": "The ball is visible in the left zone.", "mode": "test"}


def test_deterministic_replay_preserves_every_source_timestamp() -> None:
    processor = _Processor()
    frames = [VideoFrame(index, index / 5, bytes([index])) for index in range(4)]

    run = replay_deterministic(frames, processor, session_id="clip")

    assert [item.source_frame for item in run.records] == [0, 1, 2, 3]
    assert [item[2] for item in processor.seen] == [0.0, 0.2, 0.4, 0.6]
    assert run.summary()["superseded_frames"] == 0


def test_evaluator_reports_identity_switches_and_label_flicker() -> None:
    records = []
    for index, (entity_id, label) in enumerate(
        (("pred-a", "amber"), ("pred-b", "violet"), ("pred-b", "amber"))
    ):
        records.append(
            ReplayRecord(
                source_frame=index,
                timestamp_s=float(index),
                processing_ms=1.0,
                queue_delay_ms=0.0,
                superseded_frames=0,
                result={
                    "entities": [
                        {
                            "id": entity_id,
                            "class": "cup",
                            "label": label,
                            "box": [0.1, 0.1, 0.3, 0.5],
                        }
                    ],
                    "observations": [],
                    "relations": [],
                    "events": [],
                },
            )
        )
    run = ReplayRun(
        mode="deterministic",
        records=tuple(records),
        offered_frames=3,
        superseded_frames=0,
        wall_seconds=0.1,
    )
    annotations = {
        "checkpoints": [
            {
                "timestamp_s": float(index),
                "entities": [
                    {
                        "id": "physical-cup",
                        "class": "cup",
                        "label": "amber",
                        "box": [0.1, 0.1, 0.3, 0.5],
                    }
                ],
                "relations": [],
            }
            for index in range(3)
        ]
    }

    evaluation = evaluate_replay(run, annotations)

    assert evaluation["counts"]["identity_switches"] == 1
    assert evaluation["metrics"]["identity_consistency"] == 2 / 3
    assert evaluation["metrics"]["identity_switch_rate"] == 0.5
    assert evaluation["metrics"]["label_switch_rate"] == 1.0


def test_realtime_replay_supersedes_stale_frames_but_processes_last() -> None:
    processor = _Processor(delay=0.025)
    frames = [VideoFrame(index, index * 0.003, bytes([index])) for index in range(8)]

    run = asyncio.run(
        replay_realtime(frames, processor, session_id="clip", playback_rate=1.0)
    )

    assert run.superseded_frames > 0
    assert len(run.records) < len(frames)
    assert run.records[-1].source_frame == frames[-1].source_frame
    assert processor.seen[-1][1] == frames[-1].jpeg


def test_annotations_evaluate_entities_relations_and_events(tmp_path) -> None:
    result = {
        "observations": [
            {
                "tracker_id": 1,
                "class": "cup",
                "label": "red",
                "zone": "left",
                "visibility": "visible",
                "box": [0.1, 0.1, 0.3, 0.5],
            },
            {
                "tracker_id": 2,
                "class": "occluder_box",
                "label": "",
                "zone": "center",
                "visibility": "visible",
                "box": [0.4, 0.1, 0.7, 0.8],
            },
        ],
        "entities": [
            {
                "id": "cup-0001",
                "class": "cup",
                "label": "red",
                "zone": "left",
                "visibility": "visible",
                "box": [0.1, 0.1, 0.3, 0.5],
            },
            {
                "id": "occluder_box-0002",
                "class": "occluder_box",
                "label": "",
                "zone": "center",
                "visibility": "visible",
                "box": [0.4, 0.1, 0.7, 0.8],
            },
        ],
        "relations": [
            {
                "subject": "cup-0001",
                "relation": "contained_by",
                "target": "occluder_box-0002",
                "active": True,
            }
        ],
        "events": [
            {
                "type": "zone_changed",
                "entity": "cup-0001",
                "related": None,
                "frame": 4,
            }
        ],
        "answers": [
            {
                "question": "Where is the ball?",
                "expected_answers": ["The ball is visible in the left zone"],
                "response": {"answer": "The ball is visible in the left zone."},
            }
        ],
    }
    run = ReplayRun(
        "deterministic",
        (ReplayRecord(30, 1.0, 10.0, 0.0, 0, result),),
        1,
        0,
        0.01,
    )
    annotations = {
        "schema_version": 1,
        "checkpoints": [
            {
                "timestamp_s": 1.0,
                "entities": [
                    {
                        "id": "red-cup",
                        "class": "cup",
                        "label": "red",
                        "zone": "left",
                        "visibility": "visible",
                        "box": [0.1, 0.1, 0.3, 0.5],
                    },
                    {
                        "id": "box",
                        "class": "occluder_box",
                        "zone": "center",
                        "visibility": "visible",
                        "box": [0.4, 0.1, 0.7, 0.8],
                    },
                ],
                "relations": [
                    {
                        "subject": "red-cup",
                        "relation": "contained_by",
                        "target": "box",
                    }
                ],
            }
        ],
        "expected_events": [
            {
                "type": "zone_changed",
                "entity": "red-cup",
                "start_s": 0.8,
                "end_s": 1.2,
            }
        ],
    }

    evaluation = evaluate_replay(run, annotations)
    paths = write_replay_artifacts(tmp_path, run, evaluation=evaluation)

    stable_metrics = {
        key: value
        for key, value in evaluation["metrics"].items()
        if key not in {"identity_switch_rate", "label_switch_rate"}
    }
    assert all(value == 1.0 for value in stable_metrics.values())
    assert evaluation["metrics"]["identity_switch_rate"] is None
    assert evaluation["metrics"]["label_switch_rate"] is None
    assert paths["report"].exists()
    assert json.loads(paths["summary"].read_text())["processed_frames"] == 1


def test_annotation_template_round_trips(tmp_path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotation_template("clip.mp4")))

    loaded = load_annotations(path)

    assert loaded["video"] == "clip.mp4"


def test_annotated_processor_asks_due_questions_once() -> None:
    processor = AnnotatedReplayProcessor(
        _Processor(),
        _Answerer(),
        [
            {
                "id": "where-ball",
                "timestamp_s": 0.5,
                "question": "Where is the ball?",
                "expected_answers": ["The ball is visible in the left zone"],
            }
        ],
    )

    before = processor.process_at("clip", b"a", timestamp=0.4)
    due = processor.process_at("clip", b"b", timestamp=0.6)
    after = processor.process_at("clip", b"c", timestamp=0.8)

    assert before["answers"] == []
    assert due["answers"][0]["id"] == "where-ball"
    assert after["answers"] == []
