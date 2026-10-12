"""Reusable prerecorded-video replay and evaluation for the tabletop world model."""

from __future__ import annotations

import asyncio
import html
import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class ReplayProcessor(Protocol):
    def process(self, session_id: str, jpeg: bytes) -> dict[str, Any]: ...


class ReplayAnswerer(Protocol):
    def answer(self, session_id: str, question: str) -> dict[str, Any]: ...


class AnnotatedReplayProcessor:
    """Ask timestamped benchmark questions immediately after frame processing."""

    def __init__(
        self,
        processor: ReplayProcessor,
        answerer: ReplayAnswerer,
        questions: list[dict[str, Any]],
    ) -> None:
        self.processor = processor
        self.answerer = answerer
        self.questions = sorted(questions, key=lambda item: float(item["timestamp_s"]))
        self._next_question = 0

    def process_at(
        self, session_id: str, jpeg: bytes, *, timestamp: float
    ) -> dict[str, Any]:
        process_at = getattr(self.processor, "process_at", None)
        if process_at is not None:
            result = process_at(session_id, jpeg, timestamp=timestamp)
        else:
            result = self.processor.process(session_id, jpeg)
        answers = []
        while self._next_question < len(self.questions):
            question = self.questions[self._next_question]
            if float(question["timestamp_s"]) > timestamp:
                break
            response = self.answerer.answer(session_id, str(question["question"]))
            answers.append(
                {
                    "id": question.get("id", f"question-{self._next_question + 1}"),
                    "scheduled_timestamp_s": float(question["timestamp_s"]),
                    "question": question["question"],
                    "expected_answers": question.get("expected_answers", []),
                    "response": response,
                }
            )
            self._next_question += 1
        result["answers"] = answers
        return result


@dataclass(frozen=True)
class VideoFrame:
    source_frame: int
    timestamp_s: float
    jpeg: bytes


@dataclass(frozen=True)
class ReplayRecord:
    source_frame: int
    timestamp_s: float
    processing_ms: float
    queue_delay_ms: float
    superseded_frames: int
    result: dict[str, Any]


@dataclass(frozen=True)
class ReplayRun:
    mode: str
    records: tuple[ReplayRecord, ...]
    offered_frames: int
    superseded_frames: int
    wall_seconds: float

    def summary(self) -> dict[str, Any]:
        processing = [item.processing_ms for item in self.records]
        queue_delay = [item.queue_delay_ms for item in self.records]
        return {
            "mode": self.mode,
            "offered_frames": self.offered_frames,
            "processed_frames": len(self.records),
            "superseded_frames": self.superseded_frames,
            "wall_seconds": self.wall_seconds,
            "mean_processing_ms": _mean(processing),
            "p95_processing_ms": _percentile(processing, 0.95),
            "mean_queue_delay_ms": _mean(queue_delay),
            "p95_queue_delay_ms": _percentile(queue_delay, 0.95),
        }


def iter_video_frames(
    path: str | Path,
    *,
    send_fps: float = 5.0,
    jpeg_quality: int = 90,
    max_frames: int | None = None,
) -> Iterator[VideoFrame]:
    """Decode a video and emit JPEG frames at the browser's configured cadence."""
    if send_fps <= 0:
        raise ValueError("send_fps must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Video replay requires opencv-python-headless") from error

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if native_fps <= 0:
        native_fps = 30.0
    next_emit = 0.0
    emitted = 0
    source_frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp_s = source_frame / native_fps
            if timestamp_s + 1e-9 >= next_emit:
                encoded, buffer = cv2.imencode(
                    ".jpg",
                    image,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        jpeg_quality,
                        # Source footage is already chroma-subsampled by its
                        # own video codec. Re-subsampling here compounds that
                        # loss and measurably degrades subtle color-name
                        # classification (e.g. pink vs. mint cups); 4:4:4
                        # keeps this re-encode from adding a second lossy
                        # chroma pass on top of the source video's own.
                        cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                        cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
                    ],
                )
                if not encoded:
                    raise RuntimeError(f"failed to encode video frame {source_frame}")
                yield VideoFrame(source_frame, timestamp_s, buffer.tobytes())
                emitted += 1
                next_emit += 1.0 / send_fps
                if max_frames is not None and emitted >= max_frames:
                    break
            source_frame += 1
    finally:
        capture.release()


def replay_deterministic(
    frames: Iterable[VideoFrame], processor: ReplayProcessor, *, session_id: str
) -> ReplayRun:
    """Process every offered frame in source order without wall-clock dropping."""
    started = time.monotonic()
    records: list[ReplayRecord] = []
    for frame in frames:
        inference_started = time.monotonic()
        result = _process_at_source_time(processor, session_id, frame)
        records.append(
            ReplayRecord(
                source_frame=frame.source_frame,
                timestamp_s=frame.timestamp_s,
                processing_ms=(time.monotonic() - inference_started) * 1000.0,
                queue_delay_ms=0.0,
                superseded_frames=0,
                result=result,
            )
        )
    return ReplayRun(
        mode="deterministic",
        records=tuple(records),
        offered_frames=len(records),
        superseded_frames=0,
        wall_seconds=time.monotonic() - started,
    )


@dataclass(frozen=True)
class _ScheduledFrame:
    frame: VideoFrame
    due_at: float


class _LatestVideoSlot:
    def __init__(self) -> None:
        self._pending: _ScheduledFrame | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self.superseded = 0

    def put(self, frame: _ScheduledFrame) -> None:
        if self._closed:
            return
        if self._pending is not None:
            self.superseded += 1
        self._pending = frame
        self._ready.set()

    async def get(self) -> _ScheduledFrame | None:
        while True:
            await self._ready.wait()
            pending = self._pending
            self._pending = None
            self._ready.clear()
            if pending is not None:
                if self._closed:
                    self._ready.set()
                return pending
            if self._closed:
                return None

    def close(self) -> None:
        self._closed = True
        self._ready.set()


async def replay_realtime(
    frames: Iterable[VideoFrame],
    processor: ReplayProcessor,
    *,
    session_id: str,
    playback_rate: float = 1.0,
) -> ReplayRun:
    """Replay source timing through a one-frame latest-value mailbox."""
    if playback_rate <= 0:
        raise ValueError("playback_rate must be positive")
    slot = _LatestVideoSlot()
    records: list[ReplayRecord] = []
    offered = 0
    wall_started = time.monotonic()
    first_timestamp: float | None = None

    async def produce() -> None:
        nonlocal offered, first_timestamp
        try:
            for frame in frames:
                if first_timestamp is None:
                    first_timestamp = frame.timestamp_s
                offset = (frame.timestamp_s - first_timestamp) / playback_rate
                due_at = wall_started + offset
                delay = due_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                slot.put(_ScheduledFrame(frame, due_at))
                offered += 1
        finally:
            slot.close()

    async def consume() -> None:
        while True:
            scheduled = await slot.get()
            if scheduled is None:
                return
            inference_started = time.monotonic()
            result = await asyncio.to_thread(
                _process_at_source_time, processor, session_id, scheduled.frame
            )
            records.append(
                ReplayRecord(
                    source_frame=scheduled.frame.source_frame,
                    timestamp_s=scheduled.frame.timestamp_s,
                    processing_ms=(time.monotonic() - inference_started) * 1000.0,
                    queue_delay_ms=max(
                        0.0, (inference_started - scheduled.due_at) * 1000.0
                    ),
                    superseded_frames=slot.superseded,
                    result=result,
                )
            )

    producer = asyncio.create_task(produce())
    consumer = asyncio.create_task(consume())
    try:
        await asyncio.gather(producer, consumer)
    finally:
        slot.close()
        for task in (producer, consumer):
            if not task.done():
                task.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True)
    return ReplayRun(
        mode="realtime",
        records=tuple(records),
        offered_frames=offered,
        superseded_frames=slot.superseded,
        wall_seconds=time.monotonic() - wall_started,
    )


def _process_at_source_time(
    processor: ReplayProcessor, session_id: str, frame: VideoFrame
) -> dict[str, Any]:
    process_at = getattr(processor, "process_at", None)
    if process_at is not None:
        return process_at(session_id, frame.jpeg, timestamp=frame.timestamp_s)
    return processor.process(session_id, frame.jpeg)


def load_annotations(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("annotation schema_version must be 1")
    checkpoints = data.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("annotations must contain a checkpoints list")
    previous = -1.0
    for checkpoint in checkpoints:
        timestamp = float(checkpoint["timestamp_s"])
        if timestamp < 0 or timestamp < previous:
            raise ValueError("checkpoint timestamps must be non-negative and sorted")
        if not isinstance(checkpoint.get("entities", []), list):
            raise ValueError("checkpoint entities must be a list")
        previous = timestamp
    return data


def annotation_template(video_name: str = "scene.mp4") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "video": video_name,
        "scenario": "replace-with-scenario-name",
        "notes": "Boxes use normalized [x1, y1, x2, y2] coordinates.",
        "checkpoints": [
            {
                "timestamp_s": 0.0,
                "entities": [
                    {
                        "id": "red-cup",
                        "class": "cup",
                        "label": "red",
                        "zone": "middle_left",
                        "visibility": "visible",
                        "box": [0.05, 0.2, 0.25, 0.8],
                    }
                ],
                "relations": [],
            }
        ],
        "expected_events": [
            {
                "type": "zone_changed",
                "entity": "red-cup",
                "start_s": 1.0,
                "end_s": 2.0,
            }
        ],
        "questions": [
            {
                "id": "ball-location-1",
                "timestamp_s": 2.0,
                "question": "Where is the ball?",
                "expected_answers": ["The ball is visible in the middle left zone."],
            }
        ],
    }


def evaluate_replay(
    run: ReplayRun,
    annotations: dict[str, Any],
    *,
    tolerance_s: float = 0.35,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score annotated checkpoints and event windows against a replay timeline."""
    counts: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    identity_votes: dict[str, Counter[str]] = {}
    identity_history: dict[str, list[str]] = {}
    label_history: dict[str, list[str]] = {}

    for checkpoint in annotations.get("checkpoints", []):
        timestamp = float(checkpoint["timestamp_s"])
        record = _nearest_record(run.records, timestamp)
        counts["checkpoints"] += 1
        if record is None or abs(record.timestamp_s - timestamp) > tolerance_s:
            counts["missed_checkpoints"] += 1
            failures.append({"timestamp_s": timestamp, "issue": "no nearby frame"})
            continue
        predicted = record.result.get("entities", [])
        expected = checkpoint.get("entities", [])
        matches = _match_entities(expected, predicted, iou_threshold)
        counts["expected_entities"] += len(expected)
        counts["predicted_entities"] += len(predicted)
        counts["matched_entities"] += len(matches)
        gt_to_pred: dict[str, str] = {}
        for expected_index, predicted_index, overlap in matches:
            truth = expected[expected_index]
            guess = predicted[predicted_index]
            gt_id = str(truth["id"])
            pred_id = str(guess["id"])
            gt_to_pred[gt_id] = pred_id
            identity_votes.setdefault(pred_id, Counter())[gt_id] += 1
            identity_history.setdefault(gt_id, []).append(pred_id)
            predicted_label = str(guess.get("label", ""))
            if predicted_label:
                label_history.setdefault(pred_id, []).append(predicted_label)
            if "box" in truth:
                counts["box_matches"] += 1
                sums["box_iou"] += overlap
            for field in ("label", "zone", "visibility"):
                if field in truth:
                    counts[f"{field}_total"] += 1
                    counts[f"{field}_correct"] += str(guess.get(field, "")) == str(
                        truth[field]
                    )
        unmatched = [
            item["id"]
            for index, item in enumerate(expected)
            if index not in {match[0] for match in matches}
        ]
        if unmatched:
            failures.append(
                {
                    "timestamp_s": timestamp,
                    "issue": "unmatched entities",
                    "entities": unmatched,
                }
            )
        visible_expected = [
            item for item in expected if item.get("visibility", "visible") == "visible"
        ]
        observations = record.result.get("observations", [])
        observation_matches = _match_entities(
            visible_expected, observations, iou_threshold
        )
        counts["visible_expected_entities"] += len(visible_expected)
        counts["observed_entities"] += len(observations)
        counts["matched_observations"] += len(observation_matches)

        annotated_relations = checkpoint.get("relations", [])
        counts["expected_relations"] += len(annotated_relations)
        expected_relations = set()
        for item in annotated_relations:
            subject = gt_to_pred.get(str(item["subject"]))
            target = gt_to_pred.get(str(item["target"]))
            if subject and target:
                expected_relations.add((subject, str(item["relation"]), target))
        predicted_relations = {
            (str(item["subject"]), str(item["relation"]), str(item["target"]))
            for item in record.result.get("relations", [])
            if item.get("active", True)
        }
        counts["predicted_relations"] += len(predicted_relations)
        counts["matched_relations"] += len(expected_relations & predicted_relations)

    for history in identity_history.values():
        counts["identity_matches"] += len(history)
        counts["identity_consistent_matches"] += Counter(history).most_common(1)[0][1]
        counts["identity_transitions"] += max(0, len(history) - 1)
        counts["identity_switches"] += sum(
            current != previous
            for previous, current in zip(history, history[1:], strict=False)
        )
    for history in label_history.values():
        counts["label_transitions"] += max(0, len(history) - 1)
        counts["label_switches"] += sum(
            current != previous
            for previous, current in zip(history, history[1:], strict=False)
        )

    identity_map = {
        predicted: votes.most_common(1)[0][0]
        for predicted, votes in identity_votes.items()
        if votes
    }
    predicted_events = _unique_events(run.records, identity_map)
    used_events: set[int] = set()
    for expected in annotations.get("expected_events", []):
        counts["expected_events"] += 1
        for index, event in enumerate(predicted_events):
            if index in used_events or event["type"] != expected["type"]:
                continue
            if (
                not float(expected["start_s"])
                <= event["timestamp_s"]
                <= float(expected["end_s"])
            ):
                continue
            if expected.get("entity") and event.get("entity") != expected["entity"]:
                continue
            if expected.get("related") and event.get("related") != expected["related"]:
                continue
            used_events.add(index)
            counts["matched_events"] += 1
            break

    for record in run.records:
        for answer in record.result.get("answers", []):
            expected_answers = answer.get("expected_answers", [])
            if isinstance(expected_answers, str):
                expected_answers = [expected_answers]
            actual = str(answer.get("response", {}).get("answer", ""))
            counts["question_total"] += 1
            if any(
                _normalized_text(actual) == _normalized_text(item)
                for item in expected_answers
            ):
                counts["question_correct"] += 1
            else:
                failures.append(
                    {
                        "timestamp_s": record.timestamp_s,
                        "issue": "incorrect answer",
                        "question": answer.get("question"),
                        "expected_answers": expected_answers,
                        "actual": actual,
                    }
                )

    metrics = {
        "checkpoint_coverage": _ratio(
            counts["checkpoints"] - counts["missed_checkpoints"], counts["checkpoints"]
        ),
        "entity_precision": _ratio(
            counts["matched_entities"], counts["predicted_entities"]
        ),
        "entity_recall": _ratio(
            counts["matched_entities"], counts["expected_entities"]
        ),
        "observation_precision": _ratio(
            counts["matched_observations"], counts["observed_entities"]
        ),
        "observation_recall": _ratio(
            counts["matched_observations"], counts["visible_expected_entities"]
        ),
        "mean_box_iou": _ratio(sums["box_iou"], counts["box_matches"]),
        "label_accuracy": _ratio(counts["label_correct"], counts["label_total"]),
        "identity_consistency": _ratio(
            counts["identity_consistent_matches"], counts["identity_matches"]
        ),
        "identity_switch_rate": _ratio(
            counts["identity_switches"], counts["identity_transitions"]
        ),
        "label_switch_rate": _ratio(
            counts["label_switches"], counts["label_transitions"]
        ),
        "zone_accuracy": _ratio(counts["zone_correct"], counts["zone_total"]),
        "visibility_accuracy": _ratio(
            counts["visibility_correct"], counts["visibility_total"]
        ),
        "relation_precision": _ratio(
            counts["matched_relations"], counts["predicted_relations"]
        ),
        "relation_recall": _ratio(
            counts["matched_relations"], counts["expected_relations"]
        ),
        "expected_event_recall": _ratio(
            counts["matched_events"], counts["expected_events"]
        ),
        "question_accuracy": _ratio(
            counts["question_correct"], counts["question_total"]
        ),
    }
    return {
        "metrics": metrics,
        "counts": dict(counts),
        "failures": failures[:100],
    }


def write_replay_artifacts(
    output_dir: str | Path,
    run: ReplayRun,
    *,
    evaluation: dict[str, Any] | None = None,
    video: str | Path | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timeline = output / "timeline.jsonl"
    timeline.write_text(
        "".join(
            json.dumps(asdict(item), sort_keys=True) + "\n" for item in run.records
        ),
        encoding="utf-8",
    )
    summary = run.summary()
    if video is not None:
        summary["video"] = str(video)
    if evaluation is not None:
        summary["evaluation"] = evaluation
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = output / "report.html"
    report_path.write_text(_html_report(summary), encoding="utf-8")
    return {"timeline": timeline, "summary": summary_path, "report": report_path}


def _nearest_record(
    records: tuple[ReplayRecord, ...], timestamp_s: float
) -> ReplayRecord | None:
    if not records:
        return None
    return min(records, key=lambda item: abs(item.timestamp_s - timestamp_s))


def _match_entities(expected, predicted, iou_threshold):
    candidates: list[tuple[float, int, int, float]] = []
    for expected_index, truth in enumerate(expected):
        for predicted_index, guess in enumerate(predicted):
            if truth.get("class") != guess.get("class"):
                continue
            if "box" in truth and "box" in guess:
                overlap = _box_iou(truth["box"], guess["box"])
                if overlap < iou_threshold:
                    continue
                score = overlap
            else:
                overlap = 0.0
                score = 2.0 if truth.get("label") == guess.get("label") else 1.0
            candidates.append((score, expected_index, predicted_index, overlap))
    matches = []
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    for _, expected_index, predicted_index, overlap in sorted(candidates, reverse=True):
        if expected_index in used_expected or predicted_index in used_predicted:
            continue
        matches.append((expected_index, predicted_index, overlap))
        used_expected.add(expected_index)
        used_predicted.add(predicted_index)
    return matches


def _box_iou(first, second) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _unique_events(records, identity_map):
    seen: set[tuple[Any, ...]] = set()
    events = []
    for record in records:
        for item in record.result.get("events", []):
            key = (
                item.get("type"),
                item.get("entity"),
                item.get("related"),
                item.get("frame"),
            )
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "type": item.get("type"),
                    "entity": identity_map.get(item.get("entity"), item.get("entity")),
                    "related": identity_map.get(
                        item.get("related"), item.get("related")
                    ),
                    "timestamp_s": record.timestamp_s,
                }
            )
    return events


def _html_report(summary: dict[str, Any]) -> str:
    replay_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_display(value))}</td></tr>"
        for key, value in summary.items()
        if key != "evaluation"
    )
    evaluation = summary.get("evaluation") or {}
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_display(value))}</td></tr>"
        for key, value in evaluation.get("metrics", {}).items()
    )
    failures = html.escape(json.dumps(evaluation.get("failures", []), indent=2))
    empty_metrics = "<tr><td>No annotations supplied.</td></tr>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>World-Gemma video evaluation</title><style>
body{{font:16px system-ui;margin:2rem auto;max-width:960px;padding:0 1rem;
color:#17202a}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}
th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
th{{width:55%}}pre{{background:#f5f6f7;padding:1rem;overflow:auto}}.note{{color:#566573}}
</style></head><body><h1>World-Gemma video evaluation</h1>
<p class="note">Generated locally from prerecorded footage.
Source frames are not embedded.</p>
<h2>Replay</h2><table>{replay_rows}</table>
<h2>Annotated metrics</h2><table>{metric_rows or empty_metrics}</table>
<h2>Failures</h2><pre>{failures}</pre></body></html>"""


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _normalized_text(value: Any) -> str:
    text = " ".join(str(value).lower().split())
    return text.rstrip(".!?")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.5))
    return ordered[index]
