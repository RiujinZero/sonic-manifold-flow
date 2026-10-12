"""Append-only usage analytics for the public world demo.

Logs event metadata (timestamps, session ids, answer modes, question
*lengths*) to a JSON-lines file for later review -- never the raw text
a visitor typed, matching the demo's existing "not retained" privacy
stance for anything a visitor says. Session ids are per-session opaque
tokens already (see world_demo.DemoRuntime.new_session), not personal
identifiers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "logs" / "world-demo" / "analytics.jsonl"
_LOCK = Lock()


def _log_path() -> Path:
    override = os.environ.get("MANIFOLDFLOW_ANALYTICS_LOG")
    return Path(override) if override else _DEFAULT_PATH


def log_event(event: str, **fields: object) -> None:
    path = _log_path()
    record = {"ts": time.time(), "event": event, **fields}
    line = json.dumps(record, sort_keys=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        # Analytics are best-effort; a logging failure (e.g. read-only
        # filesystem without the logs volume mounted) must never break
        # the demo itself.
        pass
