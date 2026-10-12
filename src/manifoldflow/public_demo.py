"""Public FastAPI surface for curated and live tabletop demonstrations."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from manifoldflow.world_analytics import log_event
from manifoldflow.world_demo import DemoRuntime
from manifoldflow.world_schedule import is_online, status as schedule_status

try:  # Keep the core package importable without the optional web extra.
    from fastapi import FastAPI, HTTPException, Request, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover - exercised only without optional deps
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = Request = WebSocket = HTMLResponse = JSONResponse = None  # type: ignore[assignment,misc]
    StaticFiles = None  # type: ignore[assignment,misc]

_STATIC = Path(__file__).with_name("static")


class LiveProcessor(Protocol):
    def process(self, session_id: str, jpeg: bytes) -> dict[str, Any]: ...

    def close_session(self, session_id: str) -> None: ...


class LatestFrameSlot:
    """One-frame async mailbox where a new frame supersedes stale pending work."""

    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self.superseded = 0

    def put(self, frame: bytes) -> None:
        if self._closed:
            return
        if self._frame is not None:
            self.superseded += 1
        self._frame = frame
        self._ready.set()

    async def wait(self) -> None:
        await self._ready.wait()

    async def get(self) -> bytes | None:
        while True:
            await self._ready.wait()
            frame = self._frame
            self._frame = None
            self._ready.clear()
            if frame is not None:
                if self._closed:
                    self._ready.set()
                return frame
            if self._closed:
                return None

    def close(self) -> None:
        self._closed = True
        self._ready.set()


@dataclass
class _PublicSession:
    created_at: float
    last_seen: float
    questions: int = 0


def create_app(
    runtime: DemoRuntime | None = None,
    live_processor: LiveProcessor | None = None,
    online_check: Callable[[], bool] | None = None,
) -> Any:
    if FastAPI is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The public demo requires the world extra: pip install -e '.[world]'"
        )

    demo = runtime or DemoRuntime()
    online = online_check or is_online

    def current_status() -> dict[str, Any]:
        return {**schedule_status(), "online": online()}
    sessions: dict[str, _PublicSession] = {}
    active_live_sessions: set[str] = set()
    app = FastAPI(
        title="ManifoldFlow Tabletop World Model",
        docs_url=None,
        redoc_url=None,
    )
    _ALWAYS_OPEN_PATHS = {"/api/status", "/healthz", "/"}

    @app.middleware("http")
    async def gate_offline(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        needs_gate = path.startswith("/api/") and path not in _ALWAYS_OPEN_PATHS
        if needs_gate and not online():
            return JSONResponse(
                {"detail": "The demo is offline right now.", **current_status()},
                status_code=503,
            )
        return await call_next(request)

    scenario_assets = _STATIC / "scenarios"
    if scenario_assets.is_dir():
        # Read-only rendered frames for the recorded curated scenarios
        # (scripts/extract_demo_scenario.py) -- nothing else under static/
        # is exposed this way; world_demo.html itself is served explicitly
        # below, not through this mount.
        app.mount(
            "/static/scenarios",
            StaticFiles(directory=scenario_assets),
            name="scenario-assets",
        )

    def require_session(session_id: str) -> _PublicSession:
        session = sessions.get(session_id)
        now = time.monotonic()
        if session is None or now - session.last_seen > 600:
            if session is not None:
                close(session_id)
            raise HTTPException(status_code=401, detail="Session is missing or expired")
        session.last_seen = now
        return session

    def close(session_id: str) -> None:
        session = sessions.pop(session_id, None)
        demo.close_session(session_id)
        if live_processor is not None:
            live_processor.close_session(session_id)
        if session is not None:
            log_event(
                "session_closed",
                session_id=session_id,
                duration_seconds=round(time.monotonic() - session.created_at, 1),
                questions=session.questions,
            )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (_STATIC / "world_demo.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok", "live": "ready" if live_processor else "unconfigured"}

    @app.get("/api/status")
    async def demo_status() -> dict[str, Any]:
        return current_status()

    @app.post("/api/session")
    async def new_session() -> dict[str, Any]:
        session_id = demo.new_session()
        now = time.monotonic()
        sessions[session_id] = _PublicSession(now, now)
        log_event("session_created", session_id=session_id)
        return {
            "session_id": session_id,
            "expires_after_seconds": 600,
            "live_available": live_processor is not None,
        }

    @app.get("/api/scenarios")
    async def scenarios() -> list[dict[str, Any]]:
        return demo.scenario_catalog()

    @app.get("/api/scenarios/{scenario_id}/frames/{frame}")
    async def scenario_frame(
        scenario_id: str, frame: int, session_id: str
    ) -> dict[str, Any]:
        require_session(session_id)
        try:
            result = demo.render_scenario(session_id, scenario_id, frame)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown scenario") from error
        log_event("scenario_frame", session_id=session_id, scenario_id=scenario_id, frame=frame)
        return result

    @app.post("/api/chat")
    async def chat(request: Request) -> dict[str, Any]:
        payload = await request.json()
        session_id = str(payload.get("session_id", ""))
        question = str(payload.get("question", "")).strip()
        session = require_session(session_id)
        if not question or len(question) > 500:
            raise HTTPException(
                status_code=400, detail="Question must contain 1-500 characters"
            )
        if session.questions >= 30:
            raise HTTPException(status_code=429, detail="Question limit reached")
        session.questions += 1
        result = await asyncio.to_thread(demo.answer, session_id, question)
        log_event(
            "chat",
            session_id=session_id,
            question_length=len(question),
            mode=result.get("mode"),
            evidence_count=len(result.get("evidence") or []),
        )
        return result

    @app.delete("/api/session/{session_id}")
    async def end_session(session_id: str) -> dict[str, bool]:
        require_session(session_id)
        close(session_id)
        return {"erased": True}

    @app.post("/api/session/{session_id}/erase")
    async def beacon_erase(session_id: str) -> dict[str, bool]:
        if session_id in sessions:
            close(session_id)
        return {"erased": True}

    @app.websocket("/api/live/{session_id}")
    async def live(websocket: WebSocket, session_id: str) -> None:
        try:
            require_session(session_id)
        except HTTPException:
            await websocket.close(code=4401)
            return
        if not online():
            await websocket.close(code=1013)
            return
        await websocket.accept()
        if live_processor is None:
            await websocket.send_json(
                {"error": "Live inference is not configured on this deployment."}
            )
            await websocket.close(code=1013)
            return
        if session_id not in active_live_sessions and len(active_live_sessions) >= 2:
            await websocket.send_json(
                {"error": "The two live-camera slots are busy. Try again shortly."}
            )
            await websocket.close(code=4429)
            return
        active_live_sessions.add(session_id)
        slot = LatestFrameSlot()

        async def receive_latest() -> None:
            try:
                while True:
                    jpeg = await websocket.receive_bytes()
                    require_session(session_id)
                    slot.put(jpeg)
            finally:
                slot.close()

        async def process_latest() -> None:
            last_started = 0.0
            while True:
                await slot.wait()
                delay = 0.18 - (time.monotonic() - last_started)
                if delay > 0:
                    await asyncio.sleep(delay)
                jpeg = await slot.get()
                if jpeg is None:
                    return
                last_started = time.monotonic()
                result = await asyncio.to_thread(
                    live_processor.process, session_id, jpeg
                )
                result["superseded_frames"] = slot.superseded
                await websocket.send_text(json.dumps(result))

        receiver = asyncio.create_task(receive_latest())
        processor = asyncio.create_task(process_latest())
        try:
            await asyncio.wait(
                (receiver, processor), return_when=asyncio.FIRST_COMPLETED
            )
        except Exception:
            await websocket.close()
        finally:
            slot.close()
            receiver.cancel()
            processor.cancel()
            await asyncio.gather(receiver, processor, return_exceptions=True)
            active_live_sessions.discard(session_id)

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Run pip install -e '.[world]' first") from error
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
