import asyncio

from fastapi.testclient import TestClient

from manifoldflow.public_demo import LatestFrameSlot, create_app


def test_public_demo_curated_flow_and_session_erasure() -> None:
    client = TestClient(create_app(online_check=lambda: True))
    assert client.get("/").status_code == 200
    assert client.get("/healthz").json()["live"] == "unconfigured"
    created = client.post("/api/session").json()
    session = created["session_id"]
    scenarios = client.get("/api/scenarios").json()
    assert {item["id"] for item in scenarios} == {"car-occlusion", "cup-shuffle"}

    frame = client.get(
        "/api/scenarios/car-occlusion/frames/0", params={"session_id": session}
    )
    assert frame.status_code == 200
    answer = client.post(
        "/api/chat",
        json={"session_id": session, "question": "Where is the car?"},
    )
    assert "left" in answer.json()["answer"]

    assert client.delete(f"/api/session/{session}").json() == {"erased": True}
    denied = client.get(
        "/api/scenarios/car-occlusion/frames/0", params={"session_id": session}
    )
    assert denied.status_code == 401


def test_live_socket_reports_unconfigured_without_accepting_frames() -> None:
    client = TestClient(create_app(online_check=lambda: True))
    session = client.post("/api/session").json()["session_id"]
    with client.websocket_connect(f"/api/live/{session}") as websocket:
        assert "not configured" in websocket.receive_json()["error"]


def test_offline_schedule_gates_api_but_not_status_or_index() -> None:
    client = TestClient(create_app(online_check=lambda: False))
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["online"] is False

    denied = client.post("/api/session")
    assert denied.status_code == 503
    assert denied.json()["online"] is False


def test_online_schedule_allows_session_creation() -> None:
    client = TestClient(create_app(online_check=lambda: True))
    created = client.post("/api/session")
    assert created.status_code == 200
    assert "session_id" in created.json()


def test_latest_frame_slot_supersedes_pending_frames() -> None:
    async def exercise() -> tuple[bytes | None, bytes | None, int]:
        slot = LatestFrameSlot()
        slot.put(b"old")
        slot.put(b"latest")
        first = await slot.get()
        slot.close()
        second = await slot.get()
        return first, second, slot.superseded

    first, second, superseded = asyncio.run(exercise())

    assert first == b"latest"
    assert second is None
    assert superseded == 1


def test_latest_frame_slot_drains_pending_frame_after_close() -> None:
    async def exercise() -> tuple[bytes | None, bytes | None]:
        slot = LatestFrameSlot()
        slot.put(b"final")
        slot.close()
        return await slot.get(), await slot.get()

    assert asyncio.run(exercise()) == (b"final", None)
