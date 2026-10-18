import inspect
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

import main
import session_manager

DOC = (
    "Photosynthesis converts light energy into chemical energy stored in glucose. "
    "The Krebs cycle takes place in the mitochondria and produces ATP. "
    "Newton's second law states that force equals mass times acceleration. "
) * 20


@pytest.fixture
def client(offline_models):
    with TestClient(main.app) as c:
        yield c


def upload(client, text=DOC, name="notes.txt"):
    res = client.post("/upload", files={"file": (name, text.encode(), "text/plain")})
    return res


def ready_session(client) -> str:
    res = upload(client)
    assert res.status_code == 202
    sid = res.json()["session_id"]
    # TestClient runs background tasks before returning, so indexing is done.
    assert client.get(f"/status/{sid}").json()["status"] == "ready"
    return sid


def sse_events(res) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in res.text.splitlines()
        if line.startswith("data: ")
    ]


# ------------------------------------------------------------------ upload


class TestUpload:
    def test_upload_indexes_document(self, client):
        res = upload(client)
        body = res.json()
        assert res.status_code == 202
        assert body["chunk_count"] == len(main.chunk_text(DOC))
        status = client.get(f"/status/{body['session_id']}").json()
        assert status == {"status": "ready", "progress": 100, "error_detail": None}
        session = session_manager.get_session(body["session_id"])
        assert len(session["chunk_embeddings"]) == body["chunk_count"]

    def test_unsupported_type(self, client):
        res = client.post("/upload", files={"file": ("x.png", b"\x89PNG", "image/png")})
        assert res.status_code == 400
        assert "Unsupported" in res.json()["detail"]

    def test_non_utf8_txt_is_400_not_500(self, client):
        res = client.post("/upload", files={"file": ("x.txt", b"\xff\xfe\xfa", "text/plain")})
        assert res.status_code == 400

    def test_empty_document_is_ready_immediately(self, client):
        res = upload(client, text="")
        sid = res.json()["session_id"]
        assert res.json()["chunk_count"] == 0
        assert client.get(f"/status/{sid}").json()["status"] == "ready"

    def test_embedding_failure_sets_error_status(self, client, monkeypatch):
        def broken(text):
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")

        monkeypatch.setattr(main, "embed", broken)
        sid = upload(client).json()["session_id"]
        status = client.get(f"/status/{sid}").json()
        assert status["status"] == "error"
        assert "not running" in status["error_detail"]

    def test_status_unknown_session(self, client):
        assert client.get("/status/nope").status_code == 404


# -------------------------------------------------------------------- chat


class TestChat:
    def test_unknown_session(self, client):
        res = client.post("/chat", json={"session_id": "x", "message": "hi", "history": []})
        assert res.status_code == 404

    def test_not_ready(self, client):
        sid = session_manager.create_session(DOC, ["a"])
        res = client.post("/chat", json={"session_id": sid, "message": "hi", "history": []})
        assert res.status_code == 400
        assert res.json()["detail"] == "Document still indexing"

    def test_streams_tokens_then_citations(self, client, monkeypatch):
        captured = {}

        def stream(system_prompt, history, message):
            captured["system"] = system_prompt
            yield "The Krebs "
            yield "cycle."

        monkeypatch.setattr(main, "chat_stream", stream)
        sid = ready_session(client)
        res = client.post(
            "/chat",
            json={"session_id": sid, "message": "Where does the Krebs cycle happen?", "history": []},
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")

        events = sse_events(res)
        assert "".join(e.get("token", "") for e in events) == "The Krebs cycle."
        done = events[-1]
        assert done["done"] is True
        assert 1 <= len(done["citations"]) <= 5
        assert [c["index"] for c in done["citations"]] == list(range(1, len(done["citations"]) + 1))
        # Retrieved excerpts are injected into the prompt with matching numbers.
        assert "[1] " in captured["system"]
        assert "Krebs" in done["citations"][0]["chunk"]

        history = session_manager.get_session(sid)["chat_history"]
        assert history == [
            {"role": "user", "content": "Where does the Krebs cycle happen?"},
            {"role": "assistant", "content": "The Krebs cycle."},
        ]

    def test_history_is_passed_to_next_turn(self, client, monkeypatch):
        seen = []

        def stream(system_prompt, history, message):
            seen.append(list(history))
            yield "ok"

        monkeypatch.setattr(main, "chat_stream", stream)
        sid = ready_session(client)
        for q in ["first", "second"]:
            client.post("/chat", json={"session_id": sid, "message": q, "history": []})
        assert seen[0] == []
        assert [m["content"] for m in seen[1]] == ["first", "ok"]

    def test_stream_error_is_reported_and_not_recorded(self, client, monkeypatch):
        """Fix 4: errors reach the client and leave history consistent."""

        def stream(system_prompt, history, message):
            yield "partial"
            raise RuntimeError("model 'llama3' not found")

        monkeypatch.setattr(main, "chat_stream", stream)
        sid = ready_session(client)
        res = client.post("/chat", json={"session_id": sid, "message": "q", "history": []})
        events = sse_events(res)
        assert events[-1] == {"error": "model 'llama3' not found"}
        assert not any(e.get("done") for e in events)
        assert session_manager.get_session(sid)["chat_history"] == []

    def test_retrieval_failure_is_502(self, client, monkeypatch):
        sid = ready_session(client)

        def broken(*args, **kwargs):
            raise RuntimeError("Ollama is not running")

        monkeypatch.setattr(main, "retrieve", broken)
        res = client.post("/chat", json={"session_id": sid, "message": "q", "history": []})
        assert res.status_code == 502
        assert "Retrieval failed" in res.json()["detail"]


# ------------------------------------------------------------ study tools


@pytest.mark.parametrize(
    "path,fn_name,payload,result,key",
    [
        ("/summarize", "summarize", {}, "# Summary", "summary"),
        ("/flashcards", "generate_flashcards", {}, [{"id": 1}], "flashcards"),
        ("/quiz", "generate_quiz", {"difficulty": "easy"}, [{"id": 1}, {"id": 2}], "questions"),
        ("/notes", "generate_notes", {"format": "cornell"}, "notes", "notes"),
        ("/formulas", "extract_formulas", {}, [], "formulas"),
    ],
)
class TestStudyTools:
    def test_success(self, client, monkeypatch, path, fn_name, payload, result, key):
        calls = []

        def fake(*args):
            calls.append(args)
            return result

        monkeypatch.setattr(main, fn_name, fake)
        sid = ready_session(client)
        res = client.post(path, json={"session_id": sid, **payload})
        assert res.status_code == 200
        assert res.json()[key] == result
        assert calls[0][0] == DOC
        if "difficulty" in payload:
            assert res.json()["difficulty"] == "easy"
            assert calls[0][1] == "easy"
        if "format" in payload:
            assert calls[0][1] == "cornell"

    def test_unknown_session(self, client, path, fn_name, payload, result, key):
        assert client.post(path, json={"session_id": "nope", **payload}).status_code == 404

    def test_not_ready(self, client, path, fn_name, payload, result, key):
        sid = session_manager.create_session(DOC, ["a"])
        res = client.post(path, json={"session_id": sid, **payload})
        assert res.status_code == 400

    def test_model_failure(self, client, monkeypatch, path, fn_name, payload, result, key):
        def broken(*args):
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")

        monkeypatch.setattr(main, fn_name, broken)
        sid = ready_session(client)
        res = client.post(path, json={"session_id": sid, **payload})
        assert res.status_code in (500, 502)
        assert "not running" in res.json()["detail"]


# ----------------------------------------------------------- misc endpoints


def test_delete_session(client):
    sid = ready_session(client)
    assert client.delete(f"/session/{sid}").json() == {"deleted": True}
    assert client.delete(f"/session/{sid}").json()["deleted"] is False
    assert client.get(f"/status/{sid}").status_code == 404


def test_health_connected(client, monkeypatch):
    monkeypatch.setattr(main.httpx, "get", lambda *a, **k: None)
    assert client.get("/health").json() == {"ollama": "connected"}


def test_health_offline(client, monkeypatch):
    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(main.httpx, "get", refuse)
    assert client.get("/health").json() == {"ollama": "offline"}


def test_cors_allows_frontend_origin(client):
    res = client.options(
        "/upload",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert res.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_other_origin(client):
    res = client.options(
        "/upload",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in res.headers


# ------------------------------------------------------ concurrency (fix 2)


BLOCKING_ENDPOINTS = [
    "chat_endpoint",
    "summarize_document",
    "flashcards_endpoint",
    "quiz_endpoint",
    "notes_endpoint",
    "formulas_endpoint",
    "health",
]


@pytest.mark.parametrize("name", BLOCKING_ENDPOINTS)
def test_blocking_endpoints_run_in_threadpool(name):
    # FastAPI runs plain `def` handlers in a worker thread; `async def`
    # handlers that make blocking calls would stall the event loop.
    assert not inspect.iscoroutinefunction(getattr(main, name))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_slow_generation_does_not_block_other_requests(offline_models, monkeypatch):
    """Run a real server: a 3 s summary must not delay /status or /health."""
    release = threading.Event()

    def slow_summarize(text):
        release.wait(timeout=3)
        return "done"

    monkeypatch.setattr(main, "summarize", slow_summarize)
    monkeypatch.setattr(main.httpx, "get", lambda *a, **k: None)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(main.app, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)

    http = httpx.Client(timeout=10)
    slow = threading.Thread()
    try:
        sid = http.post(
            f"{base}/upload", files={"file": ("n.txt", DOC.encode(), "text/plain")}
        ).json()["session_id"]
        for _ in range(100):
            if http.get(f"{base}/status/{sid}").json()["status"] == "ready":
                break
            time.sleep(0.05)

        slow = threading.Thread(
            target=lambda: http.post(f"{base}/summarize", json={"session_id": sid}, timeout=10)
        )
        slow.start()
        time.sleep(0.3)  # let the summary request get in flight

        start = time.perf_counter()
        assert http.get(f"{base}/health", timeout=2).json() == {"ollama": "connected"}
        assert http.get(f"{base}/status/{sid}", timeout=2).status_code == 200
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"event loop blocked for {elapsed:.2f}s"
        assert slow.is_alive(), "summary should still be running"
    finally:
        release.set()
        slow.join(timeout=5)
        http.close()
        server.should_exit = True
        thread.join(timeout=5)
