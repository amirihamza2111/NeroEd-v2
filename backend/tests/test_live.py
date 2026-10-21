"""
End-to-end tests against a real Ollama server and the real reranker.

    NEROED_LIVE=1 pytest -m live -v

Requires `ollama serve` with llama3 and nomic-embed-text pulled.
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

import main

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("NEROED_LIVE") != "1", reason="set NEROED_LIVE=1"),
]

DOC = """Cellular Respiration and Classical Mechanics — Study Guide

1. Mitochondria
The mitochondrion is known as the powerhouse of the cell. It is the site of
the Krebs cycle (also called the citric acid cycle) and oxidative
phosphorylation. One glucose molecule yields approximately 36 to 38 ATP
molecules through aerobic respiration.

2. Photosynthesis
Photosynthesis takes place in the chloroplast. The overall equation is
6CO2 + 6H2O -> C6H12O6 + 6O2. Chlorophyll absorbs mostly red and blue light.

3. Newton's Laws
Newton's second law states that the net force on an object equals its mass
times its acceleration: F = m * a, where F is force in newtons, m is mass in
kilograms and a is acceleration in metres per second squared.
Kinetic energy is given by KE = 1/2 * m * v^2.

4. Key dates
Hans Krebs described the citric acid cycle in 1937 and received the Nobel
Prize in Physiology or Medicine in 1953.
"""


@pytest.fixture(autouse=True)
def clean_sessions():
    # Override conftest's per-test wipe: these tests share one indexed session.
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def sid(client):
    res = client.post("/upload", files={"file": ("guide.txt", DOC.encode(), "text/plain")})
    assert res.status_code == 202
    sid = res.json()["session_id"]
    status = client.get(f"/status/{sid}").json()
    assert status["status"] == "ready", status
    return sid


def ask(client, sid, question):
    t0 = time.perf_counter()
    res = client.post("/chat", json={"session_id": sid, "message": question, "history": []})
    assert res.status_code == 200
    events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]
    assert "error" not in events[-1], events[-1]
    answer = "".join(e.get("token", "") for e in events)
    print(f"\nQ: {question}\nA ({time.perf_counter() - t0:.1f}s): {answer[:400]}")
    return answer, events[-1]["citations"]


def test_health(client):
    assert client.get("/health").json() == {"ollama": "connected"}


def test_chat_grounded_answer(client, sid):
    answer, citations = ask(client, sid, "Where does the Krebs cycle take place?")
    assert "mitochondri" in answer.lower()
    assert citations and any("Krebs" in c["chunk"] for c in citations)


def test_chat_follow_up_uses_history(client, sid):
    answer, _ = ask(client, sid, "In what year did he describe it?")
    assert "1937" in answer


def test_chat_formula(client, sid):
    answer, _ = ask(client, sid, "What is Newton's second law?")
    assert "m" in answer and "a" in answer and ("F" in answer or "force" in answer.lower())


def test_chat_refuses_out_of_scope(client, sid):
    answer, _ = ask(client, sid, "What is the capital of Australia according to the document?")
    assert "canberra" not in answer.lower() or "could not find" in answer.lower()


def test_chat_french(client, sid):
    answer, _ = ask(client, sid, "Où se déroule la photosynthèse ?")
    assert "chloroplast" in answer.lower()


def test_summarize(client, sid):
    res = client.post("/summarize", json={"session_id": sid})
    assert res.status_code == 200
    summary = res.json()["summary"]
    print("\nSUMMARY:\n", summary[:600])
    assert len(summary) > 200


def test_summarize_long_document_map_reduce(client):
    long_doc = (DOC + "\n") * 25  # ~45k chars, well beyond one context window
    res = client.post("/upload", files={"file": ("long.txt", long_doc.encode(), "text/plain")})
    sid = res.json()["session_id"]
    assert client.get(f"/status/{sid}").json()["status"] == "ready"
    res = client.post("/summarize", json={"session_id": sid})
    assert res.status_code == 200
    assert len(res.json()["summary"]) > 200


def test_flashcards(client, sid):
    data = client.post("/flashcards", json={"session_id": sid}).json()
    print(f"\nflashcards: {data['count']}")
    assert data["count"] >= 5
    assert all({"question", "answer"} <= card.keys() for card in data["flashcards"])


def test_quiz(client, sid):
    data = client.post("/quiz", json={"session_id": sid, "difficulty": "medium"}).json()
    print(f"\nquiz questions: {data['count']}")
    assert data["count"] >= 5
    assert all("question" in q and "type" in q for q in data["questions"])


def test_formulas(client, sid):
    data = client.post("/formulas", json={"session_id": sid}).json()
    print(f"\nformulas: {[f.get('formula') for f in data['formulas']]}")
    assert data["count"] >= 1
