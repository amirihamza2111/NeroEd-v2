import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402
import ollama_client  # noqa: E402
import session_manager  # noqa: E402


def fake_embed(text: str) -> list[float]:
    """Deterministic bag-of-letters embedding so similar text scores higher."""
    vec = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            vec[ord(ch) - ord("a")] += 1.0
    return vec


class FakeReranker:
    """Scores a pair by how many query words appear in the chunk."""

    def predict(self, pairs):
        scores = []
        for query, chunk in pairs:
            words = set(query.lower().split())
            scores.append(float(sum(w in chunk.lower() for w in words)))
        return scores


@pytest.fixture(autouse=True)
def clean_sessions():
    session_manager.sessions.clear()
    yield
    session_manager.sessions.clear()


@pytest.fixture
def offline_models(monkeypatch):
    """Replace every model call with fast deterministic fakes."""
    monkeypatch.setattr(main, "embed", fake_embed)
    monkeypatch.setattr(ollama_client, "embed", fake_embed)
    monkeypatch.setattr(ollama_client, "_reranker", FakeReranker())
