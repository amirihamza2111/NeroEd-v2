import contextlib
import json

import httpx
import pytest

import ollama_client
from conftest import FakeReranker, fake_embed


def response(status: int, body) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Response(
        status, content=content, request=httpx.Request("POST", "http://test")
    )


class PostRecorder:
    """Stands in for httpx.post and records every call."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.handler(url, kwargs)


def chat_reply(text: str):
    return lambda url, kw: response(200, {"message": {"content": text}})


# ---------------------------------------------------------------- embed (fix 1)


class TestEmbed:
    def test_returns_embedding_and_sets_long_timeout(self, monkeypatch):
        post = PostRecorder(lambda url, kw: response(200, {"embeddings": [[0.1, 0.2]]}))
        monkeypatch.setattr(ollama_client._client, "post", post)

        assert ollama_client.embed("hello") == pytest.approx([0.1, 0.2])
        assert post.calls[0]["url"] == "/api/embed"
        assert post.calls[0]["json"]["input"] == ["hello"]

        timeout = post.calls[0]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        # Must comfortably exceed a cold model load, unlike httpx's 5 s default.
        assert timeout.read >= 60

    def test_timeout_is_reported_as_timeout(self, monkeypatch):
        def boom(url, kw):
            raise httpx.ReadTimeout("slow")

        monkeypatch.setattr(ollama_client._client, "post", PostRecorder(boom))
        with pytest.raises(RuntimeError, match="timed out"):
            ollama_client.embed("hello")

    def test_connection_refused(self, monkeypatch):
        def boom(url, kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(ollama_client._client, "post", PostRecorder(boom))
        with pytest.raises(RuntimeError, match="not running"):
            ollama_client.embed("hello")

    def test_http_error_surfaces_ollama_message(self, monkeypatch):
        monkeypatch.setattr(
            ollama_client._client,
            "post",
            PostRecorder(lambda url, kw: response(404, {"error": "model not found"})),
        )
        with pytest.raises(RuntimeError, match="model not found"):
            ollama_client.embed("hello")


# ------------------------------------------------------- retrieval + reranking


class TestRetrieve:
    def test_empty(self):
        assert ollama_client.retrieve("q", [], []) == []

    def test_ranks_relevant_chunk_first(self, monkeypatch):
        monkeypatch.setattr(ollama_client, "embed", fake_embed)
        chunks = [
            "zzz qqq xxx",
            "photosynthesis converts light into chemical energy",
            "bbb ccc ddd",
        ]
        embs = [fake_embed(c) for c in chunks]
        result = ollama_client.retrieve("photosynthesis light energy", chunks, embs)
        assert result[0]["index"] == 1
        assert {"chunk", "score", "index"} <= result[0].keys()

    def test_returns_at_most_twenty_candidates(self, monkeypatch):
        monkeypatch.setattr(ollama_client, "embed", fake_embed)
        chunks = [f"chunk number {i} " + "a" * i for i in range(50)]
        embs = [fake_embed(c) for c in chunks]
        assert len(ollama_client.retrieve("chunk", chunks, embs, top_k=5)) == 20

    def test_scores_are_normalised(self, monkeypatch):
        monkeypatch.setattr(ollama_client, "embed", fake_embed)
        chunks = ["apple banana", "banana cherry", "cherry date"]
        embs = [fake_embed(c) for c in chunks]
        for item in ollama_client.retrieve("banana", chunks, embs):
            assert 0.0 <= item["score"] <= 1.0


class TestRerank:
    def test_empty(self):
        assert ollama_client.rerank("q", []) == []

    def test_reorders_and_truncates(self, monkeypatch):
        monkeypatch.setattr(ollama_client, "_reranker", FakeReranker())
        candidates = [
            {"chunk": "nothing here", "score": 0.9, "index": 0},
            {"chunk": "mitochondria powerhouse cell", "score": 0.1, "index": 1},
            {"chunk": "cell wall", "score": 0.5, "index": 2},
        ]
        out = ollama_client.rerank("mitochondria cell", candidates, top_k=2)
        assert [c["index"] for c in out] == [1, 2]
        assert all("reranker_score" in c for c in out)


# ------------------------------------------------- chat + context window (fix 3)


class TestChat:
    def test_requests_full_context_window(self, monkeypatch):
        post = PostRecorder(chat_reply("hi"))
        monkeypatch.setattr(ollama_client._client, "post", post)
        assert ollama_client.chat("sys", [], "user") == "hi"
        assert post.calls[0]["json"]["options"]["num_ctx"] == 8192

    def test_history_is_forwarded_in_order(self, monkeypatch):
        post = PostRecorder(chat_reply("ok"))
        monkeypatch.setattr(ollama_client._client, "post", post)
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        ollama_client.chat("sys", history, "c")
        roles = [m["role"] for m in post.calls[0]["json"]["messages"]]
        assert roles == ["system", "user", "assistant", "user"]


def fake_stream(resp: httpx.Response, recorder: list | None = None):
    def _stream(method, url, **kwargs):
        if recorder is not None:
            recorder.append(kwargs)
        return contextlib.nullcontext(resp)

    return _stream


def ndjson(*objs) -> bytes:
    return b"\n".join(json.dumps(o).encode() for o in objs) + b"\n"


class TestChatStream:
    def test_yields_tokens_until_done(self, monkeypatch):
        body = ndjson(
            {"message": {"content": "Hel"}},
            {"message": {"content": "lo"}},
            {"message": {"content": ""}, "done": True},
            {"message": {"content": "IGNORED"}},
        )
        calls: list = []
        monkeypatch.setattr(ollama_client._client, "stream", fake_stream(response(200, body), calls))
        assert list(ollama_client.chat_stream("s", [], "u")) == ["Hel", "lo"]
        assert calls[0]["json"]["options"]["num_ctx"] == 8192

    def test_http_error_raises(self, monkeypatch):
        resp = response(404, {"error": "model 'llama3' not found"})
        monkeypatch.setattr(ollama_client._client, "stream", fake_stream(resp))
        with pytest.raises(RuntimeError, match="not found"):
            list(ollama_client.chat_stream("s", [], "u"))

    def test_error_line_mid_stream_raises(self, monkeypatch):
        body = ndjson({"message": {"content": "partial"}}, {"error": "out of memory"})
        monkeypatch.setattr(ollama_client._client, "stream", fake_stream(response(200, body)))
        gen = ollama_client.chat_stream("s", [], "u")
        assert next(gen) == "partial"
        with pytest.raises(RuntimeError, match="out of memory"):
            next(gen)

    def test_connection_refused(self, monkeypatch):
        def boom(*a, **kw):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(ollama_client._client, "stream", boom)
        with pytest.raises(RuntimeError, match="not running"):
            list(ollama_client.chat_stream("s", [], "u"))


# ------------------------------------------------------ summarization (fix 3)


class TestSummarize:
    def test_short_document_is_sent_whole_in_one_call(self, monkeypatch):
        post = PostRecorder(chat_reply("SUMMARY"))
        monkeypatch.setattr(ollama_client._client, "post", post)
        doc = "Short document about cells. " * 10
        assert ollama_client.summarize(doc) == "SUMMARY"
        assert len(post.calls) == 1
        assert doc in post.calls[0]["json"]["messages"][-1]["content"]

    def test_long_document_is_map_reduced(self, monkeypatch):
        post = PostRecorder(chat_reply("dense notes for this section"))
        monkeypatch.setattr(ollama_client._client, "post", post)
        size = ollama_client.SUMMARY_SECTION_CHARS
        # Mark the very end so we can prove it is not silently dropped.
        doc = "a" * (size * 2) + "b" * 100 + "THE_END_MARKER"

        ollama_client.summarize(doc)

        section_calls = post.calls[:-1]
        final_call = post.calls[-1]
        assert len(section_calls) == 3
        assert "THE_END_MARKER" in section_calls[-1]["json"]["messages"][-1]["content"]
        for call in post.calls:
            assert len(call["json"]["messages"][-1]["content"]) < size + 1000
        assert "section by section" in final_call["json"]["messages"][-1]["content"]

    def test_notes_that_never_shrink_are_capped(self, monkeypatch):
        size = ollama_client.SUMMARY_SECTION_CHARS
        # Pathological model that answers every section with a huge reply.
        post = PostRecorder(chat_reply("n" * size))
        monkeypatch.setattr(ollama_client._client, "post", post)
        ollama_client.summarize("x" * (size * 3))
        final_prompt = post.calls[-1]["json"]["messages"][-1]["content"]
        assert len(final_prompt) < size + 1000


# ----------------------------------------------------------- JSON generators


@pytest.mark.parametrize(
    "fn", [ollama_client.generate_flashcards, ollama_client.extract_formulas]
)
class TestJsonGenerators:
    def test_parses_fenced_json(self, monkeypatch, fn):
        payload = '```json\n[{"id": 1, "question": "q", "answer": "a"}]\n```'
        monkeypatch.setattr(ollama_client._client, "post", PostRecorder(chat_reply(payload)))
        assert fn("doc") == [{"id": 1, "question": "q", "answer": "a"}]

    def test_invalid_json_returns_empty(self, monkeypatch, fn):
        monkeypatch.setattr(ollama_client._client, "post", PostRecorder(chat_reply("not json")))
        assert fn("doc") == []

    def test_uses_zero_temperature(self, monkeypatch, fn):
        post = PostRecorder(chat_reply("[]"))
        monkeypatch.setattr(ollama_client._client, "post", post)
        fn("doc")
        options = post.calls[0]["json"]["options"]
        assert options["temperature"] == 0.0
        assert options["num_ctx"] == 8192

    def test_input_is_truncated(self, monkeypatch, fn):
        post = PostRecorder(chat_reply("[]"))
        monkeypatch.setattr(ollama_client._client, "post", post)
        fn("z" * 50_000)
        assert post.calls[0]["json"]["messages"][-1]["content"].count("z") == 8000


def test_quiz_difficulty_in_prompt(monkeypatch):
    post = PostRecorder(chat_reply("[]"))
    monkeypatch.setattr(ollama_client._client, "post", post)
    assert ollama_client.generate_quiz("doc", "hard") == []
    assert "HARD" in post.calls[0]["json"]["messages"][0]["content"]
    assert post.calls[0]["json"]["options"]["temperature"] == 0.0


def test_free_text_keeps_default_temperature(monkeypatch):
    post = PostRecorder(chat_reply("text"))
    monkeypatch.setattr(ollama_client._client, "post", post)
    ollama_client.summarize("short doc")
    for call in post.calls:
        assert "temperature" not in call["json"]["options"]


# ------------------------------------------------- transport + speed settings


def test_default_url_avoids_localhost_ipv6_delay():
    # "localhost" resolves to ::1 first on Windows while Ollama listens on IPv4,
    # which added ~2 s to every request.
    assert "127.0.0.1" in ollama_client.OLLAMA_URL


def test_embed_many_sends_one_request_per_batch(monkeypatch):
    post = PostRecorder(
        lambda url, kw: response(200, {"embeddings": [[1.0, 0.0]] * len(kw["json"]["input"])})
    )
    monkeypatch.setattr(ollama_client._client, "post", post)
    matrix = ollama_client.embed_many(["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert len(post.calls) == 1
    assert post.calls[0]["json"]["input"] == ["a", "b", "c"]
    assert post.calls[0]["json"]["keep_alive"] == ollama_client.KEEP_ALIVE


def test_chat_keeps_model_loaded(monkeypatch):
    post = PostRecorder(chat_reply("hi"))
    monkeypatch.setattr(ollama_client._client, "post", post)
    ollama_client.chat("s", [], "u")
    assert post.calls[0]["json"]["keep_alive"] == ollama_client.KEEP_ALIVE


def test_tokenize_ignores_punctuation():
    assert ollama_client.tokenize("Where does it take place?") == [
        "where", "does", "it", "take", "place",
    ]


def test_bm25_matches_question_words_with_punctuation(monkeypatch):
    monkeypatch.setattr(ollama_client, "embed", fake_embed)
    chunks = ["the krebs cycle takes place in the mitochondria", "zzz qqq", "yyy www"]
    embs = [fake_embed("aaaa")] * 3  # identical, so only BM25 can separate them
    result = ollama_client.retrieve("Where does it take place?", chunks, embs)
    assert result[0]["index"] == 0


def test_retrieve_uses_prebuilt_bm25(monkeypatch):
    monkeypatch.setattr(ollama_client, "embed", fake_embed)
    chunks = ["alpha beta", "gamma delta", "epsilon zeta"]
    bm25 = ollama_client.build_bm25(chunks)
    calls = []
    real = bm25.get_scores
    bm25.get_scores = lambda q: calls.append(q) or real(q)
    monkeypatch.setattr(ollama_client, "build_bm25", lambda c: pytest.fail("rebuilt BM25"))
    ollama_client.retrieve("gamma", chunks, [fake_embed(c) for c in chunks], bm25=bm25)
    assert calls == [["gamma"]]


def test_replies_are_length_capped(monkeypatch):
    # A model that misses its stop token must not generate until the request
    # times out; every chat call carries a hard output limit.
    post = PostRecorder(chat_reply("hi"))
    monkeypatch.setattr(ollama_client._client, "post", post)
    ollama_client.chat("s", [], "u")
    ollama_client.chat("s", [], "u", temperature=0.0)
    for call in post.calls:
        assert call["json"]["options"]["num_predict"] == 2048
