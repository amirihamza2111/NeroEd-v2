import numpy as np

from selection import formula_dense_text, representative_text


def topic_chunks(n_topics: int, per_topic: int, dim: int = 16):
    """Chunks whose embeddings form tight, well-separated topic clusters."""
    rng = np.random.default_rng(0)
    chunks, rows = [], []
    centres = np.eye(dim)[:n_topics]
    for i in range(n_topics * per_topic):
        topic = i % n_topics
        chunks.append(f"topic{topic} passage {i} " + "x" * 80)
        rows.append(centres[topic] + rng.normal(0, 0.01, dim))
    matrix = np.asarray(rows, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return chunks, matrix


class TestRepresentativeText:
    def test_short_document_is_unchanged(self):
        doc = "short document"
        assert representative_text(doc, [doc], np.ones((1, 4)), budget_chars=100) == doc

    def test_missing_embeddings_fall_back_to_prefix(self):
        doc = "a" * 500
        assert representative_text(doc, ["a" * 500], [], budget_chars=100) == "a" * 100

    def test_covers_every_topic_within_budget(self):
        chunks, matrix = topic_chunks(n_topics=6, per_topic=50)
        doc = "".join(chunks)
        text = representative_text(doc, chunks, matrix, budget_chars=1200)
        assert len(text) <= 1200
        # Taking the first 1,200 characters would only ever see the opening
        # chunks; clustering pulls passages from all six topics.
        assert {f"topic{t}" for t in range(6)} <= set(text.split())

    def test_keeps_document_order(self):
        chunks, matrix = topic_chunks(n_topics=4, per_topic=30)
        text = representative_text("".join(chunks), chunks, matrix, budget_chars=1000)
        positions = [int(p.split()[2]) for p in text.split("\n\n")]
        assert positions == sorted(positions)


class TestFormulaDenseText:
    def test_short_document_is_unchanged(self):
        assert formula_dense_text("F = m * a", ["F = m * a"], budget_chars=100) == "F = m * a"

    def test_prefers_chunks_with_maths(self):
        prose = ["The history of the city is long and varied. " * 3] * 40
        maths = ["Newton: F = m * a and KE = 1/2 * m * v^2 where v = d / t"]
        chunks = prose[:20] + maths + prose[20:]
        text = formula_dense_text("".join(chunks), chunks, budget_chars=200)
        assert "F = m * a" in text

    def test_no_maths_falls_back_to_prefix(self):
        chunks = ["plain words only here " * 5] * 30
        doc = "".join(chunks)
        assert formula_dense_text(doc, chunks, budget_chars=50) == doc[:50]
