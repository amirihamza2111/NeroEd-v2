"""
Pick which parts of a large document to send to the model.

The model's context window holds a few thousand words, so for big documents
the study tools cannot read everything. Instead of only the first pages, these
helpers choose passages spread across the whole document:

- representative_text: k-means clusters the chunk embeddings into topics and
  keeps the chunk closest to each cluster centre, so every topic contributes.
- formula_dense_text: ranks chunks by how much maths they contain.

Both return the chosen chunks in their original order so the text still reads
in sequence.
"""

import re

import numpy as np

_MATH_RE = re.compile(r"[=+\-*/^√∑∫≈≤≥±×÷]|\b(?:sin|cos|tan|log|ln|exp|sqrt)\b")


def _join(chunks: list[str], indices) -> str:
    return "\n\n".join(chunks[i] for i in sorted(indices))


def _kmeans_representatives(matrix: np.ndarray, k: int, seed: int = 0, iters: int = 15) -> list[int]:
    """Indices of the rows closest to each of k cosine k-means centroids."""
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    centroids = matrix[rng.choice(n, size=k, replace=False)]
    for _ in range(iters):
        labels = np.argmax(matrix @ centroids.T, axis=1)
        for c in range(k):
            members = matrix[labels == c]
            if len(members):
                centre = members.mean(axis=0)
                norm = np.linalg.norm(centre)
                centroids[c] = centre / norm if norm else centre
    sims = matrix @ centroids.T
    labels = np.argmax(sims, axis=1)
    picks = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if len(members):
            picks.append(int(members[np.argmax(sims[members, c])]))
    return picks


def representative_text(document_text: str, chunks: list[str], matrix, budget_chars: int) -> str:
    """
    Text of at most ~budget_chars that covers the whole document.

    Short documents are returned unchanged. `matrix` holds one row-normalised
    embedding per chunk.
    """
    if len(document_text) <= budget_chars or not chunks or matrix is None or len(matrix) != len(chunks):
        return document_text[:budget_chars]

    avg_len = max(1, sum(len(c) for c in chunks) // len(chunks))
    k = max(1, min(len(chunks), budget_chars // avg_len))
    picks = _kmeans_representatives(np.asarray(matrix, dtype=np.float32), k)
    text = _join(chunks, picks)
    return text[:budget_chars]


def formula_dense_text(document_text: str, chunks: list[str], budget_chars: int) -> str:
    """The chunks with the most mathematical notation, up to budget_chars."""
    if len(document_text) <= budget_chars or not chunks:
        return document_text[:budget_chars]

    density = [len(_MATH_RE.findall(c)) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: density[i], reverse=True)
    picks, used = [], 0
    for i in order:
        if density[i] == 0 or used + len(chunks[i]) > budget_chars:
            break
        picks.append(i)
        used += len(chunks[i]) + 2
    if not picks:
        return document_text[:budget_chars]
    return _join(chunks, picks)
