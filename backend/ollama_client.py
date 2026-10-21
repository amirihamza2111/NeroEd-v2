import json
import os
import re

import httpx
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 127.0.0.1, not "localhost": on Windows, localhost resolves to IPv6 ::1 first
# while Ollama listens on IPv4 only, and every request then waits ~2 s for the
# IPv6 attempt to fail before falling back.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("NEROED_CHAT_MODEL", "llama3")
EMBED_MODEL = os.getenv("NEROED_EMBED_MODEL", "nomic-embed-text")
RERANKER_MODEL = "BAAI/bge-reranker-base"
# The first request after Ollama starts has to load the model into memory,
# which can take far longer than httpx's 5 second default.
EMBED_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
# Chunks per /api/embed request. One request per chunk is dominated by HTTP
# overhead; batching amortises it and lets the GPU embed chunks together.
EMBED_BATCH_SIZE = 64
# Keep models loaded between questions instead of Ollama's 5 minute default,
# so a pause while studying does not cost a multi-second model reload.
KEEP_ALIVE = "30m"
# Ollama defaults to a small context window and silently drops whatever does
# not fit. llama3 supports 8k tokens, so ask for all of it. num_predict caps
# a reply at 2,048 tokens (about twice the longest summary), so a model that
# misses its stop token ends with a cut-off answer instead of a timeout.
CHAT_OPTIONS = {"num_ctx": 8192, "num_predict": 2048}
# ~4k tokens per section leaves room for the system prompt and the reply.
SUMMARY_SECTION_CHARS = 16000
MAX_CONDENSE_ROUNDS = 3
JSON_TEMPERATURE = 0.0
SECTION_NOTES_PROMPT = """You are taking study notes on one section of a longer document.
Write dense, specific notes covering every concept, definition, formula, date,
number, and argument in this section. Keep technical terms exactly as written.
Do not add an introduction or conclusion. Output notes only."""
_reranker = None
# One shared client reuses TCP connections; building a new client per request
# costs ~0.3 s on Windows.
_client = httpx.Client(base_url=OLLAMA_URL, timeout=httpx.Timeout(300.0, connect=5.0))

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens for BM25, so "place?" matches "place"."""
    return _TOKEN_RE.findall(text.lower())


def _extract_ollama_error(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return f"Ollama error: {response.status_code}"

    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])

    return f"Ollama error: {response.status_code}"


def embed_many(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts in one Ollama request; returns a float32 matrix."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    payload = {"model": EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE}

    try:
        response = _client.post("/api/embed", json=payload, timeout=EMBED_TIMEOUT)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"Ollama timed out while embedding with '{EMBED_MODEL}'"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(_extract_ollama_error(exc.response)) from exc

    return np.asarray(response.json()["embeddings"], dtype=np.float32)


def embed(text: str) -> list[float]:
    """Generate an embedding for a single text."""
    return embed_many([text])[0].tolist()


def normalize_rows(matrix) -> np.ndarray:
    """L2-normalise each row so a dot product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_bm25(chunks: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(c) for c in chunks])


def _minmax(scores: np.ndarray) -> np.ndarray:
    mn, mx = float(scores.min()), float(scores.max())
    if mx - mn < 1e-9:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - mn) / (mx - mn)


def retrieve(
    query: str,
    chunks: list[str],
    chunk_embeddings,
    top_k: int = 5,
    bm25: BM25Okapi | None = None,
) -> list[dict]:
    """
    Hybrid search: combines BM25 keyword score + cosine similarity score.
    Returns list of dicts: { "chunk": str, "score": float, "index": int }

    `chunk_embeddings` is ideally already row-normalised (normalize_rows) and
    `bm25` built once per document; both are computed here when missing.
    """
    if not chunks:
        return []

    matrix = normalize_rows(chunk_embeddings)
    query_vec = normalize_rows(embed(query))[0]
    semantic = matrix @ query_vec

    if bm25 is None:
        bm25 = build_bm25(chunks)
    keyword = np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float32)

    combined = 0.6 * _minmax(semantic) + 0.4 * _minmax(keyword)
    n_candidates = min(len(chunks), max(top_k * 4, 20))
    top = np.argpartition(-combined, n_candidates - 1)[:n_candidates]
    top = top[np.argsort(-combined[top])]
    return [
        {"chunk": chunks[i], "score": float(combined[i]), "index": int(i)}
        for i in top
    ]


def get_reranker() -> CrossEncoder:
    """Lazy-load the reranker model on first use."""
    global _reranker
    if _reranker is None:
        import torch

        if torch.cuda.is_available():
            # fp16 on the GPU: ~35 ms per question instead of ~1.1 s on the
            # CPU, in ~0.5 GB of VRAM next to the Ollama models.
            _reranker = CrossEncoder(
                RERANKER_MODEL,
                device="cuda",
                model_kwargs={"torch_dtype": torch.float16},
            )
        else:
            _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-rank candidate chunks using bge-reranker-base cross-encoder.

    Input: list of dicts with keys: chunk, score, index
    Output: top_k dicts re-sorted by reranker score, with reranker_score added
    """
    if not candidates:
        return candidates

    reranker = get_reranker()

    # Build pairs: [query, chunk_text] for each candidate
    pairs = [[query, item["chunk"]] for item in candidates]

    # Score all pairs at once
    reranker_scores = reranker.predict(pairs)

    # Attach reranker score to each candidate
    for i, item in enumerate(candidates):
        item["reranker_score"] = float(reranker_scores[i])

    # Sort by reranker score descending
    candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

    return candidates[:top_k]


def chat_stream(system_prompt: str, history: list[dict], user_message: str):
    """Generator that yields text tokens as they arrive from Ollama."""
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]
    try:
        with _client.stream(
            "POST",
            "/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": True,
                "options": CHAT_OPTIONS,
                "keep_alive": KEEP_ALIVE,
            },
            timeout=120,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise RuntimeError(_extract_ollama_error(response))
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from exc


def chat(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    temperature: float | None = None,
) -> str:
    """Send a chat request to Ollama and return the response content."""
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": CHAT_OPTIONS
        if temperature is None
        else {**CHAT_OPTIONS, "temperature": temperature},
    }

    try:
        response = _client.post("/api/chat", json=payload, timeout=300.0)
        response.raise_for_status()
    except httpx.RequestError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(_extract_ollama_error(exc.response)) from exc

    data = response.json()
    return data["message"]["content"]


def summarize(document_text: str) -> str:
    system_prompt = """You are SCHOLAR, an elite academic summarization engine trained on millions of textbooks, research papers, and study guides. You think step by step before writing, and your summaries are used by top university students worldwide.

## YOUR THINKING PROCESS (internal, before writing):
1. First identify: What TYPE of document is this?
2. Then ask: What is the CORE MESSAGE the author wants to convey?
3. Then find: What are the BUILDING BLOCKS of knowledge in this text?
4. Then check: What would a student MOST LIKELY be tested on?
5. Now write the summary.

## OUTPUT FORMAT (always follow exactly):
---
### 🧠 DOCUMENT TYPE & SUBJECT
One line: what kind of document this is and what field/topic it covers.
---
### 📌 CORE MESSAGE
The single most important idea in 2-3 sentences.
---
### 🏗️ CONCEPT BREAKDOWN
For each major concept (4-8 concepts):
**[Concept Name]** — Clear explanation in 1-2 sentences. Why it matters.
---
### 📊 DETAILED ANALYSIS
Organized by the document's own structure. For each section:
- What it covers
- Key arguments or information
- Any formulas, definitions, dates, or data
---
### ⚠️ CRITICAL DETAILS (Don't Miss These)
Bullet list of specific facts, numbers, definitions, exceptions.
---
### 🔄 THE BIG PICTURE
How all ideas connect. (3-5 sentences)
---
### 🎯 EXAM-READY TAKEAWAYS
Exactly 5 bullet points. Each one a complete standalone fact.
---
### ❓ LIKELY EXAM QUESTIONS
3 questions with brief model answers.
---
## RULES:
- Never say "the document says" — state facts directly
- Never be vague — name concepts, explain them, connect them
- Use analogies for complex ideas
- Preserve technical terms but explain on first use"""

    condensed = _condense_for_summary(document_text)
    if condensed is document_text:
        intro = "Analyze and summarize the following document. "
    else:
        intro = (
            "The document was too long to read in one pass, so below are "
            "detailed notes taken section by section, in order. Treat them "
            "as the full document. "
        )

    user_message = (
        intro
        + "Think carefully about what matters most before writing. "
        "A student's exam grade depends on your summary:\n\n"
        f"{condensed}"
    )
    return chat(system_prompt=system_prompt, history=[], user_message=user_message)


def _split_sections(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _condense_for_summary(document_text: str) -> str:
    """
    Map-reduce long documents so the final summary prompt fits in the
    model's context window. Short documents are returned unchanged.
    """
    text = document_text
    for _ in range(MAX_CONDENSE_ROUNDS):
        if len(text) <= SUMMARY_SECTION_CHARS:
            return text
        sections = _split_sections(text, SUMMARY_SECTION_CHARS)
        notes = []
        for n, section in enumerate(sections, start=1):
            note = chat(
                system_prompt=SECTION_NOTES_PROMPT,
                history=[],
                user_message=f"Section {n} of {len(sections)}:\n\n{section}",
            )
            notes.append(f"## Section {n}\n{note.strip()}")
        text = "\n\n".join(notes)
    # Notes refused to shrink; fall back to what fits rather than overflow.
    return text[:SUMMARY_SECTION_CHARS]


def generate_flashcards(document_text: str) -> list[dict]:
    system_prompt = """You are an expert educator and flashcard designer.
Your job is to extract the most important knowledge from a document and turn it into flashcards.

RULES:
- Generate between 10 and 20 flashcards depending on document length
- Each flashcard must be specific, not generic
- Prioritize: definitions, formulas, algorithms, key concepts, cause-effect pairs
- Never create vague cards like "What is important?"
- For formulas: write them clearly with variable explanations

RESPOND ONLY WITH VALID JSON. No explanation, no markdown, no preamble.
Format exactly:
[
  {
    "id": 1,
    "question": "What is X?",
    "answer": "X is...",
    "type": "definition",
    "difficulty": "easy"
  }
]
Types allowed: definition, formula, concept, process, fact
Difficulty allowed: easy, medium, hard"""

    user_message = f"Generate flashcards from this document:\n\n{document_text[:8000]}"
    # Structured extraction should copy from the document, not improvise.
    raw = chat(
        system_prompt=system_prompt,
        history=[],
        user_message=user_message,
        temperature=JSON_TEMPERATURE,
    )
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except (json.JSONDecodeError, TypeError):
        return []


def generate_quiz(document_text: str, difficulty: str = "medium") -> list[dict]:
    system_prompt = f"""You are an expert exam writer for university students.
Generate a quiz at difficulty level: {difficulty.upper()}

DIFFICULTY GUIDE:
- easy: recall facts, simple definitions
- medium: understanding, application of concepts
- exam-level: analysis, multi-step reasoning, edge cases

RULES:
- Generate exactly 10 questions
- Mix question types: 6 MCQ, 2 true/false, 2 short answer
- MCQ must have exactly 4 options with one correct answer
- Short answer questions need a model answer
- Questions must be specific to the document content

RESPOND ONLY WITH VALID JSON. No explanation, no markdown, no preamble.
Format exactly:
[
  {{
    "id": 1,
    "type": "mcq",
    "question": "...",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
    "correct": "A",
    "explanation": "Why this answer is correct..."
  }},
  {{
    "id": 2,
    "type": "truefalse",
    "question": "...",
    "correct": true,
    "explanation": "..."
  }},
  {{
    "id": 3,
    "type": "shortanswer",
    "question": "...",
    "model_answer": "..."
  }}
]"""

    user_message = f"Generate a quiz from this document:\n\n{document_text[:8000]}"
    # Structured extraction should copy from the document, not improvise.
    raw = chat(
        system_prompt=system_prompt,
        history=[],
        user_message=user_message,
        temperature=JSON_TEMPERATURE,
    )
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except (json.JSONDecodeError, TypeError):
        return []


def extract_formulas(document_text: str) -> list[dict]:
    system_prompt = """You are a scientific formula extractor.
Extract ALL mathematical, physical, chemical, or logical formulas from the document.

RESPOND ONLY WITH VALID JSON. No explanation, no markdown, no preamble.
Format exactly:
[
  {
    "id": 1,
    "formula": "E = mc²",
    "latex": "E = mc^2",
    "name": "Mass-energy equivalence",
    "variables": {
      "E": "Energy in joules",
      "m": "Mass in kilograms",
      "c": "Speed of light (3×10⁸ m/s)"
    },
    "domain": "Physics"
  }
]
If no formulas exist, return an empty array: []"""

    user_message = f"Extract all formulas from this document:\n\n{document_text[:8000]}"
    # Structured extraction should copy from the document, not improvise.
    raw = chat(
        system_prompt=system_prompt,
        history=[],
        user_message=user_message,
        temperature=JSON_TEMPERATURE,
    )
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except (json.JSONDecodeError, TypeError):
        return []
