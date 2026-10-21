import json

import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from extractor import chunk_text, extract_text
from ollama_client import (
    EMBED_BATCH_SIZE,
    OLLAMA_URL,
    build_bm25,
    chat_stream,
    embed_many,
    extract_formulas,
    generate_flashcards,
    generate_quiz,
    normalize_rows,
    rerank,
    retrieve,
    summarize,
)
from selection import formula_dense_text, representative_text
from session_manager import (
    append_message,
    create_session,
    delete_session,
    get_recent_history,
    get_session,
)

MAX_UPLOAD_MB = 100
# How much text each study tool sends to the model. Larger documents are
# condensed to representative passages from across the whole file.
SUMMARY_SOURCE_CHARS = 64_000
STUDY_TOOL_SOURCE_CHARS = 8_000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://localhost:3005",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
        "http://127.0.0.1:3005",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[dict]


class SummarizeRequest(BaseModel):
    session_id: str


class QuizRequest(BaseModel):
    session_id: str
    difficulty: str = "medium"


def process_embeddings_background(session_id: str, chunks: list[str]):
    try:
        session = get_session(session_id)
        total = len(chunks)
        if total == 0:
            session["status"] = "ready"
            session["progress"] = 100
            return

        batches = []
        for start in range(0, total, EMBED_BATCH_SIZE):
            batches.append(embed_many(chunks[start : start + EMBED_BATCH_SIZE]))
            session["progress"] = int(min(start + EMBED_BATCH_SIZE, total) / total * 99)

        # Build the search indexes once here instead of on every question.
        session["chunk_embeddings"] = normalize_rows(np.vstack(batches))
        session["bm25"] = build_bm25(chunks)
        session["progress"] = 100
        session["status"] = "ready"
    except Exception as exc:
        try:
            session = get_session(session_id)
            session["status"] = "error"
            session["error_detail"] = str(exc)
        except KeyError:
            pass

def require_ready_session(session_id: str) -> dict:
    """Return the session, or raise 404 if unknown / 400 if still indexing."""
    try:
        session = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found") from None
    if session.get("status") != "ready":
        raise HTTPException(status_code=400, detail="Document still indexing")
    return session


def study_source(session: dict, budget: int) -> str:
    return representative_text(
        session["document_text"], session["chunks"], session["chunk_embeddings"], budget
    )


def run_model(fn, *args):
    """Call a model helper, mapping Ollama failures to 502 and bugs to 500."""
    try:
        return fn(*args)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB"
        )

    try:
        # PDF/DOCX parsing is CPU-bound; keep it off the event loop.
        document_text = await run_in_threadpool(
            extract_text, file_bytes, file.filename or ""
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = chunk_text(document_text)

    session_id = create_session(document_text, chunks)
    background_tasks.add_task(process_embeddings_background, session_id, chunks)

    return JSONResponse(
        status_code=202,
        content={"session_id": session_id, "chunk_count": len(chunks)}
    )

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    try:
        session = get_session(session_id)
        return {
            "status": session.get("status", "embedding"),
            "progress": session.get("progress", 0),
            "error_detail": session.get("error_detail", None)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found") from None


# The rules stay fixed and the excerpts travel in the user message. Compared
# with the previous 1,000-token prompt that embedded the excerpts, this was as
# accurate or better in an A/B test on llama3 (24/24 vs 22/24), cited fewer
# non-existent excerpts, and used about half the prompt tokens.
CHAT_SYSTEM_PROMPT = """You are NeroEd, a study assistant for university students.

Language: write your whole answer in the language of the student's question (English, French or Arabic), even when the excerpts are in another language.

Answer only from the numbered document excerpts in the student's message:
- Give the answer in the first sentence, then only the explanation the question needs.
- Copy exact numbers, names, dates and formulas from the excerpts.
- Use **bold** for key terms and numbered steps for processes.
- If the excerpts do not contain the answer, reply exactly: "I could not find that in the provided document excerpts."

End with one line "Sources: [n]" naming only the excerpts you used. The message says how many excerpts there are; never cite a number above that."""


def build_chat_user_message(excerpts: list[str], question: str) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(excerpts))
    # Small models follow the language rule more reliably when it sits right
    # after the question than when it is only in the system prompt.
    return (
        f"Document excerpts (numbered 1 to {len(excerpts)}):\n{numbered}\n\n"
        f"Question: {question}\n\n"
        "(Write the answer in the same language as the question above.)"
    )


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    session = require_ready_session(req.session_id)

    try:
        # Step 1: Get top 20 candidates with hybrid search
        candidates = retrieve(
            req.message,
            session["chunks"],
            session["chunk_embeddings"],
            top_k=5,
            bm25=session.get("bm25"),
        )

        # Step 2: Rerank candidates and keep best 5
        retrieved = rerank(req.message, candidates, top_k=5)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed: {exc}"
        ) from exc

    user_message = build_chat_user_message([item["chunk"] for item in retrieved], req.message)
    # History keeps the bare questions, so earlier excerpts don't pile up in
    # the context window during a long conversation.
    history = get_recent_history(req.session_id)

    full_reply = []

    def generate():
        try:
            for token in chat_stream(CHAT_SYSTEM_PROMPT, history, user_message):
                full_reply.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        complete_reply = "".join(full_reply)
        # Record the turn only once it succeeded, so a failed reply never
        # leaves an unanswered user message in the history.
        append_message(req.session_id, "user", req.message)
        append_message(req.session_id, "assistant", complete_reply)

        citations = [
            {
                "index": i + 1,
                "chunk": item["chunk"],
                "score": round(item["score"], 3),
            }
            for i, item in enumerate(retrieved)
        ]
        yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/summarize")
def summarize_document(request: SummarizeRequest) -> dict:
    session = require_ready_session(request.session_id)
    summary = run_model(summarize, study_source(session, SUMMARY_SOURCE_CHARS))
    return {"summary": summary}


@app.delete("/session/{session_id}")
async def remove_session(session_id: str) -> dict:
    try:
        get_session(session_id)
        delete_session(session_id)
        return {"deleted": True}
    except KeyError:
        return {"deleted": False, "detail": "Session not found"}
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to delete session"
        ) from exc


@app.post("/flashcards")
def flashcards_endpoint(req: SummarizeRequest):
    session = require_ready_session(req.session_id)
    cards = run_model(generate_flashcards, study_source(session, STUDY_TOOL_SOURCE_CHARS))
    return {"flashcards": cards, "count": len(cards)}


@app.post("/quiz")
def quiz_endpoint(req: QuizRequest):
    session = require_ready_session(req.session_id)
    questions = run_model(
        generate_quiz, study_source(session, STUDY_TOOL_SOURCE_CHARS), req.difficulty
    )
    return {"questions": questions, "count": len(questions), "difficulty": req.difficulty}


@app.post("/formulas")
def formulas_endpoint(req: SummarizeRequest):
    session = require_ready_session(req.session_id)
    source = formula_dense_text(
        session["document_text"], session["chunks"], STUDY_TOOL_SOURCE_CHARS
    )
    formulas = run_model(extract_formulas, source)
    return {"formulas": formulas, "count": len(formulas)}


@app.get("/health")
def health() -> dict:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2)
        return {"ollama": "connected"}
    except Exception:
        return {"ollama": "offline"}
