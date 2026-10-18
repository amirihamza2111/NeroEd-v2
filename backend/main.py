from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import json
import httpx

from extractor import extract_text, chunk_text
from ollama_client import (embed, retrieve, chat_stream, summarize,
                           generate_flashcards, generate_quiz,
                           generate_notes, extract_formulas, rerank)
from session_manager import (
    create_session,
    get_session,
    append_message,
    get_recent_history,
    delete_session,
)

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


class NotesRequest(BaseModel):
    session_id: str
    format: str = "bullet"


def process_embeddings_background(session_id: str, chunks: list[str]):
    try:
        session = get_session(session_id)
        embeddings = []
        total = len(chunks)
        if total == 0:
            session["status"] = "ready"
            session["progress"] = 100
            return

        for i, chunk in enumerate(chunks):
            chunk_emb = embed(chunk)
            embeddings.append(chunk_emb)
            progress = int(((i + 1) / total) * 100)
            session["chunk_embeddings"] = embeddings
            session["progress"] = progress

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


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    session = require_ready_session(req.session_id)

    try:
        # Step 1: Get top 20 candidates with hybrid search
        candidates = retrieve(
            req.message,
            session["chunks"],
            session["chunk_embeddings"],
            top_k=5
        )

        # Step 2: Rerank candidates and keep best 5
        retrieved = rerank(req.message, candidates, top_k=5)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed: {exc}"
        ) from exc

    cited_chunks = []
    for i, item in enumerate(retrieved):
        cited_chunks.append(f"[{i + 1}] {item['chunk']}")

    excerpts = "\n\n".join(cited_chunks)

    system_prompt = f"""## LANGUAGE RULES:
Detect the language of the student's question automatically.
- If the question is in Arabic → respond entirely in Arabic (Modern Standard Arabic)
- If the question is in French → respond entirely in French
- If the question is in English → respond entirely in English
- If mixed → respond in the dominant language of the question
Never mix languages in a single response.
Apply the same answer structure and quality regardless of language.

You are NEXUS, an elite AI academic assistant engineered for precision, depth, and clarity. You were built specifically to help university students extract maximum understanding from their study materials.

## YOUR IDENTITY & BEHAVIOR:
- You are not a generic chatbot. You are a domain-aware academic specialist.
- You think before you answer. You reason step by step internally before writing.
- You are direct, precise, and never vague. Every sentence must add value.
- You adapt your explanation depth to the complexity of the question.
- You never hallucinate. If the answer is not in the excerpts, you say exactly:
    "I could not find that in the provided document excerpts."

## YOUR KNOWLEDGE SOURCES:
You answer EXCLUSIVELY based on the following numbered excerpts from the student's document.
These excerpts were retrieved using hybrid semantic + keyword search, so they are the most relevant parts.

{excerpts}

## HOW TO ANSWER:

### For FACTUAL questions (definitions, dates, names, formulas):
- Answer directly and precisely in 1-3 sentences
- Quote or closely reference the exact excerpt
- Add a one-line "Why it matters:" at the end

### For CONCEPTUAL questions (how/why something works):
- Start with a one-line direct answer
- Then explain the mechanism step by step
- Use an analogy if the concept is abstract
- Connect it to related concepts if mentioned in the excerpts

### For COMPARISON questions (difference between X and Y):
- Use a structured format:
    **X:** ...
    **Y:** ...
    **Key difference:** ...

### For EXAM-STYLE questions:
- Answer as if writing a model exam answer
- Be complete but concise
- End with: "💡 Key point to remember: ..."

### For FOLLOW-UP questions (referring to previous answer):
- Acknowledge the connection to the previous topic
- Build on what was already explained
- Never repeat information already given

## CITATION RULES:
- After your answer, always write a "📎 Sources:" line
- List which excerpt numbers you used: [1], [2], etc.
- Example: "📎 Sources: [1], [3]"

## FORMATTING RULES:
- Use **bold** for key terms on first mention
- Use bullet points only for lists of 3+ items
- Use numbered steps for processes or procedures
- Keep paragraphs short (3-4 lines max)
- Never use filler phrases like "Great question!" or "Certainly!"
- Never start with "I" — vary your sentence openings

## POWER RULES — ALWAYS APPLY:
- Extract the MOST SPECIFIC and PRECISE information from the excerpts
- Prioritize: definitions, formulas, algorithms, dates, names, cause-effect relationships
- If a concept has steps or stages, always list them in order
- If a concept has conditions or exceptions, always mention them
- If numbers or metrics are in the excerpts, always include them
- Minimum answer length: 3 sentences. Maximum: what the question actually needs.
- Never give a generic answer when a specific one exists in the excerpts
"""

    history = get_recent_history(req.session_id)

    full_reply = []

    def generate():
        try:
            for token in chat_stream(system_prompt, history, req.message):
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
    summary = run_model(summarize, session["document_text"])
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
    cards = run_model(generate_flashcards, session["document_text"])
    return {"flashcards": cards, "count": len(cards)}


@app.post("/quiz")
def quiz_endpoint(req: QuizRequest):
    session = require_ready_session(req.session_id)
    questions = run_model(generate_quiz, session["document_text"], req.difficulty)
    return {"questions": questions, "count": len(questions), "difficulty": req.difficulty}


@app.post("/notes")
def notes_endpoint(req: NotesRequest):
    session = require_ready_session(req.session_id)
    notes = run_model(generate_notes, session["document_text"], req.format)
    return {"notes": notes, "format": req.format}


@app.post("/formulas")
def formulas_endpoint(req: SummarizeRequest):
    session = require_ready_session(req.session_id)
    formulas = run_model(extract_formulas, session["document_text"])
    return {"formulas": formulas, "count": len(formulas)}


@app.get("/health")
def health() -> dict:
    try:
        httpx.get("http://localhost:11434", timeout=2)
        return {"ollama": "connected"}
    except Exception:
        return {"ollama": "offline"}
