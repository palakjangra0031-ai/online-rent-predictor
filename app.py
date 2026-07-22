"""
app.py
------
FastAPI server exposing two REST endpoints:

  POST /chat   — accepts a user message, returns a bot reply + metadata
  GET  /logs   — returns paginated interaction logs from the SQLite DB

Run with:
    uvicorn app:app --reload --port 8000

Auto-generated docs are available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import init_db, get_db, log_interaction, InteractionLog
from nlp_engine import NLPEngine

# ---------------------------------------------------------------------------
# Application lifespan — load heavy resources once at startup
# ---------------------------------------------------------------------------

nlp: NLPEngine  # module-level reference populated in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise DB tables and load the NLP model."""
    global nlp
    init_db()
    nlp = NLPEngine()
    print("[app] Startup complete — API is ready.")
    yield
    # Shutdown: nothing special needed for SQLite / HF pipeline
    print("[app] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Customer Support Bot",
    description="FAQ chatbot powered by DistilBERT with interaction logging.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS -------------------------------------------------------------------
# In production, replace "*" with your exact frontend origin, e.g.
# ["https://yourapp.com"] to prevent cross-site abuse.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend (index.html, etc.) from the /static directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, example="How do I reset my password?")


class ChatResponse(BaseModel):
    reply: str
    confidence: Optional[float] = None   # None when the fallback is triggered
    source: str                           # "faq_exact" | "faq_fuzzy" | "model" | "fallback"
    status: str                           # "answered" | "fallback"
    log_id: int                           # DB row ID for tracing


class LogEntry(BaseModel):
    id: int
    timestamp: datetime
    user_query: str
    bot_response: str
    confidence_score: Optional[float]
    status: str

    model_config = {"from_attributes": True}


class LogsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[LogEntry]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the chat UI at the root URL."""
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse, summary="Send a message to the bot")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Accepts a plain-text user message and returns the bot's reply.

    - Runs the NLP engine (FAQ lookup → DistilBERT → fallback).
    - Logs every interaction to SQLite automatically.
    - Returns the answer, confidence score, resolution source, and DB log ID.
    """
    bot_reply = nlp.answer(request.message)

    status = "fallback" if bot_reply.source == "fallback" else "answered"

    # Persist to DB regardless of whether the bot answered or escalated
    entry = log_interaction(
        db=db,
        user_query=request.message,
        bot_response=bot_reply.answer,
        confidence_score=bot_reply.confidence,
        status=status,
    )

    return ChatResponse(
        reply=bot_reply.answer,
        confidence=bot_reply.confidence,
        source=bot_reply.source,
        status=status,
        log_id=entry.id,
    )


@app.get("/logs", response_model=LogsResponse, summary="Retrieve interaction logs")
async def get_logs(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Rows per page"),
    status: Optional[str] = Query(None, description="Filter by status: 'answered' or 'fallback'"),
    db: Session = Depends(get_db),
):
    """
    Returns paginated interaction logs, optionally filtered by status.

    Useful for reviewing bot performance, detecting coverage gaps, and
    auditing all conversations for compliance.
    """
    query = db.query(InteractionLog).order_by(InteractionLog.timestamp.desc())

    if status:
        if status not in ("answered", "fallback"):
            raise HTTPException(status_code=400, detail="status must be 'answered' or 'fallback'")
        query = query.filter(InteractionLog.status == status)

    total = query.count()
    results = query.offset((page - 1) * page_size).limit(page_size).all()

    return LogsResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=results,
    )


@app.get("/health", summary="Health check")
async def health():
    """Simple liveness probe — returns 200 when the server is up."""
    return {"status": "ok"}
