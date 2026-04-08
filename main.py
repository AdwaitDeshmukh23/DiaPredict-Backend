"""
DiaPredict-AI — FastAPI Main Application
"""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import PredictionRequest, PredictionResponse, ChatRequest, ChatResponse
from scoring import calculate_diabetes_risk
from chatbot import get_chat_response

load_dotenv()

app = FastAPI(
    title="DiaPredict-AI API",
    description="Diabetes Risk Prediction and AI Health Assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Rate Limiting (simple in-memory) ────────────────────────────────────────
from collections import defaultdict
import time

request_counts = defaultdict(list)
RATE_LIMIT = 30  # requests per minute


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    if len(request_counts[ip]) >= RATE_LIMIT:
        return True
    request_counts[ip].append(now)
    return False


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if is_rate_limited(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a moment."}
        )
    return await call_next(request)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "DiaPredict-AI API is running 🩺"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_diabetes_risk(data: PredictionRequest):
    """
    Calculate diabetes risk score based on health parameters.
    Returns risk percentage, category, color, and personalized suggestions.
    """
    try:
        result = calculate_diabetes_risk(data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring engine error: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_assistant(request: ChatRequest):
    """
    AI health assistant powered by Anthropic Claude.
    Falls back to rule-based responses if API unavailable.
    """
    try:
        response = await get_chat_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

