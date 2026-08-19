"""
FastAPI backend for the Unfair ToS Clause Detector.

Run with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health          -- model load status
    POST /predict         -- classify a single clause/sentence
    POST /predict/batch   -- classify up to 100 at once
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import engine
from backend.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models once at startup, not on the first request --
    # avoids a slow/unpredictable first API call and lets /health report
    # accurate status immediately.
    engine.load_all()
    yield
    # (no explicit teardown needed -- models are released when the process exits)


app = FastAPI(
    title="Unfair ToS Clause Detector",
    description="Multi-label classifier flagging potentially unfair clauses in Terms of Service text.",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS for local dev against the Stage 6 frontend (plain HTML/JS,
# likely served from a different port/file:// origin). Tighten this before
# any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if (engine.baseline_loaded and engine.distilbert_loaded) else "degraded",
        baseline_loaded=engine.baseline_loaded,
        distilbert_loaded=engine.distilbert_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = engine.predict([req.text], model=req.model.value)[0]
    except RuntimeError as e:
        # model not loaded -- surface as 503, not a generic 500
        raise HTTPException(status_code=503, detail=str(e))
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    try:
        results = engine.predict(req.texts, model=req.model.value)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return BatchPredictResponse(results=[PredictResponse(**r) for r in results])

from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )