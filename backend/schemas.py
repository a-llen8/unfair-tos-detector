"""
Pydantic request/response schemas for the Unfair ToS Clause Detector API.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

CATEGORIES = [
    "Limitation of liability", "Unilateral termination", "Unilateral change",
    "Content removal", "Contract by using", "Choice of law",
    "Jurisdiction", "Arbitration",
]


class ModelChoice(str, Enum):
    baseline = "baseline"
    distilbert = "distilbert"
    both = "both"


# ---------- Requests ----------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="A single clause or sentence to classify.")
    model: ModelChoice = Field(
        default=ModelChoice.distilbert,
        description="Which model to run. Defaults to distilbert per the Stage 4 evaluation verdict "
                    "(macro-F1 0.795 vs baseline's 0.761 on held-out test data).",
    )

    @field_validator("text")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty or whitespace-only")
        return v


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="Batch of clauses/sentences to classify (max 100 per request).")
    model: ModelChoice = Field(default=ModelChoice.distilbert)

    @field_validator("texts")
    @classmethod
    def check_nonempty_texts(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v]
        if any(not t for t in cleaned):
            raise ValueError("texts must not contain empty or whitespace-only strings")
        return cleaned


# ---------- Responses ----------

class SingleModelResult(BaseModel):
    predictions: Dict[str, bool] = Field(..., description="Per-category thresholded flag (category name -> True/False).")
    probabilities: Dict[str, float] = Field(..., description="Per-category raw probability, 0.0-1.0.")
    is_unfair: bool = Field(..., description="True if any category was flagged.")


class PredictResponse(BaseModel):
    text: str
    baseline: Optional[SingleModelResult] = None
    distilbert: Optional[SingleModelResult] = None


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    baseline_loaded: bool
    distilbert_loaded: bool
    categories: List[str] = CATEGORIES


class ErrorResponse(BaseModel):
    detail: str
