"""
Shared inference layer for the Unfair ToS Clause Detector backend.

Loads both models ONCE at process startup (via get_engine(), typically
called from a FastAPI lifespan/startup hook) and exposes a single
predict() entrypoint that main.py's routes call. Keeping model-loading
out of main.py keeps the FastAPI layer thin and makes this testable
without spinning up a server.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

# Make src/ importable regardless of where uvicorn is launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models import transformer as transformer_module  # noqa: E402

CATEGORIES = transformer_module.CATEGORIES

BASELINE_MODEL_PATH = REPO_ROOT / "models" / "baseline_logreg.joblib"
THRESHOLDS_PATH = REPO_ROOT / "models" / "thresholds.json"


class InferenceEngine:
    """Holds both loaded models in memory. Instantiate once per process."""

    def __init__(self):
        self.baseline_pipeline = None
        self.baseline_thresholds: Optional[np.ndarray] = None
        self.transformer_bundle: Optional[transformer_module.TransformerBundle] = None
        self._baseline_error: Optional[str] = None
        self._distilbert_error: Optional[str] = None

    # ---------- Loading ----------

    def load_baseline(self):
        try:
            self.baseline_pipeline = joblib.load(BASELINE_MODEL_PATH)
            with open(THRESHOLDS_PATH) as f:
                data = json.load(f)
            thresh_map = data["baseline_thresholds"]
            self.baseline_thresholds = np.array(
                [thresh_map[c] for c in CATEGORIES], dtype=np.float32
            )
        except Exception as e:  # noqa: BLE001 - surfaced via /health, not swallowed
            self._baseline_error = f"{type(e).__name__}: {e}"

    def load_distilbert(self):
        try:
            self.transformer_bundle = transformer_module.load_model()
        except Exception as e:  # noqa: BLE001
            self._distilbert_error = f"{type(e).__name__}: {e}"

    def load_all(self):
        self.load_baseline()
        self.load_distilbert()

    @property
    def baseline_loaded(self) -> bool:
        return self.baseline_pipeline is not None

    @property
    def distilbert_loaded(self) -> bool:
        return self.transformer_bundle is not None

    # ---------- Prediction ----------

    def predict_baseline(self, texts: List[str]) -> List[dict]:
        if not self.baseline_loaded:
            raise RuntimeError(
                f"Baseline model is not loaded (load error: {self._baseline_error})"
            )
        probs = self.baseline_pipeline.predict_proba(texts)  # (n, 8)
        preds = (probs >= self.baseline_thresholds[None, :]).astype(int)
        return self._format_results(texts, probs, preds)

    def predict_distilbert(self, texts: List[str]) -> List[dict]:
        if not self.distilbert_loaded:
            raise RuntimeError(
                f"DistilBERT model is not loaded (load error: {self._distilbert_error})"
            )
        out = transformer_module.predict(self.transformer_bundle, texts)
        # transformer_module.predict already returns the per-text dict shape;
        # re-key to match what main.py expects (see _format_results below).
        return [
            {
                "predictions": r["predictions"],
                "probabilities": r["probabilities"],
                "is_unfair": r["is_unfair"],
            }
            for r in out["results"]
        ]

    @staticmethod
    def _format_results(texts: List[str], probs: np.ndarray, preds: np.ndarray) -> List[dict]:
        results = []
        for i in range(len(texts)):
            results.append(
                {
                    "predictions": {c: bool(preds[i, j]) for j, c in enumerate(CATEGORIES)},
                    "probabilities": {c: float(probs[i, j]) for j, c in enumerate(CATEGORIES)},
                    "is_unfair": bool(preds[i].any()),
                }
            )
        return results

    def predict(self, texts: List[str], model: str) -> List[dict]:
        """model: 'baseline' | 'distilbert' | 'both'. Returns a list (one
        entry per text) of dicts with 'baseline' and/or 'distilbert' keys,
        ready to slot into PredictResponse/BatchPredictResponse."""
        baseline_results = self.predict_baseline(texts) if model in ("baseline", "both") else None
        distilbert_results = self.predict_distilbert(texts) if model in ("distilbert", "both") else None

        merged = []
        for i, text in enumerate(texts):
            entry = {"text": text}
            if baseline_results is not None:
                entry["baseline"] = baseline_results[i]
            if distilbert_results is not None:
                entry["distilbert"] = distilbert_results[i]
            merged.append(entry)
        return merged


# Module-level singleton -- main.py imports this directly rather than
# constructing its own engine, so there's exactly one copy of each model
# in memory regardless of how many routes/workers touch it.
engine = InferenceEngine()


if __name__ == "__main__":
    # python -m backend.inference
    engine.load_all()
    print("baseline_loaded:", engine.baseline_loaded)
    print("distilbert_loaded:", engine.distilbert_loaded)
    sample = [
        "We may terminate your account at any time, for any reason, without notice.",
        "You will receive a full refund within 30 days of purchase.",
    ]
    out = engine.predict(sample, model="both")
    print(json.dumps(out, indent=2))
