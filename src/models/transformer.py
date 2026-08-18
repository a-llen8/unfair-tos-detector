"""
DistilBERT inference wrapper for the Unfair ToS Clause Detector.

Mirrors the interface of src/models/baseline.py so the backend can treat
both models interchangeably:
    load_model()          -> loaded model bundle
    predict_proba_matrix() -> (n_samples, n_categories) probability array
    predict()              -> thresholded predictions + probabilities

Thresholds are loaded from models/thresholds.json (produced in Stage 4 by
sweeping per-category F1 on the validation set) rather than a flat 0.5 cutoff.
Using 0.5 here would silently reproduce the miscalibrated macro-F1 ≈ 0.58
result from the Stage 3 verification run -- tuned thresholds are what get
this model to macro-F1 ≈ 0.79 on held-out test data.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CATEGORIES = [
    "Limitation of liability", "Unilateral termination", "Unilateral change",
    "Content removal", "Contract by using", "Choice of law",
    "Jurisdiction", "Arbitration",
]

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "distilbert"
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parents[2] / "models" / "thresholds.json"


@dataclass
class TransformerBundle:
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    thresholds: np.ndarray  # shape (n_categories,), in CATEGORIES order
    device: torch.device
    max_length: int = 256


def load_thresholds(thresholds_path: Path = DEFAULT_THRESHOLDS_PATH) -> np.ndarray:
    """Load per-category thresholds tuned in Stage 4. Falls back to 0.5
    per category (with a loud warning) if the file is missing, so the
    service degrades rather than crashes -- but this should never happen
    in a properly deployed build."""
    if not thresholds_path.exists():
        import warnings
        warnings.warn(
            f"thresholds.json not found at {thresholds_path} -- falling back to "
            "flat 0.5 thresholds. This will noticeably hurt accuracy; see Stage 4 notes."
        )
        return np.full(len(CATEGORIES), 0.5)

    with open(thresholds_path) as f:
        data = json.load(f)

    cats = data["categories"]
    thresh_map = data["distilbert_thresholds"]
    # preserve CATEGORIES order regardless of file order
    return np.array([thresh_map[c] for c in CATEGORIES], dtype=np.float32)


def load_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
) -> TransformerBundle:
    """Load the fine-tuned DistilBERT model, tokenizer, and tuned thresholds.
    Call this once at backend startup, not per-request."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    if model.config.num_labels != len(CATEGORIES):
        raise ValueError(
            f"Loaded model has {model.config.num_labels} output labels, "
            f"expected {len(CATEGORIES)}. Wrong checkpoint?"
        )

    thresholds = load_thresholds(thresholds_path)

    return TransformerBundle(
        model=model, tokenizer=tokenizer, thresholds=thresholds, device=device,
    )


def predict_proba_matrix(
    bundle: TransformerBundle, texts: List[str], batch_size: int = 32
) -> np.ndarray:
    """Run inference and return raw sigmoid probabilities, shape
    (len(texts), len(CATEGORIES)). No thresholding applied here."""
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = bundle.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=bundle.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(bundle.device) for k, v in enc.items()}
            out = bundle.model(**enc)
            all_logits.append(out.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    return 1 / (1 + np.exp(-logits))  # sigmoid, multi-label


def predict(bundle: TransformerBundle, texts: List[str], batch_size: int = 32) -> dict:
    """Full prediction: probabilities + thresholded 0/1 flags per category,
    per input text. Returns a dict ready to serialize into an API response."""
    probs = predict_proba_matrix(bundle, texts, batch_size=batch_size)
    preds = (probs >= bundle.thresholds[None, :]).astype(int)

    results = []
    for i, text in enumerate(texts):
        results.append(
            {
                "text": text,
                "predictions": {
                    cat: bool(preds[i, c]) for c, cat in enumerate(CATEGORIES)
                },
                "probabilities": {
                    cat: float(probs[i, c]) for c, cat in enumerate(CATEGORIES)
                },
                "is_unfair": bool(preds[i].any()),
            }
        )
    return {"model": "distilbert", "results": results}


if __name__ == "__main__":
    # Quick smoke test -- python -m src.models.transformer
    bundle = load_model()
    sample = [
        "We may terminate your account at any time, for any reason, without notice.",
        "You will receive a full refund within 30 days of purchase.",
    ]
    out = predict(bundle, sample)
    print(json.dumps(out, indent=2))
