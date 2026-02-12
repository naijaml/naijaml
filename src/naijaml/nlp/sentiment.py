"""Nigerian language sentiment analysis.

Provides sentiment classification for Nigerian languages (Yorùbá, Hausa, Igbo, Pidgin)
using a TF-IDF + Logistic Regression model trained on NaijaSenti.

No heavy dependencies - uses only numpy for inference.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Model file path
_MODEL_PATH = Path(__file__).parent / "sentiment_model.json"

# Cached model
_model: Optional[Dict] = None


def _load_model() -> Dict:
    """Load the sentiment model."""
    global _model

    if _model is not None:
        return _model

    if not _MODEL_PATH.exists():
        raise RuntimeError(
            "Sentiment model not found. Run the training script first:\n"
            "  uv run python scripts/train_tfidf_sentiment.py"
        )

    with open(_MODEL_PATH, "r", encoding="utf-8") as f:
        _model = json.load(f)

    # Convert lists back to numpy arrays for fast inference
    _model["idf"] = np.array(_model["idf"])
    _model["coef"] = np.array(_model["coef"])
    _model["intercept"] = np.array(_model["intercept"])

    return _model


def _tokenize(text: str, ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    """Tokenize text into word n-grams."""
    # Normalize unicode and lowercase
    text = unicodedata.normalize("NFC", text.lower())

    # Simple word tokenization (keep diacritics)
    words = re.findall(r'\b\w+\b', text)

    tokens = []

    # Generate n-grams
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            tokens.append(ngram)

    return tokens


def _compute_tfidf(tokens: List[str], model: Dict) -> np.ndarray:
    """Compute TF-IDF vector for tokens."""
    vocab = model["vocab"]
    idf = model["idf"]

    # Initialize sparse vector
    vec = np.zeros(len(idf))

    # Count term frequencies
    tf = {}
    for token in tokens:
        if token in vocab:
            idx = vocab[token]
            tf[idx] = tf.get(idx, 0) + 1

    # Apply sublinear TF (log(1 + tf)) and IDF
    for idx, count in tf.items():
        vec[idx] = (1 + math.log(count)) * idf[idx]

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def analyze_sentiment(text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """Analyze sentiment of Nigerian language text.

    Supports Yorùbá, Hausa, Igbo, Nigerian Pidgin, and English.

    Args:
        text: Input text to analyze.

    Returns:
        Dictionary with:
        - label: Predicted sentiment ("positive", "negative", or "neutral")
        - confidence: Confidence score (0.0 to 1.0)
        - scores: Full probability distribution over labels

    Example:
        >>> analyze_sentiment("This film too sweet!")
        {'label': 'positive', 'confidence': 0.85, 'scores': {...}}

        >>> analyze_sentiment("I no like am at all")
        {'label': 'negative', 'confidence': 0.91, 'scores': {...}}
    """
    if not text or not text.strip():
        return {
            "label": "neutral",
            "confidence": 0.33,
            "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
        }

    model = _load_model()

    # Tokenize
    ngram_range = tuple(model.get("ngram_range", [1, 2]))
    tokens = _tokenize(text, ngram_range)

    # Compute TF-IDF
    tfidf_vec = _compute_tfidf(tokens, model)

    # Compute logits: X @ W^T + b
    logits = tfidf_vec @ model["coef"].T + model["intercept"]

    # Softmax for probabilities
    probs = _softmax(logits)

    # Get prediction
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    id2label = model["id2label"]
    label = id2label[str(pred_id)]

    # Build scores dict
    scores = {id2label[str(i)]: float(probs[i]) for i in range(len(probs))}

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def get_sentiment(text: str) -> str:
    """Get sentiment label for text (simplified API).

    Args:
        text: Input text.

    Returns:
        Sentiment label: "positive", "negative", or "neutral".

    Example:
        >>> get_sentiment("E sweet die!")
        'positive'
    """
    return analyze_sentiment(text)["label"]


def get_sentiment_with_confidence(text: str) -> Tuple[str, float]:
    """Get sentiment label and confidence score.

    Args:
        text: Input text.

    Returns:
        Tuple of (label, confidence).

    Example:
        >>> get_sentiment_with_confidence("Na wa o, this thing bad!")
        ('negative', 0.87)
    """
    result = analyze_sentiment(text)
    return result["label"], result["confidence"]


def analyze_batch(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """Analyze sentiment for multiple texts.

    Args:
        texts: List of input texts.

    Returns:
        List of sentiment results (same format as analyze_sentiment).

    Example:
        >>> texts = ["I love it!", "I hate it!", "It's okay"]
        >>> results = analyze_batch(texts)
        >>> [r["label"] for r in results]
        ['positive', 'negative', 'neutral']
    """
    return [analyze_sentiment(text) for text in texts]


def is_available() -> bool:
    """Check if sentiment analysis is available (model exists).

    Returns:
        True if the sentiment model is installed.
    """
    return _MODEL_PATH.exists()
