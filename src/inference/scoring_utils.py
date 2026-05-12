# -*- coding: utf-8 -*-
"""Score normalisation and sharpening utilities used by InferenceScorer."""

from typing import Optional

import numpy as np


def safe_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns 0.5 array if range < ε."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def zscore_normalize(
    x: np.ndarray,
    clip_sigma: float = 2.5,
    target_lo: float = 0.05,
    target_hi: float = 0.95,
) -> np.ndarray:
    """Z-score normalisation with σ-clipping, then rescale to [target_lo, target_hi]."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 1:
        return np.full_like(x, (target_lo + target_hi) / 2)
    mu = float(x.mean())
    std = float(x.std())
    if std < 1e-8:
        ranks = np.argsort(np.argsort(-x)).astype(np.float32)
        return (target_hi - (target_hi - target_lo) * ranks / max(len(x) - 1, 1)).astype(np.float32)
    z = np.clip((x - mu) / std, -clip_sigma, clip_sigma)
    lo, hi = z.min(), z.max()
    if hi - lo < 1e-8:
        return np.full_like(x, (target_lo + target_hi) / 2)
    return (target_lo + (z - lo) / (hi - lo) * (target_hi - target_lo)).astype(np.float32)


def sharpen_scores(
    scores: np.ndarray,
    method: str = "tanh",
    strength: float = 2.0,
    center: float = 0.5,
) -> np.ndarray:
    """Non-linear sharpening to push scores away from the centre."""
    scores = np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    if method == "tanh":
        shifted = np.tanh(strength * (scores - center))
        lo, hi = np.tanh(-strength), np.tanh(strength)
        out = (shifted - lo) / max(hi - lo, 1e-8)
    elif method == "sigmoid":
        out = 1.0 / (1.0 + np.exp(-strength * (scores - center)))
        lo = 1.0 / (1.0 + np.exp(strength))
        hi = 1.0 / (1.0 + np.exp(-strength))
        out = (out - lo) / max(hi - lo, 1e-8)
    elif method == "power":
        out = np.where(
            scores >= center,
            center + (1 - center) * ((scores - center) / max(1 - center, 1e-8)) ** (1 / strength),
            center - center * ((center - scores) / max(center, 1e-8)) ** (1 / strength),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def enforce_score_spread(
    scores: np.ndarray,
    min_std: float = 0.05,
    target_lo: float = 0.10,
    target_hi: float = 0.95,
) -> np.ndarray:
    """If std < min_std, force rank-based spread across [target_lo, target_hi]."""
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) <= 1 or float(scores.std()) >= min_std:
        return scores
    n = len(scores)
    ranks = np.argsort(np.argsort(-scores))
    return (target_hi - (target_hi - target_lo) * (ranks / max(n - 1, 1))).astype(np.float32)


def power_scale(scores: np.ndarray, exponent: float = 1.5) -> np.ndarray:
    """Apply x^exponent power scaling after clipping to [0, 1]."""
    return np.power(np.clip(scores, 0.0, 1.0), exponent).astype(np.float32)

def calibrate_scores(scores):
    if not scores:
        return []

    lo = min(scores)
    hi = max(scores)

    if hi - lo == 0:
        return [1.0 for _ in scores]

    return [(s - lo) / (hi - lo) for s in scores]

def compute_genre_match_score(genres_a, genres_b):
    """
    Simple genre overlap score.
    """

    set_a = set(genres_a or [])
    set_b = set(genres_b or [])

    if not set_a or not set_b:
        return 0.0

    overlap = len(set_a & set_b)
    union = len(set_a | set_b)

    return overlap / union

def diversity_penalty_scores(scores, penalty=0.1):
    """
    Apply a simple diversity penalty across ranked scores.
    """

    if not scores:
        return []

    adjusted = []

    for idx, score in enumerate(scores):
        adjusted_score = score - (idx * penalty)
        adjusted.append(adjusted_score)

    return adjusted

def popularity_penalty(score, popularity, alpha=0.1):
    """
    Penalize overly popular items slightly.
    """

    try:
        popularity = float(popularity)
    except Exception:
        popularity = 0.0

    return score - (alpha * popularity)