"""Score normalisation and sharpening utilities used by InferenceScorer."""

import numpy as np


def safe_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns 0.5 array if range < ε."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def zscore_normalize(
    x,
    target_lo=0.05,
    target_hi=0.95,
    clip_sigma=3.0,
):
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x

    ranks = x.argsort().argsort().astype(np.float32)

    if ranks.max() > 0:
        ranks /= ranks.max()

    out = target_lo + ranks * (target_hi - target_lo)

    return out.astype(np.float32)


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


def calibrate_scores(scores, domain_mask=None):
    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        return scores

    out = safe_normalize(scores)

    out = enforce_score_spread(out)

    if domain_mask is not None:
        out = out * np.asarray(domain_mask, dtype=np.float32)

    return out.astype(np.float32)


def compute_genre_match_score(cands, item_genres, user_gw=None):
    user_gw = user_gw or {}

    out = []

    for item_id in cands:
        genres = item_genres.get(item_id, [])

        score = 0.0

        for g in genres:
            score += user_gw.get(g, 0.0)

        out.append(score)

    return np.asarray(out, dtype=np.float32)


def diversity_penalty_scores(
    cands,
    scores,
    item_genres,
    penalty_strength=0.2,
):
    out = np.zeros(len(cands), dtype=np.float32)

    seen = []

    for i, item_id in enumerate(cands):
        genres = set(item_genres.get(item_id, []))

        penalty = 0.0

        for prev in seen:
            overlap = len(genres & prev)

            if overlap > 0:
                penalty += penalty_strength * overlap

        out[i] = -penalty

        seen.append(genres)

    return out


def popularity_penalty(popularity, alpha=1.0):
    popularity = np.asarray(popularity, dtype=np.float32)

    return 1.0 / (1.0 + alpha * popularity)
