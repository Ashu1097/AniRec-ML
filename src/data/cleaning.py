# -*- coding: utf-8 -*-
"""Text and metadata cleaning utilities."""

import html
import re
import unicodedata
from typing import Dict, List, Optional

import numpy as np

_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_SPECIAL = re.compile(r"[^a-z0-9\s]")
_RE_WHITESPACE = re.compile(r"\s+")

_GENRE_ALIASES: Dict[str, str] = {
    "sci-fi": "Sci-Fi", "sci fi": "Sci-Fi", "science fiction": "Sci-Fi",
    "slice of life": "Slice of Life", "slice-of-life": "Slice of Life",
    "shounen": "Shounen", "shonen": "Shounen", "shoujo": "Shoujo",
    "seinen": "Seinen", "josei": "Josei", "mecha": "Mecha", "isekai": "Isekai",
    "romance": "Romance", "romantic": "Romance", "action": "Action",
    "adventure": "Adventure", "comedy": "Comedy", "drama": "Drama",
    "fantasy": "Fantasy", "horror": "Horror", "mystery": "Mystery",
    "psychological": "Psychological", "thriller": "Thriller",
    "sports": "Sports", "sport": "Sports", "supernatural": "Supernatural",
    "music": "Music", "history": "Historical", "historical": "Historical",
    "war": "War", "animation": "Animation", "documentary": "Documentary",
    "family": "Family", "crime": "Crime",
}


def clean_text(
    text: Optional[str],
    *,
    min_words: int = 3,
    max_words: int = 200,
) -> str:
    """Strip HTML, URLs, non-ASCII, and normalise whitespace."""
    if not text or not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _RE_HTML_TAG.sub(" ", t)
    t = _RE_URL.sub(" ", t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = _RE_SPECIAL.sub(" ", t.lower())
    t = _RE_WHITESPACE.sub(" ", t).strip()
    words = t.split()
    if len(words) < min_words:
        return ""
    return " ".join(words[:max_words])


def clean_genres(genres: List[str], *, remove_adult: bool = True) -> List[str]:
    """Canonicalise genre names and optionally remove adult categories."""
    _adult = {"Adult", "Hentai", "Short", "Talk Show", "Reality TV", "Game Show"}
    seen, result = set(), []
    for g in genres:
        if not g or not isinstance(g, str):
            continue
        canonical = _GENRE_ALIASES.get(g.strip().lower(), g.strip().title())
        if remove_adult and canonical in _adult:
            continue
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def normalize_popularity(counts, clip_pct=95.0):
    arr = np.asarray(counts, dtype=np.float32)

    if arr.size == 0:
        return arr

    if np.all(arr == 0):
        return np.zeros_like(arr, dtype=np.float32)

    upper = np.percentile(arr, clip_pct)

    arr = np.clip(arr, 0, upper)

    arr = np.log1p(arr)

    ranks = arr.argsort().argsort().astype(np.float32)

    if ranks.max() > 0:
        ranks /= ranks.max()

    return ranks.astype(np.float32)


def normalize_ratings(ratings: np.ndarray) -> np.ndarray:
    """Clip to [0, 10] and divide by 10."""
    return (np.clip(ratings.astype(np.float64), 0, 10) / 10.0).astype(np.float32)
