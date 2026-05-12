# -*- coding: utf-8 -*-
"""Catalog loading helpers for AniRec v22."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

_DEFAULT_DRIVE_ROOT = os.environ.get("ANIREC_OUTPUT", "./AniRec_output")
_CKPT        = Path(_DEFAULT_DRIVE_ROOT) / "v22"
_DATA_DIR    = _CKPT / "data"
_EMBED_DIR   = _CKPT / "embeddings"
_CACHE_DIR   = _CKPT / "api_cache"

ANIME_CATALOG_PATH   = _DATA_DIR / "anime_catalog.json"
MOVIE_CATALOG_PATH   = _DATA_DIR / "movie_catalog.json"
INTERACTIONS_PATH    = _DATA_DIR / "interactions.json"
ML_INTERACTIONS_PATH = _DATA_DIR / "ml_interactions.json"


def load_cached_data():
    """Load all cached artefacts from disk."""
    anime_catalog: Dict[int, dict] = {}
    movie_catalog: Dict[int, dict] = {}
    anime_interactions: Dict[str, List[dict]] = {}
    movie_interactions: Dict[str, List[dict]] = {}

    if ANIME_CATALOG_PATH.exists():
        with open(ANIME_CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        anime_catalog = {
            int(item["id"]): item
            for item in data
            if isinstance(item, dict) and "id" in item
        }

    if MOVIE_CATALOG_PATH.exists():
        with open(MOVIE_CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        movie_catalog = {
            int(item["id"]): item
            for item in data
            if isinstance(item, dict) and "id" in item
        }

    if INTERACTIONS_PATH.exists():
        with open(INTERACTIONS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        interaction_source = (
            saved.get("anime", saved)
            if isinstance(saved, dict)
            else {}
        )
        for uname, raw_list in interaction_source.items():
            records = []
            for raw in raw_list:
                if not isinstance(raw, dict):
                    continue
                item_id = int(raw.get("item_id") or raw.get("anilist_id") or 0)
                records.append({
                    "item_id": item_id, "anilist_id": item_id,
                    "weight": float(raw.get("weight", 1.0)),
                    "ts": int(raw.get("ts", 0)),
                    "title": raw.get("title", ""),
                    "genres": raw.get("genres", []),
                })
            if records:
                anime_interactions[uname] = sorted(records, key=lambda x: x.get("ts", 0))
        movie_interactions.update(saved.get("movies", {}))

    if ML_INTERACTIONS_PATH.exists():
        with open(ML_INTERACTIONS_PATH, "r", encoding="utf-8") as f:
            movie_interactions.update(json.load(f))

    return anime_catalog, movie_catalog, anime_interactions, movie_interactions


def run_preprocess_pipeline(force: bool = False):
    """Full preprocessing pipeline stub — imports heavy deps lazily."""
    raise NotImplementedError(
        "run_preprocess_pipeline requires all API keys and dependencies. "
        "Run the notebook directly for full preprocessing."
    )