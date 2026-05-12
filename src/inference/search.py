# -*- coding: utf-8 -*-
"""SmartSearch: fuzzy item-title lookup for the inference scorer."""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from thefuzz import fuzz as _fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False


class SmartSearch:
    """
    Fast fuzzy title search over the anime + movie name dictionaries.

    Lookup order:
      1. Exact match (O(1) hash lookup).
      2. Substring containment (linear scan, sorted by length).
      3. Fuzzy ratio via *thefuzz* (if installed) or difflib.

    Args:
        anime_names:  ``{item_idx: title}`` for anime items.
        movie_names:  ``{item_idx: title}`` for movie items.
        item_text_np: Optional text-embedding matrix (unused in search,
                      kept for interface compatibility).
    """

    def __init__(
        self,
        anime_names: Dict[int, str],
        movie_names: Dict[int, str],
        item_text_np: Optional[np.ndarray] = None,
    ) -> None:
        self.item_text_np   = item_text_np
        self._idx_to_name:  Dict[int, str]  = {}
        self._name_to_idx:  Dict[str, int]  = {}
        self._all_names:    List[str]        = []
        self._all_ids:      List[int]        = []

        for idx, nm in {**anime_names, **movie_names}.items():
            self._idx_to_name[idx]      = nm
            self._name_to_idx[nm.lower()] = idx
            self._all_names.append(nm.lower())
            self._all_ids.append(idx)

    # ------------------------------------------------------------------
    def find(
        self, query: str, top_k: int = 1
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the best matching item index for a title query.

        Returns:
            ``(item_idx, matched_title)`` or ``(None, None)`` if no match.
        """
        if not query:
            return None, None
        q = query.lower().strip()

        # 1. Exact match
        if q in self._name_to_idx:
            idx = self._name_to_idx[q]
            return idx, self._idx_to_name[idx]

        # 2. Substring match
        matches = [(n, i) for n, i in zip(self._all_names, self._all_ids)
                   if q in n]
        if matches:
            matches.sort(key=lambda x: len(x[0]))
            nm, idx = matches[0]
            return idx, self._idx_to_name[idx]

        # 3. Fuzzy match
        if HAS_FUZZ:
            best_score, best_idx = 0, None
            for nm, idx in zip(self._all_names, self._all_ids):
                s = _fuzz.partial_ratio(q, nm)
                if s > best_score:
                    best_score, best_idx = s, idx
            if best_score >= 70 and best_idx is not None:
                return best_idx, self._idx_to_name[best_idx]
        else:
            close = difflib.get_close_matches(
                q, self._all_names, n=1, cutoff=0.6)
            if close:
                idx = self._name_to_idx[close[0]]
                return idx, self._idx_to_name[idx]

        return None, None

    # ------------------------------------------------------------------
    def find_many(
        self, queries: List[str]
    ) -> List[Tuple[Optional[int], Optional[str]]]:
        """Batch version of :meth:`find`."""
        return [self.find(q) for q in queries]