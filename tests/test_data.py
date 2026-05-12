# -*- coding: utf-8 -*-
"""Tests for data cleaning, caching, and utility functions."""

import os
import tempfile

import numpy as np
import pytest

from src.data.cleaning import (
    clean_genres,
    clean_text,
    normalize_popularity,
    normalize_ratings,
)
from src.data.cache import DiskCache


class TestCleanText:
    def test_basic_cleaning(self):
        text = "Hello <b>World</b>! Visit http://example.com"
        out = clean_text(text)
        assert "<b>" not in out
        assert "http" not in out

    def test_html_entities(self):
        text = "Attack &amp; Titan &lt;great&gt;"
        out = clean_text(text)
        assert "&amp;" not in out
        assert "&lt;" not in out

    def test_too_short_returns_empty(self):
        out = clean_text("Hi")
        assert out == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_max_words_truncation(self):
        long_text = " ".join(["word"] * 300)
        out = clean_text(long_text, max_words=200)
        assert len(out.split()) <= 200

    def test_unicode_normalisation(self):
        text = "Neon Génesis Evangelion is great anime"
        out = clean_text(text)
        assert isinstance(out, str)
        assert len(out) > 0


class TestCleanGenres:
    def test_canonical_mapping(self):
        out = clean_genres(["sci-fi", "slice of life"])
        assert "Sci-Fi" in out
        assert "Slice of Life" in out

    def test_removes_adult(self):
        out = clean_genres(["Action", "Adult", "Hentai"], remove_adult=True)
        assert "Adult" not in out
        assert "Hentai" not in out
        assert "Action" in out

    def test_deduplication(self):
        out = clean_genres(["Action", "action", "ACTION"])
        assert len(out) == 1

    def test_empty_input(self):
        assert clean_genres([]) == []

    def test_none_entries_skipped(self):
        out = clean_genres([None, "Action", ""])
        assert "Action" in out
        assert len(out) == 1


class TestNormalizePopularity:
    def test_output_range(self):
        counts = np.array([0, 10, 100, 1000, 10000], dtype=np.float64)
        out = normalize_popularity(counts)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-6

    def test_all_zeros(self):
        counts = np.zeros(10, dtype=np.float64)
        out = normalize_popularity(counts)
        assert np.all(out == 0.0)

    def test_monotone(self):
        counts = np.array([1, 5, 10, 50, 100], dtype=np.float64)
        out = normalize_popularity(counts)
        assert np.all(np.diff(out) > 0)

    def test_outlier_clipping(self):
        counts = np.array([1] * 99 + [1_000_000], dtype=np.float64)
        out = normalize_popularity(counts, clip_pct=95.0)
        # The outlier should not dominate; the 99 regular items should have spread
        assert float(out[:99].max()) > 0.5


class TestNormalizeRatings:
    def test_output_range(self):
        ratings = np.array([0.0, 5.0, 7.5, 10.0])
        out = normalize_ratings(ratings)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-6

    def test_clip_below_zero(self):
        ratings = np.array([-1.0, 0.0, 5.0])
        out = normalize_ratings(ratings)
        assert float(out[0]) == 0.0

    def test_clip_above_ten(self):
        ratings = np.array([10.0, 15.0])
        out = normalize_ratings(ratings)
        assert float(out[0]) == 1.0
        assert float(out[1]) == 1.0


class TestDiskCache:
    def test_set_and_get(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        cache.set("my_key", {"value": 42, "list": [1, 2, 3]})
        result = cache.get("my_key")
        assert result == {"value": 42, "list": [1, 2, 3]}

    def test_missing_key_returns_none(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        assert cache.get("nonexistent_key") is None

    def test_exists(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        assert not cache.exists("k")
        cache.set("k", [1, 2, 3])
        assert cache.exists("k")

    def test_delete(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        cache.set("k", 123)
        cache.delete("k")
        assert cache.get("k") is None

    def test_set_none_raises(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        with pytest.raises(ValueError):
            cache.set("k", None)

    def test_overwrite(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_unicode_values(self, tmp_path):
        cache = DiskCache(tmp_path, prefix="test")
        cache.set("anime", {"title": "進撃の巨人", "score": 9.0})
        result = cache.get("anime")
        assert result["title"] == "進撃の巨人"