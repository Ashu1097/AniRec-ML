"""Tests for src.data.cleaning — text and metadata normalisation."""

import numpy as np
import pytest

from src.data.cleaning import (
    clean_genres,
    clean_text,
    normalize_popularity,
    normalize_ratings,
)


# ---------------------------------------------------------------------------
class TestCleanText:
    def test_basic_cleanup(self):
        assert clean_text("Hello World!", min_words=1) == "hello world"

    def test_html_stripped(self):
        result = clean_text("<b>Bold</b> and <i>italic</i>", min_words=1)
        assert "<b>" not in result
        assert "bold" in result

    def test_url_stripped(self):
        result = clean_text("Visit https://example.com for info", min_words=1)
        assert "http" not in result
        assert "visit" in result

    def test_empty_returns_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_min_words_filter(self):
        assert clean_text("ok", min_words=3) == ""
        assert clean_text("ok enough words here", min_words=3) != ""

    def test_max_words_truncation(self):
        text = " ".join(["word"] * 300)
        result = clean_text(text, min_words=1, max_words=10)
        assert len(result.split()) == 10

    def test_non_ascii_normalised(self):
        result = clean_text("Café au lait", min_words=1)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
class TestCleanGenres:
    def test_alias_normalisation(self):
        assert "Sci-Fi" in clean_genres(["sci-fi"])
        assert "Sci-Fi" in clean_genres(["science fiction"])

    def test_deduplication(self):
        result = clean_genres(["Action", "action", "ACTION"])
        assert result.count("Action") == 1

    def test_adult_removed_by_default(self):
        result = clean_genres(["Action", "Adult", "Drama"])
        assert "Adult" not in result
        assert "Action" in result

    def test_adult_kept_when_disabled(self):
        result = clean_genres(["Action", "Adult"], remove_adult=False)
        assert "Adult" in result

    def test_empty_list(self):
        assert clean_genres([]) == []

    def test_invalid_values_skipped(self):
        result = clean_genres([None, "", "Action"])
        assert "Action" in result
        assert len(result) == 1


# ---------------------------------------------------------------------------
class TestNormalizePopularity:
    def test_output_range(self):
        counts = np.array([0, 10, 100, 1000, 10000], dtype=np.float32)
        result = normalize_popularity(counts)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_all_zeros(self):
        counts = np.zeros(5, dtype=np.float32)
        result = normalize_popularity(counts)
        assert (result == 0).all()

    def test_monotonic(self):
        counts = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
        result = normalize_popularity(counts)
        assert (np.diff(result) >= 0).all()

    def test_dtype(self):
        counts = np.array([1, 2, 3], dtype=np.float32)
        result = normalize_popularity(counts)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
class TestNormalizeRatings:
    def test_basic(self):
        ratings = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        result = normalize_ratings(ratings)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-6)

    def test_clipping(self):
        ratings = np.array([-1.0, 11.0], dtype=np.float32)
        result = normalize_ratings(ratings)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_dtype(self):
        ratings = np.array([7.5], dtype=np.float64)
        result = normalize_ratings(ratings)
        assert result.dtype == np.float32
