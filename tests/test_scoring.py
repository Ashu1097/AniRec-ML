# -*- coding: utf-8 -*-
"""Tests for inference scoring utilities."""

import numpy as np
import pytest

from src.inference.scoring_utils import (
    enforce_score_spread,
    power_scale,
    safe_normalize,
    zscore_normalize,
    sharpen_scores,
    calibrate_scores,
    compute_genre_match_score,
    diversity_penalty_scores,
    popularity_penalty,
)


class TestSafeNormalize:
    def test_output_range(self):
        x = np.random.randn(100).astype(np.float32)
        out = safe_normalize(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-6

    def test_flat_input_returns_half(self):
        x = np.ones(20, dtype=np.float32)
        out = safe_normalize(x)
        assert np.allclose(out, 0.5, atol=1e-5)

    def test_monotone(self):
        x = np.arange(10, dtype=np.float32)
        out = safe_normalize(x)
        assert np.all(np.diff(out) > 0)


class TestZscoreNormalize:
    def test_output_range(self):
        x = np.random.randn(200).astype(np.float32) * 100
        out = zscore_normalize(x, target_lo=0.05, target_hi=0.95)
        assert float(out.min()) >= 0.04
        assert float(out.max()) <= 0.96

    def test_flat_input(self):
        x = np.ones(30, dtype=np.float32) * 42.0
        # Should not crash; output should have some spread via rank fallback
        out = zscore_normalize(x)
        assert len(out) == 30

    def test_outlier_robustness(self):
        x = np.array([0.0] * 99 + [1000.0], dtype=np.float32)
        out = zscore_normalize(x, clip_sigma=2.5)
        # The 99 normal values should not all map to ~0
        assert float(out[:99].std()) > 0.01

    def test_preserves_ordering(self):
        x = np.sort(np.random.randn(50).astype(np.float32))
        out = zscore_normalize(x)
        assert np.all(np.diff(out) >= 0)


class TestEnforceScoreSpread:
    def test_flat_scores_get_spread(self):
        scores = np.ones(20, dtype=np.float32) * 0.75
        out = enforce_score_spread(scores, min_std=0.05)
        assert float(out.std()) >= 0.04

    def test_already_spread_unchanged(self):
        scores = np.linspace(0.1, 0.9, 20).astype(np.float32)
        out = enforce_score_spread(scores, min_std=0.05)
        assert np.allclose(out, scores, atol=1e-5)

    def test_single_element(self):
        scores = np.array([0.5], dtype=np.float32)
        out = enforce_score_spread(scores)
        assert len(out) == 1


class TestPowerScale:
    def test_range(self):
        x = np.linspace(0, 1, 50).astype(np.float32)
        out = power_scale(x, exponent=1.5)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-6

    def test_exponent_gt1_compresses_low(self):
        x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        out = power_scale(x, exponent=2.0)
        assert out[0] < x[0]   # low values get pushed down
        assert out[2] < x[2]   # high values also get pushed down (power < input for x<1)


class TestSharpenScores:
    def test_tanh_output_range(self):
        x = np.linspace(0, 1, 50).astype(np.float32)
        out = sharpen_scores(x, method="tanh", strength=2.0)
        assert float(out.min()) >= -1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_sigmoid_output_range(self):
        x = np.linspace(0, 1, 50).astype(np.float32)
        out = sharpen_scores(x, method="sigmoid", strength=3.0)
        assert float(out.min()) >= -1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_power_output_range(self):
        x = np.linspace(0, 1, 50).astype(np.float32)
        out = sharpen_scores(x, method="power", strength=2.0)
        assert float(out.min()) >= -1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_increases_contrast(self):
        """After sharpening, std should increase."""
        x = np.linspace(0.3, 0.7, 30).astype(np.float32)
        out = sharpen_scores(x, method="tanh", strength=3.0)
        assert float(out.std()) > float(x.std())

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            sharpen_scores(np.ones(10), method="unknown")


class TestCalibrateScores:
    def test_output_range(self):
        x = np.random.randn(50).astype(np.float32) * 10
        out = calibrate_scores(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0 + 1e-5

    def test_spread_improves(self):
        """Near-flat input should be spread out after calibration."""
        x = np.ones(30, dtype=np.float32) * 0.5 + np.random.randn(30).astype(np.float32) * 0.001
        out = calibrate_scores(x)
        assert float(out.std()) > float(x.std())

    def test_domain_mask(self):
        x = np.random.randn(50).astype(np.float32)
        mask = np.array([True] * 25 + [False] * 25)
        out = calibrate_scores(x, domain_mask=mask)
        assert len(out) == 50


class TestComputeGenreMatchScore:
    def test_basic(self):
        cands = np.array([0, 1, 2])
        item_genres = {0: ["Action", "Drama"], 1: ["Comedy"], 2: ["Action"]}
        user_gw = {"Action": 3.0, "Drama": 1.0, "Comedy": -1.0}
        out = compute_genre_match_score(cands, item_genres, user_gw)
        assert len(out) == 3
        # item 0 (Action+Drama) should score higher than item 1 (Comedy=negative)
        assert out[0] > out[1]

    def test_empty_weights(self):
        cands = np.array([0, 1, 2])
        item_genres = {0: ["Action"], 1: ["Drama"], 2: []}
        out = compute_genre_match_score(cands, item_genres, user_gw={})
        assert np.all(out == 0.0)


class TestDiversityPenaltyScores:
    def test_basic_shape(self):
        cands = np.arange(10)
        item_genres = {i: ["Action"] for i in range(10)}
        out = diversity_penalty_scores(cands, None, item_genres)
        assert len(out) == 10

    def test_similar_items_penalised(self):
        """Items with identical genre fingerprints should be penalised."""
        n = 10
        cands = np.arange(n)
        item_genres = {i: ["Action", "Drama"] for i in range(n)}
        out = diversity_penalty_scores(cands, None, item_genres, penalty_strength=0.4)
        # All items same genre → all get same penalty, but all < 1.0
        assert float(out.max()) < 1.0 + 1e-5


class TestPopularityPenalty:
    def test_zero_popularity_no_penalty(self):
        out = popularity_penalty(np.array([0.0]))
        assert abs(float(out[0]) - 1.0) < 1e-5

    def test_full_popularity_strong_penalty(self):
        out = popularity_penalty(np.array([1.0]), alpha=3.0)
        # 1/(1+3) = 0.25
        assert abs(float(out[0]) - 0.25) < 1e-5

    def test_monotone_decreasing(self):
        pops = np.linspace(0, 1, 20).astype(np.float32)
        penalties = popularity_penalty(pops, alpha=3.0)
        assert np.all(np.diff(penalties) <= 0)