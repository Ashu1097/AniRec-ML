# -*- coding: utf-8 -*-
"""Unit tests for FeedbackStore."""

import tempfile
from pathlib import Path

import pytest

from src.data.feedback import FeedbackStore


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(tmp_path / "feedback_test.db")


class TestFeedbackStore:
    def test_record_and_retrieve(self, store):
        store.record("user1", 42, 1)
        signals = store.get_user_signals("user1")
        assert signals.get(42) == 1

    def test_negative_signal(self, store):
        store.record("user1", 99, -1)
        disliked = store.get_disliked_items("user1")
        assert 99 in disliked

    def test_invalid_signal_raises(self, store):
        with pytest.raises(ValueError):
            store.record("user1", 1, 0)

    def test_upsert_behaviour(self, store):
        store.record("user1", 5, 1)
        store.record("user1", 5, -1)  # overwrite
        signals = store.get_user_signals("user1")
        assert signals.get(5) == -1

    def test_count(self, store):
        store.record("u1", 1, 1)
        store.record("u1", 2, -1)
        assert store.count() == 2

    def test_bpr_pairs(self, store):
        store.record("user1", 10, 1)
        store.record("user1", 20, -1)
        uid_map  = {"user1": 0}
        iid_map  = {10: 0, 20: 1}
        pos, neg = store.get_bpr_pairs(uid_map, iid_map)
        assert (0, 0) in pos
        assert (0, 1) in neg

    def test_unknown_user_empty(self, store):
        signals  = store.get_user_signals("nobody")
        disliked = store.get_disliked_items("nobody")
        assert signals == {}
        assert disliked == set()

    def test_repr(self, store):
        assert "FeedbackStore" in repr(store)