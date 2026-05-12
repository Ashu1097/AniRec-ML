# -*- coding: utf-8 -*-
"""Unit tests for DiskCache."""

import pytest

from src.utils.cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache(tmp_path, "test_prefix")


class TestDiskCache:
    def test_set_and_get(self, cache):
        cache.set("key1", {"data": 42})
        result = cache.get("key1")
        assert result == {"data": 42}

    def test_missing_key_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_exists(self, cache):
        assert not cache.exists("k")
        cache.set("k", [1, 2, 3])
        assert cache.exists("k")

    def test_delete(self, cache):
        cache.set("del_me", 99)
        cache.delete("del_me")
        assert not cache.exists("del_me")

    def test_delete_nonexistent_silent(self, cache):
        cache.delete("ghost")  # should not raise

    def test_refuses_none(self, cache):
        with pytest.raises(ValueError):
            cache.set("bad", None)

    def test_overwrite(self, cache):
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"