"""Tests for the Canon + memory store.

One SQLite file holds both what Darrick said (Canon) and what Neil told the bot
(episodic memory). They must never bleed into each other's search results.
"""

import numpy as np
import pytest

from brain.store.db import Store
from pipeline.transcript import Chunk


def chunk(text: str, episode: str = "ep1", start: float = 0.0) -> Chunk:
    return Chunk(text=text, speakers=("DARRICK",), episode=episode, start=start, end=start + 10)


def vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


class TestCanon:
    def test_empty_search_returns_nothing(self, store):
        assert store.search_canon(vec(1, 0), k=5) == []

    def test_add_then_find(self, store):
        store.add_chunks([chunk("wemby is him")], [vec(1, 0)])
        hits = store.search_canon(vec(1, 0), k=5)
        assert len(hits) == 1
        assert hits[0].text == "wemby is him"

    def test_ranks_by_similarity(self, store):
        store.add_chunks(
            [chunk("close", start=0), chunk("far", start=20)],
            [vec(1, 0.1), vec(0, 1)],
        )
        hits = store.search_canon(vec(1, 0), k=5)
        assert [h.text for h in hits] == ["close", "far"]

    def test_k_limits_results(self, store):
        store.add_chunks(
            [chunk(f"c{i}", start=i * 20) for i in range(5)],
            [vec(1, i / 10) for i in range(5)],
        )
        assert len(store.search_canon(vec(1, 0), k=2)) == 2

    def test_reingesting_the_same_chunk_does_not_duplicate(self, store):
        store.add_chunks([chunk("same")], [vec(1, 0)])
        added = store.add_chunks([chunk("same")], [vec(1, 0)])
        assert added == 0
        assert len(store.search_canon(vec(1, 0), k=10)) == 1

    def test_hit_carries_provenance(self, store):
        store.add_chunks([chunk("take", episode="ep42", start=90.0)], [vec(1, 0)])
        hit = store.search_canon(vec(1, 0), k=1)[0]
        assert hit.episode == "ep42"
        assert hit.start == 90.0

    def test_mismatched_lengths_raise(self, store):
        with pytest.raises(ValueError, match="same number"):
            store.add_chunks([chunk("a"), chunk("b")], [vec(1, 0)])

    def test_persists_across_reopen(self, tmp_path):
        path = tmp_path / "p.db"
        Store(path).add_chunks([chunk("durable")], [vec(1, 0)])
        assert Store(path).search_canon(vec(1, 0), k=1)[0].text == "durable"


class TestMemory:
    def test_add_and_search(self, store):
        store.add_memory("neil roots for okc", vec(1, 0))
        hits = store.search_memories(vec(1, 0), k=5)
        assert hits[0].text == "neil roots for okc"

    def test_memories_are_not_returned_by_canon_search(self, store):
        store.add_memory("private note", vec(1, 0))
        assert store.search_canon(vec(1, 0), k=5) == []

    def test_canon_is_not_returned_by_memory_search(self, store):
        store.add_chunks([chunk("a take")], [vec(1, 0)])
        assert store.search_memories(vec(1, 0), k=5) == []

    def test_recent_returns_newest_first(self, store):
        store.add_memory("first", vec(1, 0))
        store.add_memory("second", vec(1, 0))
        assert [m.text for m in store.recent_memories(limit=2)] == ["second", "first"]

    def test_recent_respects_limit(self, store):
        for i in range(5):
            store.add_memory(f"m{i}", vec(1, 0))
        assert len(store.recent_memories(limit=3)) == 3


class TestStats:
    def test_counts_both_stores(self, store):
        store.add_chunks([chunk("a")], [vec(1, 0)])
        store.add_memory("b", vec(1, 0))
        assert store.stats() == {"canon_chunks": 1, "memories": 1}
