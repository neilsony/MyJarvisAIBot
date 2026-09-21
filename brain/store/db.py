"""One SQLite file holding both halves of what the bot knows.

  Canon    — what Darrick said on the show. Ingested once, read constantly.
  Memories — what Neil told the bot. Written constantly, read constantly.

They live in one file but never mix in search results: retrieving "your goal is
to move into PM work" when the bot was asked about Wemby would be a bug, and a
confusing one to trace.

**On vector search:** this brute-forces cosine similarity in numpy rather than
using a vector index. At this corpus size — ~500 episodes, ~25k chunks, 384
dimensions, under 40MB — a full scan runs in tens of milliseconds, which is
noise next to a network round trip to Claude. It also avoids a native
dependency that would need compiling on both the Mac and the Pi. If the corpus
ever grows an order of magnitude, swap the body of `_search` for a real index;
nothing outside this module needs to know.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.transcript import Chunk

__all__ = ["MemoryRecord", "SearchHit", "Store"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS canon (
    id        INTEGER PRIMARY KEY,
    hash      TEXT NOT NULL UNIQUE,
    text      TEXT NOT NULL,
    speakers  TEXT NOT NULL,
    episode   TEXT NOT NULL,
    start     REAL NOT NULL,
    end       REAL NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'note',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'subsec')),
    embedding  BLOB NOT NULL
);
"""


@dataclass(frozen=True)
class SearchHit:
    """A retrieved passage with enough provenance to check it."""

    text: str
    score: float
    episode: str | None
    start: float | None


@dataclass(frozen=True)
class MemoryRecord:
    text: str
    kind: str
    created_at: str


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    """Canon and memory, in one SQLite file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- Canon ---------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], embeddings: list[np.ndarray]) -> int:
        """Insert chunks, skipping any already present. Returns how many were new.

        Dedup is by content hash so re-running ingestion after adding episodes
        is safe and cheap — the common case, since the corpus grows weekly.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must be the same number "
                f"({len(chunks)} vs {len(embeddings)})"
            )

        added = 0
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            digest = hashlib.sha256(
                f"{chunk.episode}|{chunk.start}|{chunk.text}".encode()
            ).hexdigest()
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO canon "
                "(hash, text, speakers, episode, start, end, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    digest,
                    chunk.text,
                    ",".join(chunk.speakers),
                    chunk.episode,
                    chunk.start,
                    chunk.end,
                    _to_blob(embedding),
                ),
            )
            added += cursor.rowcount
        self._conn.commit()
        return added

    def search_canon(self, query: np.ndarray, k: int = 8) -> list[SearchHit]:
        """The `k` Canon passages closest to `query`."""
        rows = self._conn.execute(
            "SELECT text, episode, start, embedding FROM canon"
        ).fetchall()
        return self._rank(rows, query, k, with_provenance=True)

    def all_canon_text(self) -> list[str]:
        """Every ingested chunk's raw text, in no particular order.

        For offline analysis over the whole corpus — e.g.
        `pipeline.register_candidates` — rather than a similarity search.
        """
        rows = self._conn.execute("SELECT text FROM canon").fetchall()
        return [r["text"] for r in rows]

    # -- Memory --------------------------------------------------------------

    def add_memory(self, text: str, embedding: np.ndarray, kind: str = "note") -> None:
        """Record something worth remembering between conversations."""
        self._conn.execute(
            "INSERT INTO memories (text, kind, embedding) VALUES (?, ?, ?)",
            (text, kind, _to_blob(embedding)),
        )
        self._conn.commit()

    def search_memories(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        """The `k` memories closest to `query`."""
        rows = self._conn.execute("SELECT text, embedding FROM memories").fetchall()
        return self._rank(rows, query, k, with_provenance=False)

    def recent_memories(self, limit: int = 10) -> list[MemoryRecord]:
        """The most recent memories, newest first — the "what were we just
        talking about" channel, distinct from semantic recall."""
        rows = self._conn.execute(
            "SELECT text, kind, created_at FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            MemoryRecord(text=r["text"], kind=r["kind"], created_at=r["created_at"])
            for r in rows
        ]

    # -- Shared --------------------------------------------------------------

    def _rank(
        self,
        rows: list[sqlite3.Row],
        query: np.ndarray,
        k: int,
        *,
        with_provenance: bool,
    ) -> list[SearchHit]:
        if not rows:
            return []

        matrix = np.stack([_from_blob(r["embedding"]) for r in rows])
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
        # A zero-magnitude embedding scores 0 rather than producing a NaN that
        # would silently poison the ordering.
        scores = np.divide(
            matrix @ query, norms, out=np.zeros(len(rows), dtype=np.float32), where=norms != 0
        )

        best = np.argsort(-scores)[:k]
        return [
            SearchHit(
                text=rows[i]["text"],
                score=float(scores[i]),
                episode=rows[i]["episode"] if with_provenance else None,
                start=rows[i]["start"] if with_provenance else None,
            )
            for i in best
        ]

    def stats(self) -> dict[str, int]:
        """Row counts, for the CLI status line."""
        canon = self._conn.execute("SELECT COUNT(*) AS n FROM canon").fetchone()["n"]
        memories = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        return {"canon_chunks": int(canon), "memories": int(memories)}
