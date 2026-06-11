"""SQLite FTS5 keyword-search knowledge base for the RAG pipeline."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from data.persian import normalize

if TYPE_CHECKING:
    from data.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    """A single search result returned by :meth:`KnowledgeBase.search`."""

    raw_text: str
    source: str
    chunk_index: int
    score: float


class KnowledgeBase:
    """Persistent keyword-search store backed by SQLite FTS5."""

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the database at *db_path*.

        Args:
            db_path: Filesystem path or ``":memory:"`` for a transient DB.
        """
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the FTS5 virtual table if it does not already exist."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks "
            "USING fts5(text, raw_text UNINDEXED, source UNINDEXED, "
            "chunk_index UNINDEXED, tokenize='unicode61')"
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index a batch of :class:`~data.chunking.Chunk` objects."""
        rows = [(normalize(chunk.text), chunk.text, chunk.source, chunk.index) for chunk in chunks]
        self.conn.executemany(
            "INSERT INTO chunks (text, raw_text, source, chunk_index) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        """Return up to *top_k* hits matching *query* (OR of terms)."""
        normalized = normalize(query)
        terms = normalized.split()
        if not terms:
            return []
        match_expr = " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms)
        cursor = self.conn.execute(
            "SELECT raw_text, source, chunk_index, bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks) LIMIT ?",
            (match_expr, top_k),
        )
        return [
            Hit(
                raw_text=row[0],
                source=row[1],
                chunk_index=int(row[2]),
                score=float(row[3]),
            )
            for row in cursor
        ]

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        result = self.conn.execute("SELECT count(*) FROM chunks").fetchone()
        return result[0]

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self.conn.execute("DELETE FROM chunks")
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
