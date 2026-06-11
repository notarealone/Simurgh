"""Split text into overlapping character windows."""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Return overlapping windows of *text* as :class:`Chunk` objects."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    idx = 0
    start = 0

    while start < len(text):
        window = text[start : start + chunk_size].strip()
        start += step
        if not window:
            continue
        chunks.append(Chunk(text=window, source=source, index=idx))
        idx += 1

    return chunks
