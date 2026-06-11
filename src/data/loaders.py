"""Load raw text from document files."""

from pathlib import Path

import pypdf


def load_document(path: str | Path) -> str:
    """Extract raw text from a PDF, TXT, or Markdown file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = pypdf.PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
