"""Normalize Persian text before indexing/search."""

import re

import hazm

_normalizer = hazm.Normalizer()

_DIGIT_TABLE = str.maketrans(
    "".join(chr(i) for i in range(0x0660, 0x066A))  # Arabic-Indic 0-9
    + "".join(chr(i) for i in range(0x06F0, 0x06FA)),  # Persian 0-9
    "0123456789" * 2,
)


def normalize(text: str) -> str:
    """Return a normalized copy of *text*."""
    text = _normalizer.normalize(text)
    text = text.translate(_DIGIT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
