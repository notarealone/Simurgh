"""Build a grounded RAG chat prompt from retrieved hits.

Two prompt variants are available: "en" (English instructions) and "fa" (Persian
instructions). Both instruct the model to answer in Persian; which one yields better
answers is model-dependent, so the variant is selectable for comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.store import Hit

_SYSTEM_EN = (
    "You are a Persian-language educational assistant. Answer the user's question using "
    "ONLY the retrieved passages below. If the answer is not in them, say so honestly and "
    "do not invent anything. Always write your answer in Persian (فارسی), in fluent prose, "
    "and end it with the bracketed numbers of the sources you used (for example [1] or [2])."
)

_SYSTEM_FA = (
    "شما یک دستیار آموزشی فارسی‌زبان هستید. فقط و فقط بر اساس «متن‌های بازیابی‌شده» که در ادامه "
    "می‌آیند به پرسش کاربر پاسخ دهید. اگر پاسخ در متن‌ها وجود ندارد، صادقانه بنویسید «بر اساس منابع "
    "موجود نمی‌دانم» و از خودتان چیزی نسازید. پاسخ را به فارسی روان بدهید و در پایانِ پاسخ، شمارهٔ "
    "منبع‌هایی را که استفاده کرده‌اید داخل کروشه بیاورید (مثلاً [1] یا [2])."
)

# Per-variant scaffolding: the system message plus the labels that wrap the retrieved context.
_VARIANTS = {
    "en": {
        "system": _SYSTEM_EN,
        "context_header": "Retrieved passages:",
        "source_label": "source",
        "empty": "(No documents found.)",
        "question_label": "Question",
    },
    "fa": {
        "system": _SYSTEM_FA,
        "context_header": "متن‌های بازیابی‌شده:",
        "source_label": "منبع",
        "empty": "(هیچ متنی یافت نشد.)",
        "question_label": "پرسش",
    },
}


def build_rag_prompt(query: str, hits: list[Hit], variant: str = "en") -> list[dict[str, str]]:
    """Assemble system + user messages for a grounded RAG turn.

    *variant* selects the prompt language ("en" or "fa"); both ask for a Persian answer.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant!r} (expected 'en' or 'fa')")
    v = _VARIANTS[variant]

    if not hits:
        context = v["empty"]
    else:
        parts = [
            f"[{i}] ({v['source_label']}: {hit.source}) {hit.raw_text}"
            for i, hit in enumerate(hits, start=1)
        ]
        context = "\n\n".join(parts)

    user_content = f"{v['context_header']}\n{context}\n\n{v['question_label']}: {query}"

    return [
        {"role": "system", "content": v["system"]},
        {"role": "user", "content": user_content},
    ]
