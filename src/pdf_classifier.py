"""Lightweight PDF/topic classifier to help focus retrieval."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

from langchain_core.documents import Document

CATEGORY_KEYWORDS = {
    "goods": [
        "goods",
        "product",
        "item",
        "equipment",
        "device",
        "hardware",
        "delivery",
        "warranty",
    ],
    "services": [
        "service",
        "installation",
        "repair",
        "consultancy",
        "engineer",
        "maintenance",
        "contractor",
    ],
    "digital": [
        "software",
        "application",
        "app",
        "digital",
        "licence",
        "subscription",
        "online",
        "cloud",
    ],
    "finance": [
        "payment",
        "refund",
        "interest",
        "credit",
        "loan",
        "invoice",
        "settlement",
    ],
    "dispute": [
        "claim",
        "litigation",
        "dispute",
        "breach",
        "damages",
        "proceedings",
    ],
}

CATEGORY_DISPLAY = {
    "goods": "Goods",
    "services": "Services",
    "digital": "Digital",
    "finance": "Finance",
    "dispute": "Dispute",
    "general": "General",
}


def _infer_category_from_text(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(text_lower.count(keyword) for keyword in keywords)
        if score:
            scores[category] = score

    if not scores:
        return "general"
    # Pick category with highest score; tie broken alphabetically
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def classify_pdf_documents(documents: Sequence[Document], source: str | None = None) -> List[str]:
    """Annotate each Document with a category and return summary categories for the file."""
    categories: Set[str] = set()
    for doc in documents:
        category = _infer_category_from_text(doc.page_content or "")
        doc.metadata["category"] = category
        if source:
            doc.metadata["source_file"] = source
        categories.add(category)
    if not categories:
        categories.add("general")
    return sorted(categories)


def guess_question_categories(question: str, max_groups: int = 2) -> List[str]:
    """Suggest categories to focus retrieval for a question."""
    if not question:
        return []
    text_lower = question.lower()
    ranked = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(text_lower.count(keyword) for keyword in keywords)
        if score:
            ranked.append((category, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [category for category, _ in ranked[:max_groups]]


def format_category_list(categories: Iterable[str]) -> str:
    display_names = [CATEGORY_DISPLAY.get(cat, cat.title()) for cat in categories]
    return ", ".join(display_names)
