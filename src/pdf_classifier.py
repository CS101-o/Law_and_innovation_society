# src/pdf_classifier.py
"""Improved PDF/topic classifier: ensures accurate categorization of PDFs for focused retrieval."""

from __future__ import annotations
from typing import Iterable, List, Sequence
from langchain_core.documents import Document
import re
import string

# --- Keywords per category ---
CATEGORY_KEYWORDS = {
    "goods": [
        "goods", "product", "products", "item", "equipment", "appliance",
        "device", "hardware", "delivery", "warranty", "sale of goods",
        "consumer goods", "merchantable quality", "fitness for purpose"
    ],
    "services": [
        "service", "services", "installation", "repair", "consultancy",
        "consultation", "engineer", "maintenance", "contractor",
        "performed with reasonable care", "service provider"
    ],
    "digital": [
        "software", "application", "app", "digital", "licence", "license",
        "subscription", "online", "cloud", "data", "download", "code"
    ],
    "finance": [
        "payment", "refund", "interest", "credit", "loan", "invoice",
        "settlement", "compensation", "price", "consideration"
    ],
    "dispute": [
        "claim", "litigation", "dispute", "breach", "damages",
        "proceedings", "tribunal", "court", "remedy"
    ],
}

# Hints from filenames
CATEGORY_FILE_HINTS = {
    "goods": ["goods", "product", "retail", "merchant"],
    "services": ["service", "services", "contractor", "installation", "employment"],
    "digital": ["digital", "software", "it", "data", "technology", "online"],
    "finance": ["finance", "payment", "credit", "loan", "interest", "price"],
    "dispute": ["dispute", "litigation", "tribunal", "appeal", "court"],
}

CATEGORY_DISPLAY = {
    "goods": "Goods",
    "services": "Services",
    "digital": "Digital",
    "finance": "Finance",
    "dispute": "Dispute",
    "general": "General",
}


def _normalize_text(text: str) -> str:
    """Lowercase and remove punctuation."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def _score_category(text: str, source: str | None = None) -> dict[str, float]:
    """Return weighted scores for each category based on text content and filename."""
    text_norm = _normalize_text(text)
    scores: dict[str, float] = {category: 0.0 for category in CATEGORY_KEYWORDS}

    # Count keyword occurrences
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            keyword_norm = _normalize_text(keyword)
            # Use word boundaries for exact matching
            matches = re.findall(rf"\b{re.escape(keyword_norm)}\b", text_norm)
            if matches:
                scores[category] += len(matches) * (1.5 if " " in keyword else 1.0)

    # Add filename hints
    if source:
        source_norm = _normalize_text(source)
        for category, hints in CATEGORY_FILE_HINTS.items():
            for hint in hints:
                hint_norm = _normalize_text(hint)
                if hint_norm in source_norm:
                    scores[category] += 2.5

    return scores


def classify_pdf_documents(documents: Sequence[Document], source: str | None = None) -> List[str]:
    """Assign category to each document chunk and return top 3 file-level categories."""
    aggregate_scores = {category: 0.0 for category in CATEGORY_KEYWORDS}
    for doc in documents:
        text = doc.page_content or ""
        scores = _score_category(text, source)
        # Assign the chunk category
        top_category, top_score = max(scores.items(), key=lambda item: item[1])
        doc.metadata["category"] = top_category if top_score > 0 else "general"
        if source:
            doc.metadata["source_file"] = source
        # Aggregate for the whole file
        for category, score in scores.items():
            aggregate_scores[category] += score

    # Get top 3 file-level categories
    ranked = sorted(aggregate_scores.items(), key=lambda item: -item[1])
    top_categories = [cat for cat, score in ranked[:3] if score > 0]
    if not top_categories:
        top_categories = ["general"]
    return top_categories


def guess_question_categories(question: str, max_groups: int = 2) -> List[str]:
    """Return top categories relevant to a question."""
    if not question:
        return []
    scores = _score_category(question)
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    return [cat for cat, score in ranked[:max_groups] if score > 0]


def format_category_list(categories: Iterable[str]) -> str:
    """Return display-ready string for categories."""
    categories = list(categories)
    if len(categories) > 1 and "general" in categories:
        categories = [c for c in categories if c != "general"]
    return ", ".join([CATEGORY_DISPLAY.get(c, c.title()) for c in categories])