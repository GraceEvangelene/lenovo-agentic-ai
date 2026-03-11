"""Tools for summarizing product reviews into sales insights."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _get_reviews_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "reviews.json"


def load_reviews() -> dict[str, list[dict]]:
    """Return a mapping: product_name -> list of reviews."""
    with open(_get_reviews_path(), "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[dict]] = {}
    for item in data:
        out[item["product"]] = item.get("reviews", [])
    return out


_STOPWORDS = {
    "the",
    "and",
    "for",
    "but",
    "with",
    "a",
    "an",
    "to",
    "of",
    "is",
    "are",
    "in",
    "on",
    "it",
    "its",
    "this",
    "that",
    "be",
    "can",
    "get",
    "very",
    "overall",
    "slightly",
    "bit",
    "good",
    "great",
    "excellent",
    "perfect",
    "ideal",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    toks = [
        t.strip("-")
        for t in re.split(r"\s+", text)
        if t and t not in _STOPWORDS and len(t) >= 3
    ]
    return toks


def summarize_reviews(product_name: str, top_n_themes: int = 5) -> dict:
    """
    Summarize reviews into:
    - avg_rating
    - sentiment_counts
    - top_themes
    - highlights (2 short quotes)
    - cautions (2 short quotes)
    """
    reviews_map = load_reviews()
    reviews = reviews_map.get(product_name, [])
    if not reviews:
        return {
            "product": product_name,
            "available": False,
            "avg_rating": None,
            "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0},
            "top_themes": [],
            "highlights": [],
            "cautions": [],
        }

    ratings = [int(r.get("rating", 0) or 0) for r in reviews]
    avg_rating = round(sum(ratings) / max(len(ratings), 1), 2)

    sentiments = [str(r.get("sentiment", "neutral")).lower() for r in reviews]
    sentiment_counts = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative"),
    }

    tokens: list[str] = []
    for r in reviews:
        tokens.extend(_tokenize(str(r.get("review", ""))))

    # Prefer meaningful single-word themes; collapse obvious variants
    normalize = {
        "battery": {"battery", "battery-life"},
        "portable": {"portable", "portability", "lightweight", "travel"},
        "performance": {"performance", "powerful", "responsive", "fast"},
        "fans": {"fans", "fan", "cooling", "noisy", "loud"},
        "price": {"price", "pricey", "affordable", "budget"},
        "keyboard": {"keyboard", "typing"},
        "display": {"display", "screen", "brighter", "vibrant"},
        "build": {"build", "quality", "premium", "durable", "design"},
        "gpu": {"gpu", "graphics"},
    }

    bucket_counts = Counter()
    raw_counts = Counter(tokens)
    for key, variants in normalize.items():
        bucket_counts[key] = sum(raw_counts[v] for v in variants if v in raw_counts)

    # Fill remaining slots with top raw tokens not captured
    captured = set().union(*normalize.values())
    leftovers = [(w, c) for w, c in raw_counts.most_common() if w not in captured]

    top_themes: list[str] = []
    for k, _ in bucket_counts.most_common():
        if bucket_counts[k] > 0:
            top_themes.append(k)
        if len(top_themes) >= top_n_themes:
            break
    for w, _ in leftovers:
        if len(top_themes) >= top_n_themes:
            break
        top_themes.append(w)

    # Pick highlights/cautions
    positives = [r for r in reviews if str(r.get("sentiment", "")).lower() == "positive"]
    neutrals = [r for r in reviews if str(r.get("sentiment", "")).lower() == "neutral"]
    negatives = [r for r in reviews if str(r.get("sentiment", "")).lower() == "negative"]

    highlights = [r.get("review") for r in (positives[:2] if positives else reviews[:2])]
    cautions_pool = negatives + neutrals
    cautions = [r.get("review") for r in (cautions_pool[:2] if cautions_pool else [])]

    return {
        "product": product_name,
        "available": True,
        "avg_rating": avg_rating,
        "sentiment_counts": sentiment_counts,
        "top_themes": top_themes,
        "highlights": [h for h in highlights if h],
        "cautions": [c for c in cautions if c],
        "review_count": len(reviews),
    }


def sales_insights_for_products(product_names: list[str]) -> dict[str, dict]:
    """Return insights per product name."""
    out: dict[str, dict] = {}
    for name in product_names:
        out[name] = summarize_reviews(name)
    return out

