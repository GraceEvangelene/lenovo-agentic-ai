"""Semantic product search using SentenceTransformers and FAISS."""

import json
from pathlib import Path

# Lazy imports to avoid loading heavy models at import time
_index = None
_products = None
_model = None


def _get_data_path():
    """Resolve path to laptops.json relative to project root."""
    return Path(__file__).resolve().parent.parent / "data" / "laptops.json"


def _load_products():
    """Load laptop data from JSON."""
    global _products
    if _products is None:
        with open(_get_data_path(), "r") as f:
            _products = json.load(f)
    return _products


def build_index():
    """Build FAISS index from laptop descriptions and use_case for semantic search."""
    global _index, _products, _model
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError:
        return None

    if _index is not None:
        return _index

    try:
        _products = _load_products()
        if _model is None:
            # This may trigger a model download; if networking is restricted, fall back.
            _model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [
            f"{p.get('name', '')} {p.get('description', '')} {p.get('use_case', '')}"
            for p in _products
        ]
        embeddings = _model.encode(texts)
        dimension = embeddings.shape[1]
        _index = faiss.IndexFlatL2(dimension)
        _index.add(embeddings.astype("float32"))
        return _index
    except Exception:
        # Any failure (model download blocked, etc.) -> disable semantic search
        return None


def search_products(query: str, top_k: int = 5, use_semantic: bool = True) -> list[dict]:
    """
    Return laptops relevant to the query (use case / description).
    Falls back to keyword match if SentenceTransformers/FAISS not available.
    """
    products = _load_products()

    if use_semantic:
        try:
            idx = build_index()
            if idx is None:
                use_semantic = False
        except Exception:
            use_semantic = False

    if use_semantic and _index is not None and _model is not None:
        q_emb = _model.encode([query])
        _, indices = _index.search(q_emb.astype("float32"), min(top_k, len(products)))
        return [products[i] for i in indices[0] if i < len(products)]

    # Keyword fallback: match query words in use_case and description
    query_lower = query.lower()
    words = set(query_lower.split())
    scored = []
    for p in products:
        text = f"{p.get('use_case', '')} {p.get('description', '')} {p.get('name', '')}".lower()
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]] if scored else products[:top_k]
