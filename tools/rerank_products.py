"""Reranking tool using NVIDIA NIM.

Uses `nvidia/nv-rerankqa-mistral-4b-v3`
via the OpenAI Python client to rerank products by relevance to a query.
Falls back to existing ordering if the client is not available.
"""

from __future__ import annotations

import os
from typing import List, Dict

from openai import OpenAI


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
_client: OpenAI | None = None
if NVIDIA_API_KEY:
    _client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")


def rerank_products(query: str, products: List[Dict]) -> List[Dict]:
    """Return products reranked by relevance to the query."""
    if _client is None or not products:
        return products

    try:
        docs = [
            f"{p.get('name','')} {p.get('description','')} {p.get('use_case','')}"
            for p in products
        ]
        # Use OpenAI's rerank interface if available; fallback to chat-based scoring otherwise.
        if hasattr(_client, "responses"):
            # Newer OpenAI clients expose rerank through the responses API.
            resp = _client.responses.create(
                model="nvidia/nv-rerankqa-mistral-4b-v3",
                input=query,
                documents=docs,
            )
            # Expect rankings in `output[0].ranking`
            ranking = getattr(resp.output[0], "ranking", [])
            order = [r.index for r in ranking if 0 <= r.index < len(products)]
        else:
            # Fallback: score each doc via chat completions
            scores = []
            for i, d in enumerate(docs):
                r = _client.chat.completions.create(
                    model="nvidia/nv-rerankqa-mistral-4b-v3",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Query: {query}\n\nDocument: {d}\n\nGive a relevance score from 0 to 10.",
                        }
                    ],
                    temperature=0,
                )
                text = (r.choices[0].message.content or "").strip()
                try:
                    score = float(text.split()[0])
                except Exception:
                    score = 0.0
                scores.append((score, i))
            scores.sort(reverse=True)
            order = [i for _, i in scores]

        return [products[i] for i in order]
    except Exception:
        return products

