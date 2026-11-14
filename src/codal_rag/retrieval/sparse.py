"""Dependency-free BM25 and rank-only fusion; scores never imply entailment."""
from __future__ import annotations

from collections import Counter
import math
import re

from codal_rag.normalization.persian import NormalizerConfig, normalize_plain_text


def terms(text: str) -> list[str]:
    # A ZWNJ is a word boundary for both Arabic/Persian spelling variants.
    return re.findall(r"\w+", normalize_plain_text(text, NormalizerConfig()).replace("\u200c", " ").lower())


def identity(hit: dict) -> tuple[str, str]:
    c = hit["chunk"]
    return c["document_id"], c["chunk_id"]


class BM25:
    def __init__(self, metadata: dict[str, dict], k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.metadata, self.k1, self.b = metadata, k1, b
        self.frequencies = {key: Counter(terms(c.get("embedding_text") or
                            c.get("retrieval_text") or c.get("text") or c.get("content") or ""))
                            for key, c in metadata.items()}
        self.lengths = {key: sum(tf.values()) for key, tf in self.frequencies.items()}
        self.average = sum(self.lengths.values()) / max(len(metadata), 1)
        df = Counter(t for tf in self.frequencies.values() for t in tf)
        self.idf = {t: math.log(1 + (len(metadata) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query: str, top_k: int, allowed_ids: set[str] | None = None) -> list[dict]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be positive")
        q = set(terms(query))
        scored = []
        for key, tf in self.frequencies.items():
            if allowed_ids is not None and key not in allowed_ids:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.lengths[key] / (self.average or 1))
            score = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                        for t in q if tf[t])
            if score > 0:
                scored.append({"score": score, "vector_id": int(key), "chunk": self.metadata[key]})
        scored.sort(key=lambda h: (-h["score"], identity(h)))
        return [{**h, "rank": i} for i, h in enumerate(scored[:top_k], 1)]


def reciprocal_rank_fusion(rankings: dict[str, list[dict]], top_k: int, k: int = 60) -> list[dict]:
    if k <= 0 or top_k <= 0:
        raise ValueError("RRF parameters must be positive")
    fused = {}
    for method, hits in rankings.items():
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = identity(hit)
            if key in seen:
                continue
            seen.add(key)
            row = fused.setdefault(key, {**hit, "score": 0.0, "retrieval_ranks": {}})
            row["score"] += 1.0 / (k + rank)
            row["retrieval_ranks"][method] = rank
    ordered = sorted(fused.values(), key=lambda h: (-h["score"], identity(h)))
    return [{**h, "rank": i} for i, h in enumerate(ordered[:top_k], 1)]
