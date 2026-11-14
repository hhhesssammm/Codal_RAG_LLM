"""Evidence packing and immutable, source-derived v2 provenance."""
from __future__ import annotations

from copy import deepcopy
import re

from codal_rag.extraction.evidence import build_evidence_context, retrieval_to_evidence
from codal_rag.retrieval.scope import canonical
from codal_rag.retrieval.sparse import identity


def token_upper_bound(text: str) -> int:
    """Conservative byte-token budget, NOT provider billing tokens.

    A generation-tokenizer counter can be injected instead. UTF-8 byte length
    bounds ordinary byte/subword tokenizers without downloading another model.
    """
    return len(text.encode("utf-8"))


def content(hit: dict) -> str:
    return (hit["chunk"].get("text") or hit["chunk"].get("content") or "").strip()


def redundant_text(a: str, b: str) -> bool:
    a, b = canonical(a), canonical(b)
    if not a or not b:
        return False
    # Token boundaries retain financial signs and prevent 10 matching 100 or -10.
    def token_string(s):
        return ' ' + ' '.join(re.findall(r'[+-]?\d[\d,.]*|\w+|[^\w\s]', s)) + ' '
    ta, tb = token_string(a), token_string(b)
    if ta == tb or ta in tb or tb in ta:
        return True
    # Do not collapse different numbers/signs, even in otherwise identical prose.
    if re.findall(r"[-+]?\d[\d,.]*", a) != re.findall(r"[-+]?\d[\d,.]*", b):
        return False
    def shingles(s):
        words = s.split()
        return {tuple(words[i:i+4]) for i in range(max(len(words)-3, 0))}
    left, right = shingles(a), shingles(b)
    return bool(left and right) and len(left & right) / len(left | right) >= 0.90


def merge_hits(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], list[dict]]:
    merged, dropped = deepcopy(existing), []
    for hit in incoming:
        if not isinstance(hit, dict) or not isinstance(hit.get("chunk"), dict):
            raise ValueError("Malformed retrieval candidate")
        chunk = hit["chunk"]
        if not chunk.get("document_id") or not chunk.get("chunk_id") or not content(hit):
            raise ValueError("Evidence requires source identity and nonempty text")
        old = next((h for h in merged if identity(h) == identity(hit)), None)
        if old and old["chunk"] != chunk:
            raise ValueError("A source chunk changed during answering")
        duplicate = old or next((h for h in merged if
            h["chunk"]["document_id"] == chunk["document_id"] and
            redundant_text(content(h), content(hit)) and
            len(canonical(content(hit))) <= len(canonical(content(h)))), None)
        if duplicate:
            dropped.append({"chunk_id": chunk["chunk_id"], "duplicate_of": identity(duplicate)[1]})
        else:
            merged.append(deepcopy(hit))
    return merged, dropped


def pack_hits(hits: list[dict], budget: int, counter=token_upper_bound):
    """Never truncate a table or invent a source span. Skip whole oversized units."""
    selected, excluded = [], []
    for hit in hits:
        proposed = selected + [hit]
        size = counter(build_evidence_context(retrieval_to_evidence(proposed), max_chars_per_evidence=None))
        if size <= budget:
            selected = proposed
        else:
            excluded.append(identity(hit)[1])
    evidence = retrieval_to_evidence(selected)
    return selected, evidence, excluded, counter(build_evidence_context(evidence, max_chars_per_evidence=None)) if evidence else 0


def trace_sources(evidence, hits):
    index = {identity(h): h["chunk"] for h in hits}
    fields = ("document_id", "chunk_id", "source_file", "source_url", "company", "symbol",
              "year", "financial_period", "publication_date", "document_type", "heading_path",
              "section", "subsection", "source_block_ids", "table_part", "table_parts",
              "table_title", "table_headers", "source_locations", "source_units", "page", "row", "cell")
    return {e.evidence_id: {k: deepcopy(c[k]) for k in fields if k in c and c[k] is not None}
            for e in evidence for c in [index[(e.document_id, e.chunk_id)]]}
