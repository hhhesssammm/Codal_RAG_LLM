"""One normalized query and one vector shared by both experimental methods."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from codal_rag.config import AppConfig, load_config
from codal_rag.indexing.embeddings import create_embedding_model, embedding_signature
from codal_rag.indexing.faiss_store import FaissVectorStore
from codal_rag.models import DualRetrievalResult
from codal_rag.normalization.persian import NormalizerConfig, normalize_plain_text
from codal_rag.storage.json_io import load_json, save_json
from codal_rag.retrieval.sparse import BM25, reciprocal_rank_fusion
from codal_rag.retrieval.scope import RetrievalScope, canonical


class DualRetrievalEngine:
    def __init__(self, baseline_db_dir: Path | None = None, proposed_db_dir: Path | None = None,
                 *, config: AppConfig | None = None, embedding_model=None):
        self.config = config or load_config()
        baseline_db_dir = baseline_db_dir or self.config.path("vector_db", "baseline")
        proposed_db_dir = proposed_db_dir or self.config.path("vector_db", "proposed")
        # Check files/configuration before loading expensive model weights.
        baseline = self.read_index_config(baseline_db_dir)
        proposed = self.read_index_config(proposed_db_dir)
        for manifest in (baseline, proposed):
            signature = embedding_signature(self.config, manifest.get("embedding_dimension", 0))
            for key, value in signature.items():
                if key not in manifest or manifest[key] != value:
                    raise ValueError(f"Index/config mismatch for {key}; rebuild both indexes")
        if baseline.get("embedding_dimension") != proposed.get("embedding_dimension"):
            raise ValueError("Baseline and Proposed dimensions differ")
        if not baseline.get("documents") or baseline.get("documents") != proposed.get("documents"):
            raise ValueError("Baseline and Proposed document corpora differ; rebuild both indexes")
        if not baseline.get('source_corpus_sha256') or baseline.get('source_corpus_sha256') != proposed.get('source_corpus_sha256'):
            raise ValueError('Baseline and Proposed source contents differ; rebuild both indexes')
        self.baseline_store = FaissVectorStore.load(baseline_db_dir)
        self.proposed_store = FaissVectorStore.load(proposed_db_dir)
        self.embedding_model = embedding_model or create_embedding_model(self.config)
        if self.embedding_model.model_name != self.config.embedding.model:
            raise ValueError("Retrieval model does not match configuration")
        for store in (self.baseline_store, self.proposed_store):
            if store.dimension != self.embedding_model.dimension or store.dimension != baseline['embedding_dimension']:
                raise ValueError("Index dimension does not match the embedding model; rebuild")
        self.index_config = {"baseline": baseline, "proposed": proposed}
        self._sparse = {}

    def catalog(self, method: str = "proposed") -> list[dict]:
        return list(self._store(method).metadata.values())

    def _store(self, method):
        if method not in {"baseline", "proposed"}:
            raise ValueError("Method must be baseline or proposed")
        return getattr(self, f"{method}_store")

    def _search_method(self, query, vector, method, top_k, scope=None, section_hints=()):
        store = self._store(method)
        allowed = None if scope is None else {i for i, c in store.metadata.items() if scope.matches(c)}
        # Hints choose actual matching headings, with a recorded global-in-scope
        # fallback when no heading matches. Identity constraints are never relaxed.
        if section_hints:
            sections = {i for i, c in store.metadata.items()
                        if (allowed is None or i in allowed) and any(
                            canonical(h) in canonical(" ".join(c.get("heading_path") or []))
                            for h in section_hints if canonical(h))}
            if sections:
                allowed = sections
        r = self.config.retrieval
        depth = top_k if r.mode == "dense" else max(top_k, r.candidate_k)
        dense = store.search(vector, depth, allowed_ids=allowed) if r.mode != "bm25" else []
        if r.mode == "dense":
            return dense
        if method not in self._sparse:
            self._sparse[method] = BM25(store.metadata, r.bm25_k1, r.bm25_b)
        sparse = self._sparse[method].search(query, depth, allowed)
        if r.mode == "bm25":
            return sparse[:top_k]
        return reciprocal_rank_fusion({"dense": dense, "bm25": sparse}, top_k, r.rrf_k)

    def search_method(self, query: str, method: str, top_k: int | None = None, *,
                      scope: RetrievalScope | None = None, section_hints=()) -> list[dict]:
        top_k = self.config.top_k if top_k is None else top_k
        if type(top_k) is not int or top_k <= 0 or not isinstance(query, str) or not query.strip():
            raise ValueError("A nonempty query and positive top_k are required")
        normalized = self.normalize_query(query)
        if not normalized.strip():
            raise ValueError('Query is empty after normalization')
        vector = None if self.config.retrieval.mode == "bm25" else self.embedding_model.encode_query(normalized)
        return self._search_method(normalized, vector, method, top_k, scope, section_hints)

    @staticmethod
    def read_index_config(directory: Path) -> dict:
        path = Path(directory) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing index config: {path}; run codal-rag index with mode both")
        result = load_json(path)
        if not isinstance(result, dict):
            raise ValueError(f"Invalid index configuration: {path}")
        return result

    @staticmethod
    def normalize_query(query: str) -> str:
        return normalize_plain_text(query, NormalizerConfig())

    def search(self, query: str, top_k: int | None = None, *,
               scope: RetrievalScope | None = None) -> DualRetrievalResult:
        top_k = self.config.top_k if top_k is None else top_k
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")
        normalized = self.normalize_query(query)
        if not normalized.strip():
            raise ValueError("Query is empty after normalization")
        vector = None if self.config.retrieval.mode == "bm25" else self.embedding_model.encode_query(normalized)
        result = DualRetrievalResult(query, normalized, top_k,
            self._search_method(normalized, vector, "baseline", top_k, scope),
            self._search_method(normalized, vector, "proposed", top_k, scope))
        if scope is not None:
            result.scope = scope.to_dict()
        return result

    def result_payload(self, result: DualRetrievalResult) -> dict:
        payload = {**asdict(result), "schema_version": 1,
                "embedding_model": self.config.embedding.model,
                "index_config": self.index_config}
        if self.config.retrieval.mode != "dense" or hasattr(result, "scope"):
            payload.update(schema_version=2, retrieval_configuration=asdict(self.config.retrieval),
                           scope=getattr(result, "scope", None))
        return payload

    def save_result(self, result: DualRetrievalResult, path: str | Path | None = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return save_json(self.result_payload(result), path or self.config.path("retrieval_results", f"retrieval_{stamp}.json"))
