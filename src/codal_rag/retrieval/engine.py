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

    def search(self, query: str, top_k: int | None = None) -> DualRetrievalResult:
        top_k = self.config.top_k if top_k is None else top_k
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")
        normalized = self.normalize_query(query)
        if not normalized.strip():
            raise ValueError("Query is empty after normalization")
        vector = self.embedding_model.encode_query(normalized)
        return DualRetrievalResult(query, normalized, top_k,
            self.baseline_store.search(vector, top_k), self.proposed_store.search(vector, top_k))

    def result_payload(self, result: DualRetrievalResult) -> dict:
        return {**asdict(result), "schema_version": 1,
                "embedding_model": self.config.embedding.model,
                "index_config": self.index_config}

    def save_result(self, result: DualRetrievalResult, path: str | Path | None = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        return save_json(self.result_payload(result), path or self.config.path("retrieval_results", f"retrieval_{stamp}.json"))
