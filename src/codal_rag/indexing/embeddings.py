"""Shared multilingual embedding adapter and an explicitly labeled smoke backend."""
from __future__ import annotations

import hashlib
import re
import numpy as np

from codal_rag.config import AppConfig


class EmbeddingModel:
    def __init__(self, model_name: str, batch_size: int = 16,
                 normalize_embeddings: bool = True, *, query_prefix: str = "query: ",
                 passage_prefix: str = "passage: ", revision: str | None = None,
                 local_files_only: bool = False, model=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.revision = revision
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install codal-rag[models] for multilingual embeddings") from exc
            model = SentenceTransformer(model_name, revision=revision,
                                        local_files_only=local_files_only)
        self.model = model
        self.dimension = model.get_sentence_embedding_dimension()
        self.max_seq_length = model.max_seq_length

    def prepare_passage(self, text: str) -> str:
        return self.passage_prefix + text.strip()

    def prepare_query(self, query: str) -> str:
        return self.query_prefix + query.strip()

    def count_model_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def _check_length(self, texts: list[str]) -> None:
        for i, text in enumerate(texts):
            if self.count_model_tokens(text) > self.max_seq_length:
                raise ValueError(f"Embedding input {i} exceeds model context ({self.max_seq_length}); "
                                 "reduce chunking.max_tokens or shorten the query. No truncation applied.")

    def validate_passages(self, texts: list[str]) -> None:
        self._check_length([self.prepare_passage(t) for t in texts])

    def _encode(self, texts: list[str], show_progress_bar: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._check_length(texts)
        # Empty prompt prevents a model's saved default prompt from duplicating prefixes.
        vectors = self.model.encode(texts, batch_size=self.batch_size,
            show_progress_bar=show_progress_bar, convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings, prompt="")
        result = np.ascontiguousarray(vectors, dtype=np.float32)
        if result.shape != (len(texts), self.dimension) or not np.isfinite(result).all():
            raise ValueError("Embedding model returned invalid dimensions or non-finite vectors")
        return result

    def encode_passages(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return self._encode([self.prepare_passage(t) for t in texts], show_progress_bar)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([self.prepare_query(query)], False)


class HashEmbeddingModel:
    """Deterministic bag-of-words smoke fixture. Scores are NOT semantic benchmarks."""
    def __init__(self, dimension: int = 128, normalize: bool = True):
        self.model_name = "smoke-hash-v1"
        self.dimension = dimension
        self.normalize_embeddings = normalize

    def encode_passages(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % self.dimension
                vectors[row, bucket] += 1
        if self.normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.where(norms == 0, 1, norms)
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_passages([query])


def create_embedding_model(config: AppConfig):
    e = config.embedding
    if e.backend == "hash":
        return HashEmbeddingModel(e.hash_dimension, e.normalize)
    return EmbeddingModel(e.model, e.batch_size, e.normalize, query_prefix=e.query_prefix,
                          passage_prefix=e.passage_prefix, revision=e.revision,
                          local_files_only=e.local_files_only)


def embedding_signature(config: AppConfig, dimension: int) -> dict:
    e = config.embedding
    return {"embedding_model": e.model, "backend": e.backend, "revision": e.revision,
            "embedding_dimension": dimension, "normalize_embeddings": e.normalize,
            "query_prefix": e.query_prefix, "passage_prefix": e.passage_prefix,
            "similarity": e.similarity, "faiss_index": "IndexFlatIP + IndexIDMap2"}
