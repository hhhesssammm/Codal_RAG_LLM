"""Exact FAISS inner-product search with validated vector/record associations."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import faiss
import numpy as np

from codal_rag.storage.json_io import digest, load_json, save_json


class FaissVectorStore:
    def __init__(self, dimension: int):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")
        self.dimension = dimension
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self.metadata: dict[str, dict] = {}
        self.next_id = 0

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def _matrix(self, values, *, query=False) -> np.ndarray:
        result = np.ascontiguousarray(values, dtype=np.float32)
        if query and result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2 or result.shape[1] != self.dimension:
            raise ValueError(f"Expected a 2D embedding matrix with dimension {self.dimension}")
        if query and result.shape[0] != 1:
            raise ValueError("Search accepts exactly one query vector")
        if not np.isfinite(result).all():
            raise ValueError("Embedding vectors must be finite")
        return result

    def add(self, embeddings: np.ndarray, records: list[dict]) -> None:
        vectors = self._matrix(embeddings)
        if len(vectors) != len(records) or any(not isinstance(r, dict) for r in records):
            raise ValueError("Embedding and metadata record counts/types must match")
        ids = np.arange(self.next_id, self.next_id + len(records), dtype=np.int64)
        self.index.add_with_ids(vectors, ids)
        self.metadata.update({str(int(i)): dict(record) for i, record in zip(ids, records)})
        self.next_id += len(records)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query = self._matrix(query_embedding, query=True)
        if not self.size:
            return []
        scores, ids = self.index.search(query, min(top_k, self.size))
        results = []
        for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]), 1):
            key = str(int(vector_id))
            if key not in self.metadata:
                raise ValueError("FAISS vector has no metadata; rebuild the index")
            results.append({"rank": rank, "score": float(score), "vector_id": int(vector_id),
                            "chunk": self.metadata[key]})
        return results

    def save(self, directory: str | Path, config: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".faiss")
        os.close(fd)
        try:
            faiss.write_index(self.index, temporary)
            index_hash = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
            os.replace(temporary, directory / "index.faiss")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        save_json(self.metadata, directory / "metadata.json")
        # Always replace config, avoiding a stale manifest when called without one.
        save_json(config or {}, directory / "config.json")
        save_json({"schema_version": 1, "dimension": self.dimension, "size": self.size,
                   "next_id": self.next_id, "metadata_sha256": digest(self.metadata),
                   "config_sha256": digest(config or {}), "index_sha256": index_hash},
                  directory / "db_info.json")

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        directory = Path(directory)
        for name in ("index.faiss", "metadata.json", "db_info.json", "config.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Missing index artifact {directory / name}; run codal-rag index")
        info = load_json(directory / "db_info.json")
        metadata = load_json(directory / "metadata.json")
        config = load_json(directory / "config.json")
        if not isinstance(info, dict) or not isinstance(metadata, dict) or not isinstance(config, dict):
            raise ValueError("Index manifests and metadata must be JSON objects; rebuild the index")
        if info.get("schema_version") != 1:
            raise ValueError("Legacy/unversioned FAISS index: rebuild with codal-rag index")
        if any(type(info.get(key)) is not int for key in ("dimension", "size", "next_id")):
            raise ValueError("Invalid index dimensions/counts; rebuild the index")
        if info["dimension"] <= 0 or info["size"] < 0:
            raise ValueError("Invalid index dimensions/counts; rebuild the index")
        if (digest(metadata) != info.get("metadata_sha256") or
            digest(config) != info.get("config_sha256") or
            hashlib.sha256((directory / "index.faiss").read_bytes()).hexdigest() != info.get("index_sha256")):
            raise ValueError("Index artifacts are inconsistent or corrupted; rebuild the index")
        obj = cls(info["dimension"])
        try:
            obj.index = faiss.read_index(str(directory / "index.faiss"))
        except RuntimeError as exc:
            raise ValueError("Unreadable FAISS index; rebuild the index") from exc
        if not isinstance(obj.index, faiss.IndexIDMap2) or obj.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected IndexIDMap2 with inner-product search")
        ids = faiss.vector_to_array(obj.index.id_map).tolist()
        if (obj.index.d != obj.dimension or obj.size != info["size"] or
            len(set(ids)) != len(ids) or set(metadata) != {str(i) for i in ids} or
            any(not isinstance(record, dict) for record in metadata.values())):
            raise ValueError("Index dimensions, IDs, or metadata do not match; rebuild the index")
        obj.metadata = metadata
        obj.next_id = info["next_id"]
        if type(obj.next_id) is not int or obj.next_id <= max(ids, default=-1):
            raise ValueError("Invalid next vector ID")
        return obj
