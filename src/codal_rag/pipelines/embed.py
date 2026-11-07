from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from codal_rag.indexing.embeddings import create_embedding_model, embedding_signature
from codal_rag.indexing.representations import get_baseline_embedding_text, get_proposed_embedding_text
from codal_rag.storage.json_io import digest, load_json, save_json


def methods(config) -> tuple[str, ...]:
    return ("baseline", "proposed") if config.mode == "both" else (config.mode,)


def load_chunks(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "corpus.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing corpus snapshot: {path}; run codal-rag chunk")
    records = load_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a nonempty chunk list: {path}")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or not record.get("chunk_id") or not record.get("document_id"):
            raise ValueError("Each chunk must have chunk_id and document_id")
        if record["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk ID: {record['chunk_id']}")
        seen.add(record["chunk_id"])
    return records


def embed_corpus(config, *, embedding_model=None) -> dict:
    model = embedding_model or create_embedding_model(config)
    if model.model_name != config.embedding.model:
        raise ValueError("Injected embedding model does not match configured model")
    output = {}
    for method in methods(config):
        chunks = load_chunks(config.path("chunks", method))
        corpus_manifest = load_json(config.path("chunks", method, "manifest.json"))
        if corpus_manifest.get("chunks_sha256") != digest(chunks):
            raise ValueError("Chunk snapshot/manifest mismatch; rerun chunk")
        text_fn = get_proposed_embedding_text if method == "proposed" else get_baseline_embedding_text
        texts = [text_fn(chunk) for chunk in chunks]
        vectors = np.ascontiguousarray(model.encode_passages(texts), dtype=np.float32)
        if vectors.shape != (len(chunks), model.dimension) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding matrix")
        if config.embedding.normalize and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Expected unit-normalized nonzero embeddings")
        records = [{**chunk, "embedding_text": text, "embedding_model": model.model_name,
                    "pipeline": method} for chunk, text in zip(chunks, texts)]
        directory = config.path("embeddings", method)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", vectors, allow_pickle=False)
        save_json(records, directory / "records.json")
        manifest = {**embedding_signature(config, model.dimension), "schema_version": 1,
                    "pipeline": method, "records_sha256": digest(records),
                    "source_corpus_sha256": corpus_manifest["source_sha256"],
                    "source_chunks_sha256": digest(chunks), "random_seed": config.random_seed,
                    "documents": sorted({c['document_id'] for c in chunks}),
                    "vectors_sha256": hashlib.sha256((directory / "vectors.npy").read_bytes()).hexdigest()}
        save_json(manifest, directory / "manifest.json")
        output[method] = {"vectors": vectors, "records": records, "manifest": manifest}
    return output


def load_embeddings(config, method: str) -> dict:
    directory = config.path("embeddings", method)
    records = load_json(directory / "records.json")
    manifest = load_json(directory / "manifest.json")
    expected = embedding_signature(config, manifest.get("embedding_dimension", 0))
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Embedding cache/config mismatch; rerun codal-rag embed")
    chunks = load_chunks(config.path("chunks", method))
    if (digest(records) != manifest.get("records_sha256") or
        digest(chunks) != manifest.get("source_chunks_sha256") or
        hashlib.sha256((directory / "vectors.npy").read_bytes()).hexdigest() != manifest.get("vectors_sha256")):
        raise ValueError("Stale or corrupted embedding cache; rerun codal-rag embed")
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    return {"vectors": vectors, "records": records, "manifest": manifest}
