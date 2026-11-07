from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from codal_rag.chunking.baseline import fixed_size_chunking
from codal_rag.chunking.structure import StructureChunkerConfig, structure_aware_chunking
from codal_rag.chunking.tokenization import create_tokenizer
from codal_rag.config import AppConfig
from codal_rag.models import DocumentMetadata
from codal_rag.normalization.persian import normalize_markdown, assert_markdown_structure_preserved
from codal_rag.pipelines.normalize import markdown_files
from codal_rag.storage.json_io import save_json, digest


def process_document(path: Path, tokenizer, config: AppConfig, *,
                     metadata: DocumentMetadata | None = None) -> dict[str, list[dict]]:
    metadata = metadata or DocumentMetadata(document_id=path.stem, source_file=path.name)
    original = path.read_text(encoding="utf-8")
    markdown = normalize_markdown(original)
    assert_markdown_structure_preserved(original, markdown)
    result = {}
    c = config.chunking
    if config.mode in {"baseline", "both"}:
        chunks = fixed_size_chunking(markdown, metadata.document_id, tokenizer,
                                     chunk_size=c.max_tokens, overlap=c.baseline_overlap)
        records = []
        for chunk in chunks:
            record = chunk.to_dict()
            record.update(asdict(metadata))
            records.append(record)
        result["baseline"] = records
    if config.mode in {"proposed", "both"}:
        structure_config = StructureChunkerConfig(
            max_tokens=c.max_tokens, min_merge_tokens=c.min_merge_tokens,
            include_path_in_retrieval_text=c.include_path)
        result["proposed"] = [chunk.to_dict() for chunk in structure_aware_chunking(
            markdown, metadata, tokenizer, structure_config)]
    return result


def chunk_documents(input_path: str | Path, config: AppConfig, *, tokenizer=None,
                    metadata_by_id: dict[str, dict] | None = None) -> dict[str, list[dict]]:
    """Normalize and chunk once; relative paths are stable corpus document IDs.

    Replace each method's corpus.json so removed source documents cannot silently
    remain in later indexes. Indexing reads this snapshot, not stale per-file JSONs.
    """
    if metadata_by_id is not None and not isinstance(metadata_by_id, dict):
        raise ValueError("Document metadata must be an object keyed by document ID")
    source = Path(input_path)
    files = markdown_files(source)
    root = source if source.is_dir() else source.parent
    tokenizer = tokenizer or create_tokenizer(config)
    corpus: dict[str, list[dict]] = {}
    source_hashes = {}
    for path in files:
        relative = path.relative_to(root)
        document_id = relative.with_suffix("").as_posix()
        supplied = (metadata_by_id or {}).get(document_id, {})
        if not isinstance(supplied, dict) or set(supplied) - {"company", "year", "document_type"}:
            raise ValueError(f"Invalid metadata for {document_id}; use company, year, document_type")
        if any(value is not None and not isinstance(value, str) for value in supplied.values()):
            raise ValueError(f"Metadata values for {document_id} must be strings or null")
        metadata = DocumentMetadata(**{**supplied, "document_id": document_id,
                                       "source_file": relative.as_posix()})
        source_hashes[document_id] = digest(path.read_text(encoding="utf-8"))
        for method, records in process_document(path, tokenizer, config, metadata=metadata).items():
            corpus.setdefault(method, []).extend(records)
    for method, records in corpus.items():
        if not records:
            raise ValueError(f"No {method} chunks produced; documents may be empty")
        save_json(records, config.path("chunks", method, "corpus.json"))
        save_json({"source_sha256": digest(source_hashes), "documents": source_hashes,
                   "chunks_sha256": digest(records)}, config.path("chunks", method, "manifest.json"))
    return corpus
