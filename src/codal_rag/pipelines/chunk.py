from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from codal_rag.chunking.baseline import fixed_size_chunking
from codal_rag.chunking.structure import StructureChunkerConfig, structure_aware_chunking
from codal_rag.chunking.tokenization import create_tokenizer
from codal_rag.config import AppConfig
from codal_rag.models import DocumentMetadata
from codal_rag.extraction.source_units import annotate_units
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
    for method, records in result.items():
        annotate_units(markdown, records, method)
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
        extra_fields = {"symbol", "financial_period", "publication_date", "source_url", "block_locations"}
        if not isinstance(supplied, dict) or set(supplied) - {"company", "year", "document_type"} - extra_fields:
            raise ValueError(f"Invalid document metadata for {document_id}")
        if any(value is not None and not isinstance(value, str) for key, value in supplied.items() if key != 'block_locations'):
            raise ValueError(f"Metadata values for {document_id} must be strings or null")
        locations = supplied.get('block_locations', {})
        if not isinstance(locations, dict) or any(not str(k).isdigit() or not isinstance(v, dict)
                                                for k, v in locations.items()):
            raise ValueError('block_locations must map parser block IDs to original source coordinates')
        metadata = DocumentMetadata(**{**{k: v for k, v in supplied.items() if k not in extra_fields}, "document_id": document_id,
                                       "source_file": relative.as_posix()})
        source_hashes[document_id] = digest(path.read_text(encoding="utf-8"))
        for method, records in process_document(path, tokenizer, config, metadata=metadata).items():
            for record in records:
                record.update({k: v for k, v in supplied.items() if k in extra_fields and k != 'block_locations'})
                known = [{**locations[str(b)], 'block_id': b} for b in record.get('source_block_ids', []) if str(b) in locations]
                if known:
                    record['source_locations'] = known
                for unit in record.get('source_units', []):
                    if str(unit['source_block_id']) in locations:
                        unit['source_location'] = locations[str(unit['source_block_id'])].copy()
            corpus.setdefault(method, []).extend(records)
    for method, records in corpus.items():
        if not records:
            raise ValueError(f"No {method} chunks produced; documents may be empty")
        save_json(records, config.path("chunks", method, "corpus.json"))
        save_json({"source_sha256": digest(source_hashes), "documents": source_hashes,
                   "chunks_sha256": digest(records)}, config.path("chunks", method, "manifest.json"))
    return corpus
