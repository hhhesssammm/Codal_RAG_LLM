# Data contracts

All JSON is UTF-8, Unicode-preserving, and rejects non-finite values on save. Output schema version is `1` where a wrapper/manifest exists. Canonical dataclasses are in `models.py`.

## Documents and chunks

Input is one `.md` document or a directory recursively containing `.md` files. The input root defines document identity: `alpha/report.md` becomes `alpha/report`. Source paths are relative to that root. Headings are attached to following blocks rather than made standalone chunks.

`data/chunks/{baseline,proposed}/corpus.json` is an array. Both records include `chunk_id`, `document_id`, `pipeline`, `chunk_type`, `token_count`, `heading_path`, optional company/year/document_type and source_file.

| Method | Additional fields / meaning |
| --- | --- |
| Baseline | `content`, `block_index`; IDs retain `document_id_baseline_0000`; token_count counts body tokens |
| Proposed | `text`, `retrieval_text`, `section`, `subsection`, `source_block_ids`, `table_part`, `table_parts`; IDs retain `document_id_S00000`; token_count includes heading context |

Proposed chunk pipeline remains `structure_aware` in the chunk snapshot; indexed records are labeled `proposed`, matching the experiment method. `retrieval_text` includes the heading path once. Baseline embedding uses content without structural metadata. Chunk manifests contain per-document source hashes and the whole source/corpus digest.

Optional `--metadata metadata.json`:

```json
{"alpha/report": {"company": "شرکت الف", "year": "1403", "document_type": "گزارش مدیریت"}}
```

## Embeddings and indexes

Each `data/embeddings/{method}` contains `vectors.npy` (finite float32 matrix, rows in record order), `records.json`, and `manifest.json`. No pickle is used. The manifest contains settings, corpus/source hashes, vector-file digest, and record digest.

Each `data/vector_db/{method}` contains `index.faiss`, `metadata.json` (numeric vector IDs as string keys), `db_info.json`, and `config.json` (embedding manifest). Dimensions, exact index type/metric, ID sets, next_id and artifact digests must agree. Load only indexes you created/trust; checksums detect mismatched files, not the trustworthiness of a third-party native FAISS binary. Legacy indexes must be rebuilt; never mix old and new artifact files.

## Retrieval

The root contains `original_query`, `normalized_query`, `top_k`, `baseline`, `proposed`, embedding model, schema version, and both index manifests. Each hit has `rank`, `score`, `vector_id`, and the entire `chunk`. Rank starts at 1. Large k returns all available vectors. Empty queries or nonpositive k are errors.

Legacy adapters accept `baseline_results`, `proposed_results`, and nested `results`, `hits`, `items`, or `retrieval_results` lists. New writers emit only the canonical envelope.

## Extraction and citations

Extraction root: `query`, `method`, `evidence`, `extraction`, `citation_view`, `citation_validation`, `llm`, `schema_version`. The provider URL and key are never saved. A supplied mock client is explicitly labeled.

Evidence IDs (`E1`, `E2`, ...) are local to one method and question, following retrieval order. Each includes rank, chunk/document ID, source text, structural metadata, and score. Do not interpret `E1` as the same chunk across questions or methods.

`extraction` contains `query`, `explicit`, `implicit`, `latent`, boolean `insufficient_evidence`, and `notes`. Every claim is an object with `claim`, `evidence_ids`, `basis`, and confidence in [0,1]. Latent claims also require a disclaimer containing `استنباط` and at least two distinct supplied IDs. Empty outputs must report insufficient evidence. Unknown IDs, malformed JSON, incomplete outputs, missing source identity and query mismatches are errors.

`citation_view.items` contains finding ID, level, statement, confidence, and resolved evidence payloads linking finding → evidence → section → document. Missing citations and unresolved identities fail structural validation; empty Proposed heading paths remain warnings because a document can lack headings. No validator establishes factual entailment.

## Evaluation

Questions are a JSON array of `{id, question, type, expected_answer, gold_evidence, notes?}`. IDs contain 1–100 ASCII letters, digits, `_`, `-`; they must be unique. Type is explicit/implicit/latent. Expected answer and question must be nonempty. Each gold item has a document ID and a specific supporting passage. An empty gold list means retrieval is unscored; template placeholders must be annotated before use.

A batch writes `run.json` with configuration, dataset digest, question IDs, and completion status, then `{id}/gold.json`, `retrieval.json`, `baseline.json`, `proposed.json`. Only manifest-listed questions are evaluated, avoiding leftover directories from previous runs. Legacy Demo directories without a manifest are supported if all four files are present and queries/provenance agree.

A report contains `configuration`, `run_configuration`, `semantic_evaluator`, `metric_notes`, `overall`, `by_type`, `per_question`, and `artifact_sha256`. `null` means unavailable/unannotated, not a zero score. Repeated evaluation of unchanged artifacts/settings produces identical report bytes. Keep each real experiment in a separate run directory; rerunning a batch intentionally replaces the selected run's artifacts.
