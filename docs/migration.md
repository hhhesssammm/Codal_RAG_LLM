# Audit and migration report

## Audited revisions and scope

- Target `hhhesssammm/Codal_RAG_LLM`, `master` at `f9dc415ae278c03a20edde51ff7526081bade506` (`initial`). Its visible history also includes `df4cb77` (IDE directory deletion) and `1b53d8a` (initial layout).
- Read-only reference `hhhesssammm/Demo-RAG-LLM`, `master` at `208bb64bdea92f919008eb5bba2ce24588a92daf` (`section 9`), following sections 8, 5.2, 5.1 and 4.
- Changes belong exclusively to target branch `fix/complete-rag-migration`; `master` and the reference repository are not modified. No automatic merge.

The initial target had 3 pytest collection errors from obsolete imports. Packaging/configuration/docs were empty, so those failures were only the first visible blockers. The reference was inspected through a credential-free local snapshot; its configuration credential was neither reused nor copied into this project.

## Prioritized repair plan and disposition

| Priority | Verified finding | Classification and repair |
| --- | --- | --- |
| P0 | Empty pyproject, package entry point and five YAML configs; no installed command | Migration gap: typed centralized configuration, packaging/extras, CLI and `__main__` implemented |
| P0 | Flat `src.*` and `config.py` imports throughout package, apps and tests | Migration gap: canonical `codal_rag.*` imports; no path manipulation |
| P0 | Baseline imports missing `Chunk`; tokenizer default references missing `TOKENIZER_MODEL` | Also present in latest Demo: restore missing Baseline record; tokenizer follows configured embedding model |
| P0 | Chunk pipeline concatenates two scripts; shadows config, calls obsolete signature, processes one hardcoded company/report | Migration gap: reusable multi-document pipeline and stable relative IDs/metadata; no duplicate execution |
| P0 | Embed/index fragments lack imports/constants and duplicate representation/orchestration code | Migration gap: one representation module, shared embedding pipeline, validated caches and FAISS indexing |
| P0 | Split extractor references undefined SYSTEM_PROMPT | Migration gap: `promts.py` renamed to `prompts.py`, explicit integration, strict validated results |
| P0 | Empty evaluation pipeline; split metrics lose normalize_text/tokenize, SemanticEvaluator, math, TOP_K dependencies | Migration gap: preserve metric functions and restore Demo's per-method evaluator and parameterized aggregation |
| P0 | Batch launches deleted search/extraction scripts | Migration gap: direct current pipeline calls with complete/incomplete run manifest |
| P1 | Mutable duplicate Document/Evidence/Dual result declarations | Consolidated in models; original field names retained |
| P1 | Import-time model creation in tests and application code | Lazy resources and injected fakes; CLI/help/imports remain offline |
| P1 | Heading context can overflow chunk/model budget; oversized rows silently exceed limits | Inherited algorithm bugs: reserve heading space, reject impossible rows and prefixed model overflows |
| P1 | Index model name alone cannot establish experiment compatibility | Validate dimensions, prefixes, normalization, similarity, model/revision and source corpus fingerprints; fail on corrupted IDs/files |
| P1 | Any one valid citation ID previously credited a claim even if other IDs were fabricated | Strengthened structural scoring and extraction/citation validation; no semantic entailment claim |
| P1 | Semantic best-answer code mislabeled valid negative similarity as missing | Preserve negative score; distinguish unavailable from empty generated answer |
| P1 | Streamlit imports flat modules and eagerly loads unavailable data | Canonical configuration/engine, deferred resource load, missing-data guidance, preserved comparison/source/citation views |
| P2 | Empty README, architecture/contracts/evaluation/experiments/migration docs and Windows recipe | Documentation now describes actual commands/contracts, limitations and setup |

Normalization and Markdown parsing are preserved. The target's existing paragraph recognition improvement is retained. Duplicated script glue was replaced, not the original RAG methodology. No new retrieval/reranking algorithm or benchmark improvement is introduced.

## Original-to-modular mapping

| Demo module | Canonical target |
| --- | --- |
| `config.py` | `config.py` inside `codal_rag`, explicit `configs/*.yaml`, environment |
| `src/persian_normalizer.py` | `normalization/persian.py` |
| `src/markdown_parser.py` | `parsing/markdown.py` |
| `src/chunk_models.py` | `models.py` |
| `src/tokenizer_utils.py` | `chunking/tokenization.py` |
| `src/baseline_chunker.py` | `chunking/baseline.py` |
| `src/structure_chunker.py` | `chunking/structure.py` |
| `run_normalization.py` | `pipelines/normalize.py`, CLI normalize |
| `run_chunking.py`, `run_structure_chunking.py` | `pipelines/chunk.py`, CLI chunk/build |
| `run_indexing.py` | `indexing/representations.py`, `pipelines/embed.py`, `pipelines/index.py` |
| `src/embedding_model.py` | `indexing/embeddings.py` |
| `src/faiss_store.py` | `indexing/faiss_store.py` |
| `src/retrieval_engine.py` | `retrieval/engine.py`, shared result in models |
| `search_compare.py` | `pipelines/retrieve.py`, CLI retrieve, reusable cli_rendering |
| `src/evidence_context.py` | `extraction/evidence.py`, shared EvidenceItem in models |
| `src/three_level_extractor.py` | `extraction/three_level.py`, `extraction/prompts.py` |
| `src/extraction_validator.py` | `extraction/validation.py` |
| `src/citation_builder.py`, `src/citation_validator.py`, `src/citation_renderer.py` | `extraction/citations.py`, `citation_validation.py`, `cli_rendering.py` |
| `src/llm_client.py` | `llm/client.py` |
| `run_extraction.py` | `pipelines/extract.py`, `storage/legacy.py`, CLI extract |
| `src/evaluation_metrics.py` | `evaluation/{lexical,retrieval,semantic,faithfulness,aggregate}.py` |
| `run_evaluation.py` | `pipelines/evaluate.py`, `evaluation/aggregate.py`, JSON storage |
| `run_evaluation_pipeline.py` | `pipelines/batch.py` |
| `create_eval_template.py` | `evaluation/dataset.py`, CLI eval-template |
| `app.py` | `apps/streamlit_app.py` |

Paths in the right column are below `src/codal_rag` unless explicitly shown otherwise.

## Behavioral compatibility and intentional differences

Preserved: Persian normalization, Markdown block hierarchy, Baseline plain text/window overlap/IDs, normal structure-aware output/IDs, Proposed heading context, exact FAISS inner product and ID metadata, shared dual query vector/result shape, three-level fields, evidence/citation source linkage, lexical F1, heuristic relevance, available-results Precision denominator, Recall/MRR/Hit Rate, and macro/by-level aggregation. The 20/15/15 question template remains.

Intentional differences, covered by tests:

- Both paragraph/text block types use structure-aware splitting (the target already had this fix).
- Heading budgets and model limits fail clearly rather than permit silent overflow/truncation. Table rows are never split destructively.
- Any fabricated citation ID causes extraction failure and no structural credit for that claim. Missing source IDs are errors. Latent claims retain the two-ID/Persian-disclaimer requirement.
- Malformed JSON with arbitrary surrounding prose is rejected; a complete JSON code fence is accepted. This makes provider failures visible.
- Absent gold annotations yield null retrieval scores; missing/wrong document IDs cannot match a document-specific gold passage.
- Negative semantic similarity remains a valid score. An enabled evaluator with no generated candidates yields zero. Semantic calculations are explicitly optional; missing models are not silently treated as zero.
- Legacy indexes without integrity/version manifests must be rebuilt. Relative document paths replace collision-prone file stems in multi-folder inputs.
- New run/corpus snapshots prevent stale question/document artifacts from silently entering comparisons. Saved query/evidence provenance is checked before evaluation.

The pinned regression fixture was actually generated from the reference and compared against current outputs. Reference capture injects only missing plumbing (not credentials or alternate algorithm implementations). It does not claim the original incomplete application runs unmodified.

## Migration steps

1. Check out the repair branch and create a clean virtual environment. Install required extras.
2. Put Markdown inputs in a chosen folder and optional company/year/type metadata in a JSON mapping keyed by relative document ID.
3. Use `configs/base.yaml` plus explicit overrides. Set a provider model and your own rotated key in the environment. Do not copy Demo config.py.
4. Rebuild both chunk snapshots and indexes using `build`; do not reuse old FAISS artifacts. Smoke and real-model indexes are incompatible by design.
5. Update gold document IDs to the new relative IDs. Re-run retrieval/extraction or evaluate complete compatible legacy runs read-only.
6. Launch Streamlit with the same configuration. Use the CLI JSON files as the interchange format.

## Security and remaining verification

A hardcoded API credential exists in the original reference configuration. It must be revoked and rotated by its owner, even if that repository is private. Its value was not printed, copied, used, or committed. The reference stays read-only, so credential revocation/removal there is outstanding owner work. New settings accept API keys only through environment/runtime injection, ignore `.env` files in Git, hide keys from repr, and omit keys and provider URLs from saved public configuration/reports. Provider error bodies are not echoed.

This change does not rewrite repository history, revoke keys, or claim to cleanse the original repository. Production datasets, real multilingual/semantic weights, live paid providers and native Windows execution remain external verification items. See experiments.md for exact test environment and commands. All research-quality comparisons require a real annotated dataset and real model execution.
