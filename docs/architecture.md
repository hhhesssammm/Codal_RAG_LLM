# Architecture

The default historical pipeline below remains available. The optional additions are
documented in [EGAR](egar.md): `retrieval/sparse.py` adds BM25/RRF,
`retrieval/scope.py` controls identity/period routing, `egar/` implements bounded
assessment/retrieval, and `pipelines/research.py` isolates A-D and D+EGAR experiments.
New chunk metadata includes source-unit mappings without changing embedding text.

`src/codal_rag` is the only application package; installing it exposes `codal-rag` and `python -m codal_rag`. No runtime code adjusts `sys.path`, imports the old flat `src` modules, or reads `config.py`. CLI and Streamlit pass the same validated `AppConfig` into reusable pipelines. Importing a module does not start processing or load a model.

| Layer | Responsibility |
| --- | --- |
| `normalization/persian.py` | Original Persian Unicode, character, digit, date, unit, and Markdown-preservation rules |
| `parsing/markdown.py` | Original heading stack and paragraph/table/list/blockquote blocks |
| `chunking/baseline.py` | Original Markdown-to-plain-text and fixed token windows with overlap |
| `chunking/structure.py` | Heading-aware buffering, paragraph/sentence/token splitting, independent tables and row-boundary splits |
| `chunking/tokenization.py` | Lazy model tokenizer or explicit whitespace smoke tokenizer |
| `models.py` | Canonical document, Baseline chunk, structured chunk, evidence, and dual-result records |
| `indexing/representations.py` | Baseline content versus Proposed structural retrieval text |
| `indexing/embeddings.py` | Shared multilingual model, explicit prefixes, normalization and context checks |
| `indexing/faiss_store.py` | Exact `IndexFlatIP` inside `IndexIDMap2`, metadata identity and artifact integrity |
| `retrieval/engine.py` | One normalized query/vector searched against both compatible indexes |
| `extraction` | Three-level prompt, evidence assembly, fail-closed schema/ID validation and citations |
| `llm/client.py` | Configurable OpenAI-compatible endpoint, bounded retries/timeout and safe provider errors |
| `evaluation` | Lexical, retrieval, citation and optional semantic proxy metrics; aggregation |
| `pipelines` | File-oriented normalize/chunk/embed/index/retrieve/extract/batch/evaluate orchestration |
| `storage` | Strict, atomic UTF-8 JSON writes and read-only adapters for legacy retrieval envelopes |

`build` normalizes each source, checks Markdown structure, builds method-specific corpus snapshots, uses one embedding model, and saves both FAISS indexes. `embed` and `index --from-embeddings` allow inspecting/reusing cached vectors; stale caches fail validation. A whole corpus snapshot replaces the previous one, preventing deleted documents from remaining in an index through stale per-document files.

The model, revision setting, prefixes, dimension, normalization, similarity, original-document fingerprints and artifact checksums are persisted. Dual retrieval checks these settings and the corpus before searching. Exact inner product remains the default FAISS operation; with unit vectors it is cosine. Raw inner product is available only through an explicit compatible configuration. Optional hybrid retrieval fuses dense/BM25 ranks; no approximate search or neural reranker is required.

Structure-aware content budgets reserve space for heading context. Tables keep headers and split only between rows. A row that cannot fit fails clearly instead of being truncated or silently exceeding the model context. The model adapter additionally validates the actual prefixed input plus special tokens, because tokenizer boundary effects and query length can differ from chunk estimates.

Extraction is lazy: no evidence produces an empty insufficient-evidence response without an API call. Claims must contain a nonempty claim and basis, bounded confidence, and only supplied evidence IDs. Latent claims require two distinct evidence IDs and the original Persian inference disclaimer convention. Valid IDs do not prove semantic support. Evidence remains mapped to original chunk/document/source metadata; the model cannot choose a new source record.

Hash embeddings and whitespace tokenization are explicit smoke tools. The default suite uses those or injected services, together with real FAISS. Full multilingual embeddings and provider calls are separate opt-in tests. `interfaces.py` provides lightweight injection protocols rather than a service framework.
