# EGAR implementation audit

Base: `master` / `fix/complete-rag-migration`, both at
`76605cc29922b85421eb3fb2bf75ff867837763f`, inspected 2026-11-14.
No open pull requests were present. Implementation branch:
`feat/evidence-gap-adaptive-retrieval`. No merge into master is authorized.

The full package, tests, configurations, README, architecture, data contracts,
evaluation protocol, experiments and migration report were inspected. The supplied
approved proposal (`final2.docx`) confirms A-D, D versus C as the main comparison,
60 reports / 20 companies / five industries / 1401-1403, 60 development and 200
test questions, and 10,000 paired bootstrap samples at company level.

| Component | Before this extension | Implementation decision |
| --- | --- | --- |
| Persian normalization and Markdown parser | Present; preserves financial signs, digits and heading/table structure | Reuse without changing algorithms |
| Fixed and structure-aware chunkers | Present with regression fixtures; table headers repeated across parts | Preserve chunk text, IDs, defaults and embedding representations |
| Metadata and provenance | Document/company/year/type, headings and block IDs; optional PDF coordinates absent | Preserve available provenance; accept explicit extra metadata; never infer PDF coordinates |
| Embeddings and FAISS | Shared multilingual E5, normalized exact inner product, artifact hashes, offline hash backend | Reuse; add read-only constrained search |
| Sparse / hybrid retrieval | Missing | Dependency-free Persian BM25 and rank-only RRF, identical settings for both chunkers |
| Three-level extraction | Present, JSON and citation-ID validation | Keep v1 compatibility; v2 separates conceptual labels from inference levels |
| Semantic sufficiency and complementary evidence | Missing; two IDs alone counted as latent structurally | Separate structural checks, anchored evidence checks, deterministic arithmetic and optional semantic judge |
| Adaptive retrieval | Missing | Bounded deterministic controller, fail-closed final claims, explicit scope, trace log |
| Evaluation | Historical lexical/similarity proxies and 50-question template | Preserve historical evaluation; add separate CodalBench contracts, claim adjudications and paired clustered analysis |
| CLI / Streamlit | Present; retrieval comparison and saved-citation viewer | Integrate optional setting without a second app |
| Real CodalBench / human annotations | Not in repository | Do not invent; fixtures are software tests only |

Baseline: a clean editable environment ran `python -m pytest -q`: **117 passed,
2 external tests deselected, 0 failed**. The host initially lacked pytest; tests
were run after installing the repository's existing dev/UI/LLM extras in a venv.

Plan, in dependency order: hybrid retrieval and scope; versioned evidence
contracts and sufficiency analyzer; bounded controller and extraction integration;
research experiment/annotation/statistical tools; CLI/UI; deterministic tests and
documentation. No new mandatory dependencies, agents, GraphRAG, neural reranker,
trading, fine-tuning, paid Actions, or source/index writes during answering.
