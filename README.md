# Codal Intelligence

**سامانه هوشمند تحلیل گزارش‌های مالی** is a Persian, right-to-left Streamlit workspace for asking questions about Codal reports, tracing findings to their sources and inspecting research runs. Open **تحلیل هوشمند** to ask an independent question. **تنظیمات → پژوهشگر** adds retrieval diagnostics and the EGAR trace where available. See the [application guide and screenshots](docs/codal-intelligence.md).

An installable Persian financial-document RAG experiment package. It compares fixed-size Baseline chunks with structure-aware Proposed chunks using the same embedding model, normalized query vector, similarity metric, corpus, and retrieval depth. It then extracts explicit, implicit, and latent claims with traceable evidence and evaluates saved runs.

The default embedding model remains `intfloat/multilingual-e5-base`. No API credentials or document datasets are included. The deterministic smoke backend checks plumbing; its scores are **not** research results.

Optional **Evidence-Gap-Guided Adaptive Retrieval (EGAR)** adds bounded, scoped searches
for missing complementary evidence. It is disabled by default. Dense, Persian BM25 and
hybrid RRF retrieval support the approved A-D experiments; D versus C remains the main
chunking comparison. D versus D+EGAR is a separate ablation. See the [EGAR guide](docs/egar.md),
[audit](docs/egar-audit.md) and [delivery status](docs/egar-delivery.md).

```powershell
# Reproducible synthetic demonstration; no API or model download.
python -m codal_rag egar-demo --output data/egar-demo.json
# Create the unfilled approved 260-question annotation plan.
python -m codal_rag codalbench-template --output data/codalbench.json
```

## Install on Windows (PowerShell)

Python 3.10–3.13 is declared supported; Python 3.12 is recommended and was tested on Linux. Native Windows execution and real model/API integration still need verification on your machine.

```powershell
git clone https://github.com/hhhesssammm/Codal_RAG_LLM.git
cd Codal_RAG_LLM
git switch feat/codal-intelligence-ui
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]"
.\.venv\Scripts\Activate.ps1
```

If activation is disabled, invoke `.\.venv\Scripts\python.exe -m codal_rag` in place of `codal-rag`. On Linux/macOS, use `python3 -m venv .venv` and `source .venv/bin/activate`.

Core installation (`pip install -e .`) includes NumPy, PyYAML, and CPU FAISS. Extras:

| Extra | Purpose |
| --- | --- |
| `models` | Sentence Transformers, Transformers, and their model/runtime dependencies |
| `llm` | OpenAI-compatible API client |
| `ui` | Streamlit |
| `dev` | Pytest, Ruff, wheel build checks |

Install real-model/API support only when needed: `python -m pip install -e ".[models,llm,ui,dev]"`. Model weights are downloaded on first explicit use unless `embedding.local_files_only` is true. Importing modules and asking for CLI help never downloads weights or calls APIs.

## Offline smoke run

```powershell
codal-rag --config configs/base.yaml --config configs/smoke.yaml build --input examples
codal-rag --config configs/base.yaml --config configs/smoke.yaml retrieve --query "علت کاهش فروش چیست؟" --output data/retrieval.json
python -m pytest -q
```

This exercises real FAISS with deterministic hash vectors and a whitespace tokenizer. It does not represent multilingual-E5 quality. Tests also exercise extraction with mocked LLM responses, batch evaluation, and Streamlit; no key or paid service is needed.

## Real model and extraction

```powershell
# Rebuild both indexes when changing from smoke to a real embedding model.
codal-rag --config configs/base.yaml build --input C:/reports/markdown
codal-rag --config configs/base.yaml retrieve --query "علت کاهش فروش چیست؟" --output data/retrieval.json

$env:CODAL_LLM_API_KEY = '<set-your-key-locally>'
$env:CODAL_LLM_MODEL = '<your-provider-model-id>'
# Optional for another OpenAI-compatible provider:
# $env:CODAL_LLM_BASE_URL = 'https://your-provider.example/v1'
codal-rag --config configs/base.yaml extract --retrieval data/retrieval.json --method proposed --output data/extraction.json
```

Set credentials in your shell or secret manager, never in source files or YAML. `.env` files are ignored by Git and are **not automatically loaded**. `OPENAI_API_KEY` and `OPENAI_BASE_URL` are fallback environment variables. API responses with unknown evidence IDs, invalid claims, missing latent disclaimers, or invalid JSON fail validation instead of being saved as successful extractions.

## Configuration

Precedence, lowest to highest:

1. Typed, validated defaults in `codal_rag.config`.
2. YAML files explicitly listed with repeated `--config`, in their supplied order. If none are supplied, `CODAL_CONFIG` may identify one YAML file.
3. `CODAL_<SECTION>_<FIELD>` environment variables, for example `CODAL_EMBEDDING_MODEL`, `CODAL_LLM_TIMEOUT`, `CODAL_CHUNKING_MAX_TOKENS`, `CODAL_EVALUATION_SEMANTIC_ENABLED`. Top-level fields use `CODAL_TOP_K`, `CODAL_PROJECT_ROOT`, etc.
4. Explicit `--root`, `--mode`, `--top-k`, `--retrieval-mode`, and `--egar`/`--no-egar` CLI overrides.

`project_root` in YAML is relative to that YAML file's directory. All generated data paths are relative to `project_root/data_dir` (an absolute `data_dir` also works). Without configuration, the project root defaults to the current directory. Explicit CLI `--input`, `--output`, and other file arguments are relative to the shell's current directory, or may be absolute. Installed imports do not depend on either directory.

`configs/base.yaml` documents the application defaults. `baseline.yaml` and `proposed.yaml` change only the method, preserving the embedding configuration. `local.yaml` is an optional empty, non-secret override; it is never loaded implicitly. Private non-secret deployment overrides may go in ignored `configs/*.private.yaml`. Unknown fields, invalid types, impossible overlap, inconsistent cosine normalization, and YAML credentials produce clear errors. `codal-rag --config configs/base.yaml config` displays effective settings without credentials or provider URLs.

Defaults include 500 chunk tokens, 100 Baseline overlap, batch size 16, top-k 5, cosine through normalized inner product, seed 42, and disabled semantic evaluation. Pin `embedding.revision` to a model commit for research reproducibility. Model-length limits include query/passage prefixes and special tokens; over-length inputs fail rather than silently truncating.

## Commands

Global options appear **before** the subcommand. Both `codal-rag` and `python -m codal_rag` use the same entry point.

| Command | Required / useful arguments | Result |
| --- | --- | --- |
| `config` | `--config FILE` globally | Effective safe configuration |
| `normalize` | `--input FILE_OR_DIR --output DIR` | Normalized Markdown, preserving relative paths |
| `chunk` | `--input FILE_OR_DIR [--metadata JSON]` | Atomic corpus snapshots under `data/chunks/{method}` |
| `embed` | — | Vectors, records, and manifests under `data/embeddings/{method}` |
| `index` | `[--from-embeddings]` | Embeds then indexes, or checks and reuses the embedding cache |
| `build` | `--input FILE_OR_DIR [--metadata JSON]` | Chunk, embed, and index |
| `retrieve` | `--query TEXT [--output JSON]` | Both methods' hits and index provenance |
| `extract` | `--retrieval JSON --method baseline\|proposed --output JSON` | Validated claims, evidence, citations |
| `eval-template` | `--output JSON` | Blank 20 Explicit / 15 Implicit / 15 Latent questions |
| `batch` | `--questions JSON [--output DIR]` | Retrieval and extraction for both methods per question |
| `evaluate` | `[--runs DIR] [--output JSON]` | Aggregate, per-question, and by-type evaluation |
| `answer` | `--query TEXT --output JSON [--scope JSON]` | Retrieve/extract with optional EGAR |
| `egar-demo` | `--output JSON` | Offline synthetic adaptive example |
| `codalbench-template` | `--output JSON` | Unfilled approved 260-question plan |
| `validate-benchmark` | `--dataset JSON [--full-protocol]` | Validate annotations, company splits and plan |
| `experiment` | `--dataset JSON --arms D D+EGAR --output DIR` | Common v2 prompt/budget, separate research arms |
| `research-evaluate` | `--runs DIR --dataset JSON --output JSON [--adjudications JSON]` | Coverage, independent claim judgments and clustered bootstrap |

`extract --response-file JSON` explicitly replays a recorded mock LLM response for offline testing; outputs record that the LLM was mocked. It never silently substitutes for a live provider. `retrieve` and `batch` are comparison workflows and require both indexes, even if you previously built one method with `--mode`.

Documents are identified by their path relative to the supplied input directory, without `.md`, e.g. `company/report`; equal filenames in different folders remain distinct. A metadata file maps these IDs to optional `company`, `year`, and `document_type` fields. Source paths and document IDs are derived from the actual input, not guessed.

## Streamlit

```powershell
# Existing checkout and project virtual environment, from the repository root
git fetch origin
git switch feat/codal-intelligence-ui
git pull --ff-only origin feat/codal-intelligence-ui
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

The UI uses existing project dependencies. Open the local URL printed by Streamlit. For indexes built with the offline example, select `base.yaml` followed by `smoke.yaml` under **تنظیمات → پیکربندی پیشرفته**. Research mode offers a retrieval-only comparison that runs with these indexes without a provider. Open an existing extraction JSON in the assistant or an evaluation report in **ارزیابی پژوهشی** without running a model.

**مقایسه روش‌ها** runs the selected LLM-only, Baseline, Proposed and Proposed + EGAR conditions. **مدیریت اسناد** reads document metadata, source passages and the two chunk representations. The [application guide](docs/codal-intelligence.md) covers configuration, provenance, offline use and verification limits.

## Evaluation and checks

```powershell
codal-rag eval-template --output data/questions.json
# Manually annotate questions, expected answers, types, and document-specific gold evidence.
codal-rag --config configs/base.yaml batch --questions data/questions.json
codal-rag --config configs/base.yaml evaluate
python -m compileall src apps tests
python -m pytest -q
python -m pip check
python -m ruff check src apps tests
python -m build --no-isolation
```

An incomplete batch is not scored. Semantic metrics are `null` when disabled/unavailable, with the reason included. Structural citation validity and embedding similarity are explicitly separate from factual entailment. No performance superiority of Proposed over Baseline is claimed.

See [architecture](docs/architecture.md), [data contracts](docs/data-contracts.md), [evaluation protocol](docs/evaluation-protocol.md), [experiments](docs/experiments.md), and the [audit/migration report](docs/migration.md) for exact definitions, preserved behavior, deliberate fixes, and verification limits.
