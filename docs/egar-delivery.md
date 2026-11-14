# EGAR delivery and verification

Implementation base: `76605cc29922b85421eb3fb2bf75ff867837763f`.
Branch: `feat/evidence-gap-adaptive-retrieval`. No merge into master.

## Implemented behavior

The existing RAG pipeline now has optional Persian BM25 and rank-fused hybrid
retrieval, explicit document/company/period scope, versioned source-unit provenance,
and a bounded EGAR controller. The controller generates candidate claims, evaluates
missing evidence, retrieves new targeted evidence, removes redundancy, reassesses
the packed context, and either returns supported claims or abstains with a trace.
Semantic judgments are explicitly distinguished from structural citations and
deterministic arithmetic. Conceptual labels remain independent of inference levels.

The historical v1 defaults and evaluation definitions remain available. EGAR is off
unless explicitly enabled. The separate research runner supports A/B/C/D and D+EGAR
with common models, initial settings, generation prompt and context budget. D versus
C remains the main chunking comparison. D versus D+EGAR is an independent extension
comparison. There are no new mandatory dependencies or GitHub Actions workflows.

See [architecture, schemas and research protocol](egar.md) for the full technical
description, reference-paper mapping, Windows/PyCharm commands and example JSON.

## Changed and new files

| File | Change |
| --- | --- |
| `README.md` | Optional extension overview, commands and documentation links |
| `apps/streamlit_app.py` | Retrieval selection, optional EGAR, readable Persian results and research-only trace |
| `configs/base.yaml` | Explicit dense/v1/EGAR-off defaults |
| `configs/hybrid.yaml` | Shared BM25/RRF settings |
| `configs/research.yaml` | Common v2 generation/context contract, EGAR off |
| `configs/egar.yaml` | Explicit opt-in with bounded queries, rounds, calls and time |
| `docs/architecture.md` | Links new modules to the preserved architecture |
| `docs/data-contracts.md` | Documents backward-compatible v1 and optional v2/source-unit metadata |
| `docs/evaluation-protocol.md` | Separates historical proxy metrics from research evaluation |
| `docs/experiments.md` | Links controlled A-D and EGAR instructions |
| `docs/egar-audit.md` | Pre-implementation findings and plan |
| `docs/egar.md` | Full motivation, algorithm, settings, schemas, protocol, commands and limitations |
| `docs/egar-delivery.md` | This file: changed files, validation and requirement status |
| `src/codal_rag/cli.py` | Scoped answering, optional EGAR, benchmark/experiment/evaluation/demo commands |
| `src/codal_rag/config.py` | Validated retrieval, extraction and EGAR settings |
| `src/codal_rag/models.py` | Optional source-unit mappings with old default serialization preserved |
| `src/codal_rag/retrieval/engine.py` | Dense/BM25/hybrid methods, scoped heading-aware search and retrieval provenance |
| `src/codal_rag/retrieval/sparse.py` | Persian tokenization, BM25 and deterministic rank-only RRF |
| `src/codal_rag/retrieval/scope.py` | Explicit identity/period constraints and conservative routing |
| `src/codal_rag/indexing/faiss_store.py` | Read-only FAISS selector for allowed record IDs |
| `src/codal_rag/extraction/evidence.py` | Optional untruncated v2 context; historical v1 limit retained |
| `src/codal_rag/extraction/prompts.py` | Separate v2 definitions and conceptual labels; original v1 prompt retained |
| `src/codal_rag/extraction/three_level.py` | Explicit schema choice and v2 label validation |
| `src/codal_rag/extraction/validation.py` | V2 candidate validation can represent two units in one chunk; v1 remains strict |
| `src/codal_rag/extraction/source_units.py` | Exact normalized Markdown block/token mappings independent of chunk boundaries |
| `src/codal_rag/llm/client.py` | Capture provider usage when returned |
| `src/codal_rag/egar/__init__.py` | Optional module namespace, no import-time services |
| `src/codal_rag/egar/analyzer.py` | Five-state sufficiency, semantic judge contract, quotes, complementarity and checks |
| `src/codal_rag/egar/controller.py` | Bounded targeted retrieval loop, preservation of initial evidence, abstention and audit trail |
| `src/codal_rag/egar/evidence.py` | Whole-unit context packing, deduplication and source traces |
| `src/codal_rag/egar/financial.py` | Numerals/signs/units/periods, anchored readings and Decimal arithmetic |
| `src/codal_rag/egar/runtime.py` | Per-question call/time/retry accounting and nullable actual token/cost metrics |
| `src/codal_rag/egar/demo.py` | Explicit synthetic real-FAISS/BM25 demonstration with mocked semantic services |
| `src/codal_rag/pipelines/chunk.py` | Preserve extra metadata, source-unit maps and supplied original coordinates |
| `src/codal_rag/pipelines/extract.py` | V1-preserving dispatch to optional v2/EGAR with citations and operational records |
| `src/codal_rag/pipelines/batch.py` | Reuse the existing retriever when explicitly opting into EGAR |
| `src/codal_rag/pipelines/evaluate.py` | Prevent adaptive runs being silently scored with historical definitions |
| `src/codal_rag/pipelines/research.py` | Controlled A-D/EGAR runs, no gold in generation, artifact checks and paired reports |
| `src/codal_rag/evaluation/benchmark.py` | Unfilled 260-question plan, full protocol checks and company split validation |
| `src/codal_rag/evaluation/research_metrics.py` | Alternative evidence-set coverage, adjudicated claim/label/support metrics and error taxonomy |
| `src/codal_rag/evaluation/statistics.py` | Paired company-cluster bootstrap, 95% CIs, effects and sample counts |
| `tests/unit/test_egar.py` | Sufficiency, failure, numerical, complementarity, budget and compatibility cases |
| `tests/unit/test_hybrid.py` | Persian BM25/RRF, scope, read-only indexes and source coordinates |
| `tests/unit/test_research.py` | Dataset/split contracts, metrics, adjudications and clustered statistics |
| `tests/integration/test_egar_pipeline.py` | Real FAISS demo, all experiment arms, provenance, CLI and Streamlit |

## Executed verification

| Check | Result |
| --- | --- |
| Before changes: existing suite | 117 passed, 2 external tests deselected, 0 failed |
| After changes: complete suite | 165 passed, 2 external tests deselected, 0 failed |
| Existing tests removed/disabled to pass | None |
| Ruff configured checks | Passed |
| Python compile check | Passed |
| Dependency consistency (`pip check`) | No broken requirements |
| Source distribution / wheel build | Both built successfully |
| Core-only wheel outside checkout | All 64 application modules imported without optional model/API/UI packages |
| Core-only installed CLI synthetic example | Completed initial retrieval, one targeted round, reassessment and grounded mock inference |
| Streamlit AppTest | Original retrieval path and enabled EGAR path passed |
| Source/index mutation | Tests confirm answering does not change FAISS/index artifact bytes |
| Historical regression snapshot | Passed unchanged |

Verification environment: Linux, Python 3.12.14, NumPy 2.5.4, FAISS CPU 1.16.0,
PyYAML 6.0.3, Pytest 9.2.0, Ruff 0.17.5, Streamlit 1.67.0, OpenAI SDK 3.51.0,
build 1.6.1, setuptools 84.2.1, wheel 0.48.0. These are observed test versions,
not a new mandatory lockfile. Building without isolation initially needed the
already-declared setuptools/wheel build tooling installed in the local venv.

## Demonstration

```powershell
python -m codal_rag egar-demo --output data/egar-demo.json
```

Observed sequence: initial Proposed retrieval contains the synthetic paragraph about
increased short-term liabilities; the mocked assessor requests operating cash-flow
evidence; a new Persian query retrieves the separate cash-flow paragraph; the combined
two units support a **qualified** liquidity-pressure inference under the mock judge.
The run uses one supplementary round and four mocked LLM calls. The D output has
only the initially available fact. Full initial/final evidence, gaps, queries, source
maps and verdicts are in the output JSON. This demonstrates software integration,
not semantic model accuracy or a real financial conclusion.

## Requirement status and remaining evidence

| Requirement group | Status | Limit |
| --- | --- | --- |
| Repository audit and architecture preservation | Complete | Based on inspected revision above |
| Dense, BM25 and hybrid prerequisites | Complete in software/tests | Real multilingual weights were not executed |
| Optional analyzer, targeted controller and five-state output | Complete in software/tests | Generic semantic judgment depends on configured LLM; correctness is not guaranteed |
| Independent conceptual labels and v2 traceability | Complete | Original PDF coordinates must actually be supplied; they are not invented |
| Budgets, deduplication, scope, failure handling and abstention | Complete in software/tests | In-flight provider requests cannot be forcibly cancelled; unknown usage is null |
| Financial validation | Implemented for practical checks; limited domain coverage | No general accounting theorem prover, arbitrary currency reasoning or reliable automatic causality proof |
| CLI, existing Streamlit page and programmatic access | Complete in software/tests | Native Windows UI/runtime was not executed here |
| A-D and D/D+EGAR research runner | Complete in software/tests | All arms exercised only with synthetic fixtures and mocked semantic services |
| Metrics, alternative gold sets, error taxonomy and 10,000 clustered resamples | Complete mechanism | Independent adjudications are required for factual claim metrics |
| Real 60-report / 260-question experiment and improvement claims | Not run; requires data/models/annotations | No real dataset, human gold, live provider results, significance or superiority is fabricated |
| External cached-model and live-provider tests | Not run, explicitly opt-in | Requires locally available model weights and configured provider credentials |
| Merge into master | Not performed | Requires separate explicit authorization |

The code is ready for an actual research evaluation, subject to providing and
validating those inputs. Software test success does not establish the research
hypothesis. Keep test companies/labels held out until development choices are frozen.
