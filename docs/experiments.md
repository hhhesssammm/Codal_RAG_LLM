# Reproducible experiments

For the approved A-D matrix and separate D/D+EGAR ablation, use the [EGAR experiment
instructions](egar.md#windows-powershell--pycharm). The historical v1 commands and
migration verification recorded below remain available and are not new EGAR results.

1. Install the package and record Python/platform/dependency versions (`python -m pip freeze`). Use a fixed dataset, stable input root, and reviewed question annotations.
2. Pin `embedding.model` and `embedding.revision` to the intended model/commit. Set identical normalization, prefixes and similarity for both methods. Record hardware; neural-model results are not guaranteed bitwise identical across devices/runtime versions.
3. Use `build --input ...` in `mode: both`. Changing source contents or model settings requires rebuilding both indexes. Hashes check document content as well as IDs; same filenames do not establish an identical corpus.
4. Run `batch --questions ... --output data/evaluation_runs/experiment-name` with explicit LLM model/provider settings. A provider can change behavior despite a fixed model name and temperature zero; preserve returned artifacts for repeat evaluation.
5. Run `evaluate --runs data/evaluation_runs/experiment-name --output data/evaluation_results/experiment-name.json`. Keep the dataset, config, embedding/index manifests and run JSONs together. Reports include per-artifact SHA-256 digests and omit API credentials/provider URLs.

Default seed is 42 and is recorded. Exact FAISS search itself is deterministic for fixed vectors apart from tied-score ordering, which should not be treated as a meaningful ranking distinction. No approximate search is introduced. Pinning weights and dependencies is still necessary; a model name with an unspecified revision is not an immutable research artifact.

## Offline checks

```bash
python -m pytest -q
python -m compileall src apps tests
python -m pip check
python -m ruff check src apps tests
python -m build --no-isolation
```

Default tests block network connections in the test process and use deterministic fixtures. They exercise real FAISS persistence/search, a mocked SentenceTransformer adapter, mocked provider responses, actual CLI subprocesses outside the checkout, batch evaluation, and Streamlit AppTest when the UI extra is installed. Two external checks are deselected by default.

Opt-in model check (cached model only): set `CODAL_RUN_MODEL_TESTS=1`, install the models extra and run `python -m pytest -q -m external -k cached_multilingual_e5`. Opt-in provider check: set `CODAL_RUN_API_TESTS=1`, configure your own credentials/model, and run `python -m pytest -q -m external -k live_provider`. The latter sends a potentially billable request. Neither external test was run for this migration.

## Regression capture

`tests/fixtures/demo_regression.json` was captured from Demo commit `208bb64bdea92f919008eb5bba2ce24588a92daf` using synthetic Persian Markdown and deterministic vectors. Default regression tests compare repaired output to this pinned snapshot.

To regenerate from an independently obtained reference source snapshot:

```bash
python -m tests.reference_capture --reference /path/to/Demo-source --output tests/fixtures/demo_regression.json
```

The capture utility never reads/executes Demo `config.py`. It injects a whitespace tokenizer, the missing Baseline `Chunk` record, and non-secret configuration symbols only where the original imports cannot run. It exercises unchanged original algorithm bodies. This is **not** a claim that the unmodified Demo application installs or runs end to end: the reference itself lacks some symbols and packaging.

Snapshot coverage: Persian normalization, parsed block hierarchy, Baseline windows, ordinary structure-aware chunks, embedding representation, FAISS exact scores/record mapping, dual-result shape, three-level output schema, evidence/citation outputs, retrieval metrics and lexical F1. Separate tests cover documented bug fixes where equality with Demo would be undesirable (paragraph splitting, heading budgets, oversized tables, unknown citations, negative similarity, provenance and stale caches).

## Verified environment and limits

Migration checks ran on Linux with Python 3.12, NumPy 2.5.4, FAISS CPU 1.15.1, PyYAML 6.0.3, OpenAI SDK 3.46.0, Streamlit 1.67.0, Pytest 9.2.0, Ruff 0.17.4, setuptools 84.2.1, and build 1.6.1. Dependency constraints allow compatible updates; this is not a fully pinned Windows lockfile.

Real E5/semantic model weights, live providers, production Codal datasets, and native Windows execution were not exercised. No benchmark improvement is reported. Smoke and mock outputs must not be published as model-quality evidence.

## Executed migration checks

| Check | Observed result |
| --- | --- |
| Initial target `python -m pytest -q` | 3 collection errors from obsolete module imports |
| Editable installation `python -m pip install -e . --no-build-isolation` | Successful |
| Final `python -m pytest -q` | 117 passed; 2 external tests deselected; 0 failures |
| `python -m compileall -q src apps tests` | Successful |
| `python -m pip check` | No broken requirements |
| `python -m ruff check src apps tests` | All configured E9/F checks passed |
| `python -m build --no-isolation` | Source distribution and wheel built successfully |
| Clean core-only wheel installation outside checkout | All 50 application modules imported; no optional model/API/UI libraries imported |
| Installed-wheel CLI `build` and `retrieve` with smoke profile | Both real FAISS indexes built and retrieved; 1 Baseline / 4 Proposed sample chunks |
| Streamlit AppTest | Missing-index initialization and successful two-column retrieval passed |
| `git diff --check` | Passed after trimming two trailing blank lines |
| Credential-pattern scan | No matches in target working tree or fetched target history; reference credential separately flagged for rotation |

These checks use deterministic synthetic data. The sample chunk counts are execution evidence, not quality benchmarks.
