# Evaluation protocol

This page specifies the unchanged **legacy evaluator**. The approved 260-question
CodalBench contracts, exact passage coverage, independent claim adjudication and
10,000-sample paired company bootstrap are implemented separately; see [EGAR](egar.md).
Use `research-evaluate` for adaptive runs. Historical lexical/similarity metrics
are not silently redefined as factual correctness.

The runner restores Demo's per-question comparison and aggregation. The question type selects the Explicit, Implicit, or Latent claims used for answer/grounding metrics. Both methods use the same question, gold annotations, retrieval depth, and thresholds. This is an evaluation mechanism, not evidence of a quality improvement.

## Retrieval relevance and metrics

Text normalization and lexical relevance retain Demo's definitions. Matching uses substring, gold-token recall, and SequenceMatcher, taking the maximum; threshold defaults to 0.70. These are heuristic matches, not human relevance judgments. A nonempty gold document ID must match the retrieved document ID; anonymous or wrong-document chunks cannot earn credit.

| Metric | Definition |
| --- | --- |
| Precision@K | Relevant returned chunks in top K / `max(min(K, number of returned hits), 1)` |
| Recall@K | Distinct gold evidence items matched in top K / number of gold items |
| Per-question reciprocal rank | 1 / first relevant rank within top K, or 0 |
| MRR | Mean of per-question reciprocal ranks |
| Evidence Hit Rate | Mean of the indicator that any gold item was matched |

The available-results Precision denominator deliberately preserves Demo (some benchmarks divide by K even when fewer hits exist). Results should not be compared against those conventions without recomputing. Missing gold evidence yields `null` retrieval metrics and is excluded from means. The report retains match details and gold counts.

## Answer and grounding metrics

| Metric | Interpretation |
| --- | --- |
| Lexical answer correctness | Maximum multiset token F1 between expected answer and any claim in the question's level |
| Answer accuracy | Fraction whose lexical correctness is at least 0.60 |
| Semantic answer correctness | Maximum embedding cosine similarity against candidate claims, when enabled |
| Semantic accuracy | Fraction whose similarity is at least 0.70, excluding unavailable values |
| Citation Faithfulness | Fraction of claims with a nonempty list of entirely valid evidence IDs; structural validity only |
| Semantic Faithfulness | Fraction with cited-evidence similarity at least 0.50; a similarity proxy, not factual entailment |

The historical field names `supported_claims` and `semantic_faithfulness` are retained for report compatibility. They must **not** be described as factual support judgments. A contradiction can still have high embedding similarity. Human/NLI entailment assessment remains separate and is not implemented or implied.

Citation validity is intentionally stricter than Demo: a mixture of valid and fabricated IDs does not pass. The extraction pipeline already rejects such claims, but standalone evaluation must not over-credit imported outputs. Saved evidence text and chunk/document identity are checked against retrieval artifacts before scoring. Latent validation enforces distinct IDs; it does not assert that two overlapping chunks are independent sources.

No candidate claims means zero lexical score and, when the semantic evaluator is enabled, zero semantic answer score. Valid negative cosine scores remain negative rather than being mislabeled as unavailable. Disabled/unavailable semantic models yield `null` with the reason recorded; the default suite never downloads them. Enable real semantic evaluation explicitly through `evaluation.semantic_enabled` / `CODAL_EVALUATION_SEMANTIC_ENABLED`.

Aggregates are macro means over available question metrics, plus Explicit/Implicit/Latent groups. Per-question results remain available for auditing; inspect sample counts and annotation coverage alongside means. The default template contains 20/15/15 questions as in Demo, but no invented financial dataset or benchmark scores are supplied.
