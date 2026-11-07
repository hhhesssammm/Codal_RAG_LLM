from __future__ import annotations

from pathlib import Path
from codal_rag.config import AppConfig
from codal_rag.evaluation.aggregate import aggregate_method, aggregate_by_type
from codal_rag.evaluation.dataset import validate_question
from codal_rag.evaluation.lexical import answer_correctness, get_candidate_answers
from codal_rag.evaluation.retrieval import calculate_retrieval_metrics
from codal_rag.evaluation.faithfulness import calculate_citation_faithfulness, calculate_semantic_faithfulness
from codal_rag.evaluation.semantic import SemanticEvaluator
from codal_rag.storage.json_io import load_json, save_json, digest
from codal_rag.storage.legacy import select_retrieval_results

def evaluate_method(
    method: str,
    gold: dict,
    retrieval_payload: dict,
    extraction_output: dict,
    semantic_evaluator: SemanticEvaluator,
    config: AppConfig,
) -> dict:

    retrieval_results = (
        select_retrieval_results(
            payload=retrieval_payload,
            method=method,
        )
    )

    gold_evidence = (
        gold.get(
            "gold_evidence",
            [],
        )
    )

    # ----------------------------------------
    # Retrieval
    # ----------------------------------------

    retrieval_metrics = (
        calculate_retrieval_metrics(
            retrieval_results=(
                retrieval_results
            ),
            gold_evidence=(
                gold_evidence
            ),
            k=config.top_k,
            evidence_threshold=(
                config.evaluation.evidence_threshold
            ),
        )
    )

    # ----------------------------------------
    # Candidate answer
    # ----------------------------------------

    question_type = (
        gold["type"]
    )

    candidate_answers = (
        get_candidate_answers(
            extraction_output,
            question_type,
        )
    )

    expected_answer = (
        gold.get(
            "expected_answer",
            "",
        )
    )

    # ----------------------------------------
    # Answer Correctness
    # ----------------------------------------

    lexical = answer_correctness(
        expected_answer=expected_answer,
        candidate_answers=(
            candidate_answers
        ),
    )

    # ----------------------------------------
    # Semantic Correctness
    # ----------------------------------------

    semantic = (
        semantic_evaluator.best_similarity(
            expected_answer=(
                expected_answer
            ),
            candidate_answers=(
                candidate_answers
            ),
        )
    )

    # ----------------------------------------
    # Evidence / Citation Faithfulness
    # ----------------------------------------

    extraction = (
        extraction_output.get(
            "extraction"
        )
        or {}
    )

    evidence_items = (
        extraction_output.get(
            "evidence"
        )
        or []
    )

    citation_faithfulness = (
        calculate_citation_faithfulness(
            extraction=extraction,
            evidence_items=(
                evidence_items
            ),
            question_type=(
                question_type
            ),
        )
    )

    semantic_faithfulness = (
        calculate_semantic_faithfulness(
            extraction=extraction,
            evidence_items=(
                evidence_items
            ),
            question_type=(
                question_type
            ),
            semantic_evaluator=(
                semantic_evaluator
            ),
            threshold=(
                config.evaluation.faithfulness_threshold
            ),
        )
    )

    lexical_correct = (
        lexical[
            "score"
        ]
        >= config.evaluation.answer_threshold
    )

    semantic_score = semantic[
        "score"
    ]

    if semantic_score is None:

        semantic_correct = None

    else:

        semantic_correct = (
            semantic_score
            >= config.evaluation.semantic_threshold
        )

    return {

        "method": method,

        "retrieval": (
            retrieval_metrics
        ),

        "answer": {

            "expected_answer": (
                expected_answer
            ),

            "candidate_answers": (
                candidate_answers
            ),

            "answer_correctness": (
                lexical["score"]
            ),

            "answer_correct": (
                lexical_correct
            ),

            "best_lexical_answer": (
                lexical[
                    "best_answer"
                ]
            ),

            "semantic_correctness": (
                semantic_score
            ),

            "semantic_correct": (
                semantic_correct
            ),

            "best_semantic_answer": (
                semantic[
                    "best_answer"
                ]
            ),
        },

        "faithfulness": {

            "citation_faithfulness": (
                citation_faithfulness
            ),

            "semantic_faithfulness": (
                semantic_faithfulness
            ),
        },
    }


def evaluate_runs(runs_dir: str | Path, config: AppConfig, *, output=None,
                  semantic_evaluator=None) -> dict:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"No evaluation runs at {runs_dir}; run codal-rag batch")
    semantic = semantic_evaluator or SemanticEvaluator(config.evaluation.semantic_model,
                                enabled=config.evaluation.semantic_enabled)
    manifest_path = runs_dir / 'run.json'
    manifest = load_json(manifest_path) if manifest_path.exists() else None
    if manifest:
        if not manifest.get('complete'):
            raise ValueError('Batch run is incomplete; finish or rerun the batch before evaluation')
        ids = manifest.get('question_ids', [])
        if not ids or any(not isinstance(q, str) or '/' in q or '\\' in q or q in {'.','..'} for q in ids):
            raise ValueError('Invalid batch question identifiers')
        directories = [runs_dir / q for q in ids]
    else:
        # Read-only compatibility with Demo's Q*/gold,retrieval,baseline,proposed JSON layout.
        directories = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if not directories:
        raise ValueError('No question runs found')
    detailed, provenance, seen = [], {}, set()
    for directory in directories:
        artifacts = {name: load_json(directory / f'{name}.json')
                     for name in ('gold', 'retrieval', 'baseline', 'proposed')}
        gold = validate_question(artifacts['gold'])
        if gold['id'] in seen:
            raise ValueError('Duplicate question ids in evaluation runs')
        seen.add(gold['id'])
        retrieval = artifacts['retrieval']
        if retrieval.get('original_query', retrieval.get('query')) != gold['question']:
            raise ValueError(f"Retrieval query differs from gold question {gold['id']}")
        if retrieval.get('top_k', 0) < config.top_k:
            raise ValueError('Saved retrieval top_k is smaller than evaluation top_k; rerun retrieval')
        row = {key: gold[key] for key in ('id', 'question', 'type')}
        for method in ('baseline', 'proposed'):
            extraction = artifacts[method]
            if extraction.get('query') != gold['question']:
                raise ValueError(f"Extraction query differs from gold question {gold['id']}")
            # Evidence IDs alone do not establish provenance: verify saved document/chunk/text.
            hits = select_retrieval_results(retrieval, method)
            chunks = {str(hit['chunk'].get('chunk_id')): hit['chunk'] for hit in hits}
            for evidence in extraction.get('evidence', []):
                chunk = chunks.get(str(evidence.get('chunk_id')))
                if (not chunk or evidence.get('document_id') != chunk.get('document_id') or
                    evidence.get('text') != (chunk.get('text') or chunk.get('content') or '').strip()):
                    raise ValueError('Saved extraction evidence is not traceable to retrieved chunks')
            row[method] = evaluate_method(method, gold, retrieval, extraction, semantic, config)
        detailed.append(row)
        provenance[gold['id']] = {name: digest(value) for name, value in artifacts.items()}
    report = {'schema_version': 1, 'configuration': config.public_dict(),
              'semantic_evaluator': {'available': semantic.available, 'model': semantic.model_name,
                                     'reason': semantic.reason},
              'metric_notes': {'citation_faithfulness': 'Structural validity of all cited IDs, not entailment',
                               'semantic_correctness': 'Embedding similarity, not proof of correctness',
                               'semantic_faithfulness': 'Thresholded similarity proxy, not entailment',
                               'precision_denominator': 'min(k, number of returned results), as in Demo'},
              'overall': {m: aggregate_method(detailed, m, config.top_k) for m in ('baseline','proposed')},
              'by_type': aggregate_by_type(detailed, config.top_k),
              'per_question': detailed, 'artifact_sha256': provenance,
              'run_configuration': manifest.get('configuration') if manifest else None}
    save_json(report, output or config.path('evaluation_results', 'report.json'))
    return report
