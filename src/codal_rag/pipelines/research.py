"""Separate A-D / EGAR experiments with paired, auditable input settings."""
from __future__ import annotations

from dataclasses import replace
import importlib.metadata
from pathlib import Path
import platform
import time

from codal_rag.config import validate_config
from codal_rag.evaluation.benchmark import load_benchmark, public_question
from codal_rag.evaluation.research_metrics import evidence_metrics, score_question
from codal_rag.evaluation.statistics import paired_cluster_bootstrap
from codal_rag.extraction.prompts import SYSTEM_PROMPT, V2_EXTENSION
from codal_rag.pipelines.extract import extract_retrieval
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.retrieval.scope import RetrievalScope
from codal_rag.storage.json_io import digest, load_json, save_json


def package_fingerprint():
    root = Path(__file__).parents[1]
    return digest({p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in sorted(root.rglob('*.py'))})


ARMS = {'A': ('baseline', 'dense', False), 'B': ('proposed', 'dense', False),
        'C': ('baseline', 'hybrid', False), 'D': ('proposed', 'hybrid', False),
        'D+EGAR': ('proposed', 'hybrid', True)}


def run_experiments(dataset_path, config, *, arms=('A', 'B', 'C', 'D'), split='development',
                    allow_test=False, require_full=False, output_dir=None, engines=None,
                    llm_client=None, judge_client=None):
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('Select unique experiment arms A, B, C, D, D+EGAR')
    if split not in {'development', 'test'} or (split == 'test' and not allow_test):
        raise ValueError('Test split requires --allow-test after development settings are frozen')
    benchmark = load_benchmark(dataset_path, require_full=require_full)
    questions = [q for q in benchmark['questions'] if q['split'] == split]
    if not questions:
        raise ValueError('No questions in selected split')
    # Answers, levels, control labels and annotations are never passed to retrieval/generation.
    directory = Path(output_dir or config.path('research_runs'))
    configs = {a: validate_config(replace(config, retrieval=replace(config.retrieval, mode=ARMS[a][1]),
               extraction=replace(config.extraction, schema_version=2), egar=replace(config.egar, enabled=ARMS[a][2])))
               for a in arms}
    manifest = {'schema_version': 2, 'complete': False, 'arms': list(arms), 'split': split,
                'synthetic': benchmark.get('synthetic', False), 'dataset_sha256': digest(benchmark),
                'public_questions_sha256': digest([public_question(q) for q in questions]),
                'question_ids': [q['id'] for q in questions],
                'configurations': {a: c.public_dict() for a, c in configs.items()},
                'generation_prompt_sha256': digest(SYSTEM_PROMPT + V2_EXTENSION),
                'package_source_sha256': package_fingerprint(),
                'mocked_llm': llm_client is not None,
                'separately_injected_judge': judge_client is not None,
                'judge_model': getattr(judge_client or llm_client, 'model', config.llm.model),
                'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'faiss-cpu', 'PyYAML', 'codal-rag')}}}
    save_json(manifest, directory/'run.json')
    modes = sorted({ARMS[a][1] for a in arms})
    resources = {m: (engines or {}).get(m) for m in modes}
    for mode in modes:
        resources.setdefault(mode, None)
        if resources[mode] is None:
            resources[mode] = DualRetrievalEngine(config=next(c for c in configs.values() if c.retrieval.mode == mode))
        if resources[mode].config.retrieval.mode != mode:
            raise ValueError('Injected experiment retriever has incompatible settings')
    manifest['index_config'] = {m: e.index_config for m, e in resources.items()}
    for q in questions:
        root = directory/q['id']
        save_json(public_question(q), root/'question.json')
        inputs = {}
        for mode, engine in resources.items():
            started = time.monotonic()
            ranking = engine.result_payload(engine.search(q['question'], max(config.top_k, 10),
                                                         scope=RetrievalScope(**q['scope'])))
            elapsed = time.monotonic()-started
            payload = {**ranking, 'top_k': config.top_k,
                       'baseline': ranking['baseline'][:config.top_k],
                       'proposed': ranking['proposed'][:config.top_k]}
            save_json({'initial': payload, 'ranking_at_10': ranking}, root/f'{mode}_retrieval.json')
            inputs[mode] = (payload, elapsed)
        for arm in arms:
            method, mode, adaptive = ARMS[arm]
            payload, elapsed = inputs[mode]
            result = extract_retrieval(payload, method, configs[arm], llm_client=llm_client,
                engine=resources[mode], scope=RetrievalScope(**q['scope']), judge_client=judge_client)
            result['experiment'] = arm
            result['initial_retrieval_sha256'] = digest(payload)
            result['operational']['initial_retrieval_seconds'] = elapsed
            result['operational']['latency_seconds'] += elapsed
            save_json(result, root/f'{arm}.json')
    manifest['complete'] = True
    save_json(manifest, directory/'run.json')
    return directory


def _verify_trace(output, payload):
    method = output['method']
    initial = payload[method]
    egar = output.get('egar', {})
    if egar.get('enabled') and egar.get('initial_retrieval') != initial:
        raise ValueError('Initial adaptive retrieval does not match the saved original')
    candidates = initial + egar.get('newly_retrieved_evidence', [])
    chunks = {(h['chunk']['document_id'], h['chunk']['chunk_id']): h['chunk'] for h in candidates}
    for hit in output['final_retrieval']:
        c = hit['chunk']
        if chunks.get((c['document_id'], c['chunk_id'])) != c:
            raise ValueError('Final evidence was not present in a recorded retrieval')
    final = {(h['chunk']['document_id'], h['chunk']['chunk_id']): h['chunk'] for h in output['final_retrieval']}
    for e in output['evidence']:
        c = final.get((e['document_id'], e['chunk_id']))
        if c is None or e['text'] != (c.get('text') or c.get('content') or '').strip():
            raise ValueError('Evidence provenance mismatch')
    if output['initial_retrieval_sha256'] != digest(payload):
        raise ValueError('Initial retrieval hash mismatch')


def _aggregate(rows):
    keys = {k for r in rows for k, v in r['metrics'].items() if v is None or type(v) in (int, float)}
    return {k: {'mean': sum(values)/len(values) if values else None, 'n_questions': len(values),
                'n_companies': len({r['company_id'] for r in rows if r['metrics'].get(k) is not None})}
            for k in sorted(keys) for values in [[r['metrics'][k] for r in rows if type(r['metrics'].get(k)) in (int, float)]]
            }


def evaluate_experiments(runs_dir, dataset_path, *, adjudications_path=None, output=None,
                         comparisons=(('C', 'D'), ('D', 'D+EGAR')), resamples=10000, seed=42):
    root = Path(runs_dir)
    run = load_json(root/'run.json')
    if not run.get('complete') or run.get('schema_version') != 2:
        raise ValueError('Research run is incomplete or has an unsupported schema')
    benchmark = load_benchmark(dataset_path)
    if digest(benchmark) != run['dataset_sha256']:
        raise ValueError('Dataset differs from the experiment snapshot')
    questions = {q['id']: q for q in benchmark['questions']}
    judgments = load_json(adjudications_path) if adjudications_path else {}
    by_arm, artifact_hashes = {a: [] for a in run['arms']}, {}
    for qid in run['question_ids']:
        if qid not in questions or '/' in qid or '\\' in qid:
            raise ValueError('Invalid question in run manifest')
        q = questions[qid]
        for arm in run['arms']:
            method, mode, _ = ARMS[arm]
            retrieved = load_json(root/qid/f'{mode}_retrieval.json')
            result = load_json(root/qid/f'{arm}.json')
            if result['query'] != q['question'] or result['method'] != method or result['experiment'] != arm:
                raise ValueError('Experiment output identity mismatch')
            _verify_trace(result, retrieved['initial'])
            annotation = benchmark.get('annotations', {}).get(qid)
            metrics = score_question(q, annotation, result, judgments.get(qid, {}).get(arm))
            initial_metrics = evidence_metrics(retrieved['ranking_at_10'][method], (annotation or {}).get('evidence_sets', []))
            metrics['final_context_recall@5'] = metrics['recall@5']
            metrics['final_context_recall@10'] = metrics['recall@10']
            metrics['recall@5'], metrics['recall@10'] = initial_metrics['recall@5'], initial_metrics['recall@10']
            metrics['initial_complete_evidence_set_coverage'] = initial_metrics['complete_evidence_set_coverage']
            metrics['gold_validation_status'] = (annotation or {}).get('validation_status', 'unavailable')
            by_arm[arm].append({'id': qid, 'company_id': q['company_id'], 'type': q['type'], 'metrics': metrics})
            artifact_hashes[f'{qid}/{arm}'] = {'output': digest(result), 'retrieval': digest(retrieved)}
    report = {'schema_version': 2, 'run': run, 'artifact_sha256': artifact_hashes,
              'adjudications_sha256': digest(judgments) if judgments else None,
              'metric_notes': {'recall@5/10': 'Initial ranking, same for D and D+EGAR; no final context expansion implied',
                 'coverage': 'Final context contains every exact normalized gold passage in at least one alternative set',
                 'evidence_precision': 'Fraction of final chunks containing an exact normalized gold passage',
                 'correct_document_retrieval_rate': 'Fraction of final chunks from documents in the gold evidence union',
                 'semantic_evidence_support': 'Supplied human/LLM adjudication, separately labeled; null when unavailable',
                 'automatic_judge_support_rate': 'Controller judgment; not independent factual verification',
                 'retrieved_token_count': 'UTF-8 byte upper bound; actual inference usage is separately nullable',
                 'claim_metrics': 'One-to-one adjudicated gold matches, not lexical or embedding similarity'},
              'per_arm': by_arm, 'overall': {a: _aggregate(rows) for a, rows in by_arm.items()},
              'by_level': {a: {level: _aggregate([r for r in rows if r['type'] == level])
                             for level in ('explicit', 'implicit', 'latent')} for a, rows in by_arm.items()},
              'paired_comparisons': []}
    # Refuse to pool verified-human and LLM judgments into a single reported metric.
    bases = {r['metrics']['evidence_judgment_basis'] for rows in by_arm.values() for r in rows}
    if {'human_verified', 'llm_judged'} <= bases:
        raise ValueError('Human and LLM adjudications must be evaluated in separate reports')
    for left, right in comparisons:
        if left not in by_arm or right not in by_arm:
            continue
        lc, rc = run['configurations'][left], run['configurations'][right]
        for key in ('embedding', 'llm', 'retrieval', 'extraction', 'chunking', 'top_k', 'random_seed'):
            if lc[key] != rc[key]:
                raise ValueError(f'Unfair paired comparison: {key} differs')
        stats = {}
        for level in ('all', 'explicit', 'implicit', 'latent'):
            a = [r for r in by_arm[left] if level == 'all' or r['type'] == level]
            b = [r for r in by_arm[right] if level == 'all' or r['type'] == level]
            stats[level] = [paired_cluster_bootstrap(a, b, metric, resamples=resamples, seed=seed)
                for metric in ('recall@5', 'recall@10', 'complete_evidence_set_coverage', 'claim_f1',
                               'latent_claim_validity', 'conceptual_label_f1', 'unsupported_claim_rate',
                               'correct_abstention_rate', 'latency_seconds', 'llm_calls')]
        report['paired_comparisons'].append({'left': left, 'right': right, 'results': stats})
    if output is not None:
        save_json(report, output)
    return report
