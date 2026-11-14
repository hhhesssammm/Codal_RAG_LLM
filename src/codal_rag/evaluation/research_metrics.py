"""Research metrics: gold passage coverage and separately supplied adjudications."""
from __future__ import annotations

from codal_rag.retrieval.scope import canonical


ERROR_CATEGORIES = {
    'incorrect_document': 'retrieval', 'incorrect_period': 'retrieval',
    'evidence_not_retrieved': 'retrieval', 'incomplete_multi_evidence': 'retrieval',
    'duplicate_as_complementary': 'assessment', 'incorrect_numerical_reasoning': 'generation',
    'unsupported_causality': 'generation', 'contradiction_ignored': 'assessment',
    'incorrect_conceptual_label': 'generation', 'unsupported_claim': 'generation',
    'incorrect_citation': 'generation', 'unnecessary_supplementary_retrieval': 'assessment',
    'evidence_not_used': 'generation',
}


def claims(output):
    ex = output.get('extraction', {})
    return {f'{level}:{i}': {**claim, 'inference_level': level} for level in ('explicit', 'implicit', 'latent')
            for i, claim in enumerate(ex.get(level, []))}


def matches(gold, hit):
    c = hit.get('chunk', hit)
    return c.get('document_id') == gold['document_id'] and canonical(gold['text']) in canonical(c.get('text') or c.get('content') or '')


def evidence_metrics(hits, evidence_sets):
    if not evidence_sets:
        return {k: None for k in ('recall@5', 'recall@10', 'complete_evidence_set_coverage',
                                  'evidence_precision', 'correct_document_retrieval_rate')}
    def recall(k):
        return max(sum(any(matches(g, h) for h in hits[:k]) for g in s) / len(s) for s in evidence_sets)
    gold = [g for s in evidence_sets for g in s]
    documents = {g['document_id'] for g in gold}
    return {'recall@5': recall(5), 'recall@10': recall(10),
            'complete_evidence_set_coverage': float(any(all(any(matches(g, h) for h in hits) for g in s) for s in evidence_sets)),
            'evidence_precision': sum(any(matches(g, h) for g in gold) for h in hits) / max(len(hits), 1),
            'correct_document_retrieval_rate': sum(h['chunk']['document_id'] in documents for h in hits) / max(len(hits), 1)}


def score_question(question, annotation, output, adjudication=None):
    predictions = claims(output)
    final_hits = output.get('final_retrieval', [])
    evidence = {e['evidence_id']: e for e in output.get('evidence', [])}
    gold = {c['id']: c for c in (annotation or {}).get('claims', [])}
    score = evidence_metrics(final_hits, (annotation or {}).get('evidence_sets', []))
    score.update({key: None for key in ('claim_precision', 'claim_recall', 'claim_f1',
         'latent_claim_validity', 'conceptual_label_f1', 'semantic_evidence_support',
         'unsupported_claim_rate', 'contradiction_detection', 'correct_abstention_rate')})
    score['citation_validity'] = (sum(bool(c.get('evidence_ids')) and all(e in evidence for e in c['evidence_ids'])
                                             for c in predictions.values()) / len(predictions)) if predictions else None
    abstained = bool(output['extraction'].get('insufficient_evidence')) and not predictions
    if not question['answerable']:
        score['correct_abstention_rate'] = float(abstained)
    score['abstained'] = float(abstained)
    score['evidence_judgment_basis'] = 'unavailable'
    egar = output.get('egar', {})
    score['supplementary_iterations'] = egar.get('iteration_count', 0)
    # Controller assessments are not independent gold judgments.
    auto = egar.get('final_sufficiency_decision', {}).get('claims', [])
    score['automatic_judge_support_rate'] = sum(a['sufficiency_status'] == 'sufficient' for a in auto) / len(auto) if auto else None
    if adjudication is not None:
        basis = adjudication.get('assessment_method')
        if basis not in {'human_verified', 'llm_judged'}:
            raise ValueError('Adjudication must disclose human_verified or llm_judged')
        rows = adjudication.get('claims', [])
        if not isinstance(rows, list) or len(rows) != len(predictions) or {r.get('prediction_id') for r in rows} != set(predictions):
            raise ValueError('Adjudication must assess every prediction exactly once')
        matched_gold, tp, label_tp, label_fp = set(), 0, 0, 0
        support, latent = [], []
        for r in rows:
            if type(r.get('correct')) is not bool or type(r.get('semantic_support')) is not bool:
                raise ValueError('Adjudication requires boolean correctness and semantic_support')
            p = predictions[r['prediction_id']]
            labels = {canonical(v) for v in p.get('conceptual_labels', [])}
            support.append(r['semantic_support'])
            if p['inference_level'] == 'latent':
                if type(r.get('latent_valid')) is not bool:
                    raise ValueError('Latent claims require a separate validity judgment')
                latent.append(r['latent_valid'])
            if r['correct']:
                gid = r.get('gold_claim_id')
                if gid not in gold or gid in matched_gold or gold[gid]['inference_level'] != p['inference_level']:
                    raise ValueError('Correct predictions require a one-to-one gold match at the same level')
                matched_gold.add(gid)
                tp += 1
                reference_labels = {canonical(v) for v in gold[gid]['conceptual_labels']}
                label_tp += len(labels & reference_labels)
                label_fp += len(labels - reference_labels)
            else:
                label_fp += len(labels)
        if question['answerable'] and annotation is not None:
            precision, recall = tp / max(len(predictions), 1), tp / max(len(gold), 1)
            score.update(claim_precision=precision, claim_recall=recall,
                         claim_f1=2*precision*recall/(precision+recall) if precision+recall else 0.0)
            label_fn = sum(len({canonical(v) for v in c['conceptual_labels']}) for c in gold.values()) - label_tp
            denom = 2*label_tp + label_fp + label_fn
            score['conceptual_label_f1'] = 2*label_tp/denom if denom else None
        score['semantic_evidence_support'] = sum(support)/len(support) if support else None
        score['unsupported_claim_rate'] = 1-score['semantic_evidence_support'] if support else None
        score['latent_claim_validity'] = sum(latent)/len(latent) if latent else None
        score['evidence_judgment_basis'] = basis
        if type(adjudication.get('contradiction_present')) is bool:
            flagged = egar.get('final_sufficiency_decision', {}).get('sufficiency_status') == 'contradictory'
            score['contradiction_detection'] = float(flagged == adjudication['contradiction_present'])
        errors = adjudication.get('errors', [])
        if not isinstance(errors, list) or any(e not in ERROR_CATEGORIES for e in errors):
            raise ValueError('Unknown failure-analysis category')
        score['errors'] = [{"category": e, "stage": ERROR_CATEGORIES[e]} for e in errors]
    for key in ('latency_seconds', 'llm_calls', 'retrieved_token_count', 'total_inference_tokens',
                'estimated_cost', 'additional_retrieval_token_upper_bound', 'final_evidence_tokens'):
        score[key] = output.get('operational', {}).get(key)
    score['predicted_claim_count'] = len(predictions)
    score['gold_claim_count'] = len(gold) if annotation is not None else None
    return score
