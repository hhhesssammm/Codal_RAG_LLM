from copy import deepcopy

import pytest

from codal_rag.evaluation.benchmark import create_benchmark_template, validate_benchmark, public_question
from codal_rag.evaluation.research_metrics import evidence_metrics, score_question
from codal_rag.evaluation.statistics import paired_cluster_bootstrap
from codal_rag.storage.json_io import load_json
from tests.unit.test_egar import extraction, claim, hit


def benchmark():
    return {'schema_version': 2, 'synthetic': True, 'documents': [
        {'document_id': 'report', 'company_id': 'alpha', 'company': 'الف', 'year': '1403', 'industry': 'synthetic', 'split': 'development'}],
        'questions': [{'id': 'Q1', 'company_id': 'alpha', 'split': 'development', 'question': 'q',
                       'answerable': True, 'type': 'latent', 'scope': {'document_ids': ['report']}}],
        'annotations': {'Q1': {'validation_status': 'synthetic', 'claims': [
            {'id': 'G1', 'claim': 'فشار نقدینگی', 'inference_level': 'latent', 'conceptual_labels': ['ریسک نقدینگی']}],
            'evidence_sets': [[{'document_id': 'report', 'text': 'بدهی بیشتر'}, {'document_id': 'report', 'text': 'نقد کمتر'}]]}}}


def test_planned_260_template_contains_no_invented_annotations(tmp_path):
    data = load_json(create_benchmark_template(tmp_path/'plan.json'))
    assert len(data['questions']) == 260
    assert sum(q['answerable'] for q in data['questions']) == 240
    assert sum(q['split'] == 'development' for q in data['questions']) == 60
    assert data['annotations'] == {} and data['documents'] == []
    with pytest.raises(ValueError):
        validate_benchmark(data)


def test_company_split_and_gold_never_passed_to_generation():
    b = benchmark()
    assert validate_benchmark(b) == b
    assert set(public_question(b['questions'][0])) == {'id', 'question', 'scope'}
    b['questions'].append({**b['questions'][0], 'id': 'Q2', 'split': 'test'})
    with pytest.raises(ValueError, match='split|leakage'):
        validate_benchmark(b)
    with pytest.raises(ValueError, match='60/20/5'):
        validate_benchmark(benchmark(), require_full=True)


def test_complete_alternative_evidence_sets_precision_and_wrong_document():
    sets = [[{'document_id': 'report', 'text': 'بدهی بیشتر'}, {'document_id': 'report', 'text': 'نقد کمتر'}],
            [{'document_id': 'report', 'text': 'شاهد جایگزین کامل'}]]
    first = evidence_metrics([hit('بدهی بیشتر')], sets)
    assert first['recall@5'] == .5 and first['complete_evidence_set_coverage'] == 0
    assert evidence_metrics([hit('شاهد جایگزین کامل')], sets)['complete_evidence_set_coverage'] == 1
    wrong = hit('بدهی بیشتر'); wrong['chunk']['document_id'] = 'wrong'
    assert evidence_metrics([wrong], sets)['correct_document_retrieval_rate'] == 0
    assert evidence_metrics([wrong], sets)['recall@10'] == 0


def test_claim_metrics_require_independent_judgments_and_one_to_one_matches():
    b = benchmark(); q, ann = b['questions'][0], b['annotations']['Q1']
    output = {'extraction': extraction('latent', [claim('فشار نقدینگی', conceptual_labels=['ریسک نقدینگی'])]),
              'evidence': [{'evidence_id': 'E1'}], 'final_retrieval': []}
    result = score_question(q, ann, output)
    assert result['citation_validity'] == 1 and result['semantic_evidence_support'] is None
    assert result['claim_f1'] is None
    judgment = {'assessment_method': 'human_verified', 'claims': [{'prediction_id': 'latent:0',
        'gold_claim_id': 'G1', 'correct': True, 'semantic_support': True, 'latent_valid': True}]}
    result = score_question(q, ann, output, judgment)
    assert result['claim_f1'] == result['conceptual_label_f1'] == result['latent_claim_validity'] == 1
    duplicate = deepcopy(output)
    duplicate['extraction']['latent'].append(claim('فشار نقدینگی'))
    judgment['claims'].append({**judgment['claims'][0], 'prediction_id': 'latent:1'})
    with pytest.raises(ValueError, match='one-to-one'):
        score_question(q, ann, duplicate, judgment)


def test_correct_abstention_and_error_taxonomy():
    q = {**benchmark()['questions'][0], 'answerable': False}
    output = {'extraction': extraction(items=[]), 'evidence': [], 'final_retrieval': []}
    result = score_question(q, {'claims': [], 'evidence_sets': []}, output,
        {'assessment_method': 'human_verified', 'claims': [], 'errors': ['incomplete_multi_evidence']})
    assert result['correct_abstention_rate'] == 1
    assert result['errors'][0]['stage'] == 'retrieval'


def test_cluster_bootstrap_keeps_pairs_and_reports_sample_counts():
    def rows(offset):
        return [{'id': str(i), 'company_id': 'A' if i < 3 else 'B', 'type': 'latent',
                 'metrics': {'f1': .1*i+offset}} for i in range(4)]
    a, b = rows(0), rows(.2)
    result = paired_cluster_bootstrap(a, b, 'f1', resamples=10000)
    assert result['mean_difference'] == pytest.approx(.2)
    assert result['ci95'] == pytest.approx([.2, .2])
    assert result['n_questions'] == 4 and result['n_companies'] == 2
    assert paired_cluster_bootstrap(a[:1], b[:1], 'f1')['ci95'] is None
    with pytest.raises(ValueError, match='identical questions'):
        paired_cluster_bootstrap(a, b[:1], 'f1')
