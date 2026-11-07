import numpy as np
import pytest

from codal_rag.evaluation.lexical import token_f1, answer_correctness
from codal_rag.evaluation.retrieval import calculate_retrieval_metrics, is_relevant_chunk
from codal_rag.evaluation.faithfulness import calculate_citation_faithfulness, calculate_semantic_faithfulness
from codal_rag.evaluation.semantic import SemanticEvaluator
from codal_rag.evaluation.aggregate import safe_mean
from codal_rag.evaluation.dataset import load_questions, create_template
from codal_rag.storage.json_io import save_json


def test_lexical_multiset_and_normalization():
    assert token_f1('شركت كاهش', 'شرکت کاهش') == 1
    assert token_f1('a a b', 'a b b') == pytest.approx(2/3)
    assert token_f1('', '') == 0
    assert answer_correctness('a b', ['x','a b']) == {'score':1.0, 'best_answer':'a b'}


def test_retrieval_metrics_with_known_ranking():
    hits = [{'chunk': {'document_id':'d', 'text':text, 'chunk_id':str(i)}}
            for i, text in enumerate(['other', 'alpha evidence', 'beta evidence'])]
    gold = [{'document_id':'d','text':'alpha evidence'}, {'document_id':'d','text':'beta evidence'}]
    metrics = calculate_retrieval_metrics(hits, gold, k=3)
    assert metrics['precision@3'] == pytest.approx(2/3)
    assert metrics['recall@3'] == 1
    assert metrics['mrr'] == 0.5
    assert metrics['evidence_hit'] == 1
    # Preserve and explicitly document Demo's available-results precision denominator.
    assert calculate_retrieval_metrics(hits[1:], gold, k=5)['precision@5'] == 1
    assert calculate_retrieval_metrics([], gold)['precision@5'] == 0
    assert calculate_retrieval_metrics(hits, [])['recall@5'] is None


def test_gold_document_must_match():
    gold = [{'document_id':'gold','text':'exact passage'}]
    assert not is_relevant_chunk({'document_id':'other','text':'exact passage'},gold)[0]
    assert not is_relevant_chunk({'text':'exact passage'},gold)[0]


def test_structural_ids_are_not_entailment():
    extraction = {'explicit': [{'claim':'Sales doubled', 'evidence_ids':['E1']},
                                {'claim':'Sales doubled','evidence_ids':['E1','E999']}]}
    evidence = [{'evidence_id':'E1', 'text':'Sales fell by half'}]
    structural = calculate_citation_faithfulness(extraction, evidence, 'explicit')
    assert structural['score'] == 0.5  # Only the first has entirely valid IDs.
    semantic = calculate_semantic_faithfulness(extraction, evidence, 'explicit', SemanticEvaluator())
    assert semantic['score'] is None


def test_semantic_proxy_negative_similarity_and_empty_candidates():
    class Model:
        def encode(self, texts, **kwargs):
            return np.array([[1.,0.],[-1.,0.]])
    semantic = SemanticEvaluator(model=Model())
    assert semantic.best_similarity('gold',['opposite'])['score'] == -1
    assert semantic.best_similarity('gold',[])['score'] == 0
    extraction = {'explicit':[{'claim':'x','evidence_ids':['E1']}]}
    result = calculate_semantic_faithfulness(extraction,[{'evidence_id':'E1','text':'y'}],'explicit',semantic)
    assert result['score'] == 0
    assert result['score_type'] == 'embedding_similarity_proxy_not_entailment'


def test_means_and_template(tmp_path):
    assert safe_mean([None, float('nan'), 1, 0]) == 0.5
    assert safe_mean([None]) is None
    path = create_template(tmp_path/'template.json')
    import json
    questions = json.loads(path.read_text())
    assert len(questions) == 50
    assert [sum(q['type']==t for q in questions) for t in ('explicit','implicit','latent')] == [20,15,15]
    with pytest.raises(ValueError, match='fill'):
        load_questions(path)


@pytest.mark.parametrize('identifier', ['../escape', '.', 'a/b', 'a\\b', ''])
def test_unsafe_dataset_ids(tmp_path, identifier):
    path = save_json([{'id':identifier, 'type':'explicit', 'question':'q', 'expected_answer':'a', 'gold_evidence':[]}],tmp_path/'q.json')
    with pytest.raises(ValueError, match='id'):
        load_questions(path)
