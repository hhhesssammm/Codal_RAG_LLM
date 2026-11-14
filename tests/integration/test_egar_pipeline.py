from dataclasses import replace
from pathlib import Path
import json

import pytest

from codal_rag.egar.demo import run_demo, SyntheticClient, QUESTION, DEBT, CASH
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.index import index_corpus
from codal_rag.pipelines.research import run_experiments, evaluate_experiments
from codal_rag.storage.json_io import save_json, load_json
from codal_rag.cli import main


def test_reproducible_real_faiss_synthetic_demo(tmp_path):
    data = load_json(run_demo(tmp_path/'demo.json'))
    assert data['synthetic'] and not data['scientific_result']
    assert data['D+EGAR']['egar']['iteration_count'] == 1
    assert data['D+EGAR']['egar']['stop_reason'] == 'sufficient'
    assert data['D+EGAR']['extraction']['latent']
    assert not data['D']['extraction']['latent']
    assert data['D+EGAR']['operational']['llm_calls'] == 4


def prepare(tmp_path, smoke_config):
    docs = tmp_path/'input'; docs.mkdir()
    (docs/'report.md').write_text(f'# گزارش\n\n## تعهدات مالی\n\n{DEBT}\n\n## جریان نقد عملیاتی\n\n{CASH}', encoding='utf-8')
    cfg = replace(smoke_config, top_k=1, chunking=replace(smoke_config.chunking, max_tokens=100, baseline_overlap=10))
    chunk_documents(docs, cfg, metadata_by_id={'report': {'company': 'شرکت نمونه', 'year': '1403'}})
    index_corpus(cfg)
    data = {'schema_version': 2, 'synthetic': True,
        'documents': [{'document_id': 'report', 'company_id': 'synthetic', 'company': 'شرکت نمونه',
                       'year': '1403', 'industry': 'synthetic', 'split': 'development'}],
        'questions': [{'id': 'Q1', 'question': QUESTION, 'company_id': 'synthetic', 'split': 'development',
                       'type': 'latent', 'answerable': True, 'scope': {'document_ids': ['report']}}],
        'annotations': {'Q1': {'validation_status': 'synthetic', 'claims': [
            {'id': 'C1', 'claim': 'احتمال فشار نقدینگی', 'inference_level': 'latent', 'conceptual_labels': ['ریسک نقدینگی']}],
            'evidence_sets': [[{'document_id': 'report', 'text': DEBT}, {'document_id': 'report', 'text': CASH}]]}}}
    return cfg, save_json(data, tmp_path/'questions.json')


def test_all_primary_arms_and_ablation_use_identical_initial_settings(tmp_path, smoke_config):
    cfg, dataset = prepare(tmp_path, smoke_config)
    runs = run_experiments(dataset, cfg, arms=['A', 'B', 'C', 'D', 'D+EGAR'], llm_client=SyntheticClient())
    manifest = load_json(runs/'run.json')
    assert manifest['complete']
    for arm in ['A','B','C','D']:
        output = load_json(runs/'Q1'/f'{arm}.json')
        assert not output['egar']['enabled']
    d, adaptive = [load_json(runs/'Q1'/f'{arm}.json') for arm in ('D', 'D+EGAR')]
    assert d['initial_retrieval_sha256'] == adaptive['initial_retrieval_sha256']
    assert d['context_budget']['limit'] == adaptive['context_budget']['limit']
    report = evaluate_experiments(runs, dataset, resamples=50)
    assert len(report['paired_comparisons']) == 2
    assert report['per_arm']['D'][0]['metrics']['recall@10'] == report['per_arm']['D+EGAR'][0]['metrics']['recall@10']
    assert report['per_arm']['D+EGAR'][0]['metrics']['claim_f1'] is None
    adaptive['evidence'][0]['text'] = 'fabricated'
    save_json(adaptive, runs/'Q1'/'D+EGAR.json')
    with pytest.raises(ValueError, match='provenance'):
        evaluate_experiments(runs, dataset, resamples=10)


def test_cli_research_demo_template_and_split_guard(tmp_path, smoke_config):
    cfg, dataset = prepare(tmp_path, smoke_config)
    assert main(['egar-demo', '--output', str(tmp_path/'demo.json')]) == 0
    assert main(['codalbench-template', '--output', str(tmp_path/'template.json')]) == 0
    assert main(['validate-benchmark', '--dataset', str(dataset)]) == 0
    assert main(['validate-benchmark', '--dataset', str(dataset), '--full-protocol']) == 2
    with pytest.raises(ValueError, match='allow-test'):
        run_experiments(dataset, cfg, split='test', llm_client=SyntheticClient())


def test_streamlit_egar_flow(tmp_path, smoke_config, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    import codal_rag.pipelines.extract as pipeline
    cfg, _ = prepare(tmp_path, smoke_config)
    cfg = replace(cfg, egar=replace(cfg.egar, enabled=True), retrieval=replace(cfg.retrieval, mode='hybrid'))
    path = save_json(cfg.public_dict(), tmp_path/'config.json')
    monkeypatch.setenv('CODAL_CONFIG', str(path))
    monkeypatch.setattr(pipeline, 'LLMClient', lambda cfg: SyntheticClient())
    app = AppTest.from_file(str(Path(__file__).parents[2]/'apps'/'streamlit_app.py')).run()
    next(x for x in app.text_input if x.label == 'سؤال کاربر').set_value(QUESTION)
    next(x for x in app.button if x.label == 'جست‌وجو').click()
    app.run(timeout=10)
    assert not app.exception and not app.error
    assert any(x.value == 'نتیجهٔ بررسی شواهد' for x in app.subheader)
    assert 'ریسک نقدینگی' in json.dumps([x.value for x in app.caption], ensure_ascii=False)
