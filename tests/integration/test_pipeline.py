from pathlib import Path
import importlib
import json
import pkgutil
import subprocess
import sys

import pytest
import codal_rag
from codal_rag.cli import main
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.index import index_corpus
from codal_rag.pipelines.batch import run_batch
from codal_rag.pipelines.evaluate import evaluate_runs
from codal_rag.storage.json_io import load_json, save_json
from tests.fakes import FakeLLM


def test_complete_pipeline_and_reproducible_evaluation(corpus, smoke_config, tokenizer):
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    index_corpus(smoke_config)
    question = 'علت کاهش فروش چیست؟'
    questions = [{'id':f'Q{i}', 'question':question, 'type':level,
                  'expected_answer':'فروش شرکت کاهش یافته است.',
                  'gold_evidence':[{'document_id':'alpha/report','text':'فروش شرکت در سال 1403 کاهش یافته است.'}]}
                 for i,level in enumerate(('explicit','implicit','latent'))]
    path = save_json(questions, smoke_config.project_root/'questions.json')
    llm = FakeLLM()
    runs = run_batch(path, smoke_config, llm_client=llm)
    assert len(llm.calls) == 6
    assert load_json(runs/'run.json')['complete']
    output = smoke_config.path('evaluation_results','report.json')
    report = evaluate_runs(runs, smoke_config, output=output)
    assert report['overall']['baseline']['questions'] == 3
    assert set(report['by_type']) == {'explicit','implicit','latent'}
    assert report['overall']['baseline']['answer']['mean_semantic_correctness'] is None
    assert report['overall']['baseline']['faithfulness']['citation_faithfulness'] == pytest.approx(1/3)
    assert report['per_question'][0]['baseline']['answer']['answer_correctness'] == 1
    before = output.read_bytes()
    evaluate_runs(runs, smoke_config, output=output)
    assert output.read_bytes() == before
    saved = load_json(runs/'Q0'/'proposed.json')
    saved['evidence'][0]['document_id'] = 'forged'
    save_json(saved,runs/'Q0'/'proposed.json')
    with pytest.raises(ValueError, match='traceable'):
        evaluate_runs(runs, smoke_config)


def test_failed_batch_cannot_be_evaluated(corpus, smoke_config, tokenizer):
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    index_corpus(smoke_config)
    path = save_json([{'id':'Q1','question':'q','type':'explicit','expected_answer':'a','gold_evidence':[]}],smoke_config.project_root/'q.json')
    with pytest.raises(ValueError):
        run_batch(path, smoke_config, llm_client=FakeLLM({'bad':'result'}))
    with pytest.raises(ValueError, match='incomplete'):
        evaluate_runs(smoke_config.path('evaluation_runs'),smoke_config)


def test_all_modules_import_without_models_or_credentials():
    imported = [importlib.import_module(info.name) for info in pkgutil.walk_packages(codal_rag.__path__, codal_rag.__name__+'.')]
    assert len(imported) >= 40


def test_installed_cli_outside_checkout(tmp_path):
    process = subprocess.run([sys.executable,'-m','codal_rag','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert process.returncode == 0, process.stderr
    assert 'evaluate' in process.stdout
    executable = Path(sys.executable).parent / ('codal-rag.exe' if sys.platform == 'win32' else 'codal-rag')
    process = subprocess.run([str(executable),'--root',str(tmp_path),'config'],cwd=tmp_path,capture_output=True,text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['project_root'] == str(tmp_path)


def test_cli_workflows(corpus, tmp_path, capsys):
    config_path = save_json({'project_root':str(tmp_path), 'top_k':2,
        'embedding':{'backend':'hash','model':'smoke-hash-v1','tokenizer':'whitespace','query_prefix':'','passage_prefix':''}},tmp_path/'config.yaml')
    args = ['--config',str(config_path)]
    assert main(args+['normalize','--input',str(corpus),'--output',str(tmp_path/'normalized')]) == 0
    assert main(args+['build','--input',str(tmp_path/'normalized')]) == 0
    assert main(args+['embed']) == 0
    assert main(args+['index','--from-embeddings']) == 0
    retrieval = tmp_path/'retrieval.json'
    assert main(args+['retrieve','--query','q','--output',str(retrieval)]) == 0
    response = save_json({'query':'q','explicit':[], 'implicit':[], 'latent':[], 'insufficient_evidence':True,'notes':''}, tmp_path/'response.json')
    assert main(args+['extract','--retrieval',str(retrieval),'--method','proposed','--output',str(tmp_path/'extraction.json'),'--response-file',str(response)]) == 0
    assert load_json(tmp_path/'extraction.json')['llm']['mocked']
    assert main(args+['eval-template','--output',str(tmp_path/'questions.json')]) == 0
    assert main(args+['batch','--questions',str(tmp_path/'questions.json')]) == 2
    assert main(args+['evaluate','--runs',str(tmp_path/'missing')]) == 2
    assert main(args+['retrieve','--query','']) == 2


def test_batch_and_evaluate_cli_success(corpus, smoke_config, tokenizer, monkeypatch):
    import codal_rag.pipelines.extract as extraction_pipeline
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    index_corpus(smoke_config)
    monkeypatch.setattr(extraction_pipeline, 'LLMClient', lambda config: FakeLLM())
    cfg_path = save_json(smoke_config.public_dict(), smoke_config.project_root/'config.yaml')
    question_path = save_json([{'id':'Q1','question':'چرا؟','type':'explicit',
        'expected_answer':'فروش شرکت کاهش یافته است.','gold_evidence':[]}],smoke_config.project_root/'questions.json')
    args = ['--config',str(cfg_path)]
    assert main(args+['batch','--questions',str(question_path)]) == 0
    assert main(args+['evaluate']) == 0
    report = load_json(smoke_config.path('evaluation_results','report.json'))
    assert report['overall']['baseline']['questions'] == 1
