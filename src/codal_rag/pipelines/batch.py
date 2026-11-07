from __future__ import annotations

from codal_rag.evaluation.dataset import load_questions
from codal_rag.pipelines.extract import extract_retrieval
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.storage.json_io import save_json, digest


def run_batch(questions_path, config, *, engine=None, llm_client=None, output_dir=None):
    questions = load_questions(questions_path)
    engine = engine or DualRetrievalEngine(config=config)
    directory = output_dir or config.path('evaluation_runs')
    from pathlib import Path
    directory = Path(directory)
    manifest = {'schema_version': 1, 'complete': False,
                'question_ids': [q['id'] for q in questions], 'dataset_sha256': digest(questions),
                'configuration': config.public_dict(), 'mocked_llm': llm_client is not None}
    save_json(manifest, directory / 'run.json')
    for question in questions:
        question_dir = directory / question['id']
        save_json(question, question_dir / 'gold.json')
        payload = engine.result_payload(engine.search(question['question'], config.top_k))
        save_json(payload, question_dir / 'retrieval.json')
        for method in ('baseline', 'proposed'):
            extract_retrieval(payload, method, config, llm_client=llm_client,
                              output=question_dir / f'{method}.json')
    manifest['complete'] = True
    save_json(manifest, directory / 'run.json')
    return directory
