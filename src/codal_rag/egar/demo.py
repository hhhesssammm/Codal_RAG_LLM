"""Explicitly synthetic demonstration. The mock is never a production fallback."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile

from codal_rag.config import load_config
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.index import index_corpus
from codal_rag.pipelines.extract import extract_retrieval
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.retrieval.scope import RetrievalScope
from codal_rag.storage.json_io import save_json


QUESTION = 'آیا افزایش بدهی کوتاه مدت شرکت نمونه در 1403 نشانه فشار نقدینگی است؟'
DEBT = 'بدهی کوتاه مدت شرکت نمونه در سال 1403 افزایش یافته است.'
CASH = 'جریان نقد عملیاتی شرکت نمونه در سال 1403 کاهش یافته است.'
GAP = {'description': 'برای بررسی فشار نقدینگی، شاهد مکمل درباره توان ایجاد وجه نقد لازم است.',
       'query': 'جریان نقد عملیاتی شرکت نمونه در سال 1403', 'section_hints': ['جریان نقد عملیاتی']}


class SyntheticClient:
    """Deterministic fixture for exactly this demonstration, not a semantic model."""
    def generate_json(self, system_prompt, user_prompt):
        if system_prompt.startswith('You assess Persian'):
            request = json.loads(user_prompt)
            evidence = request['evidence']
            found_cash = any(CASH in e['text'] for e in evidence)
            by_id = {e['evidence_id']: e for e in evidence}
            reviews = []
            for level in ('explicit', 'implicit', 'latent'):
                for index, claim in enumerate(request['candidates'][level]):
                    reviews.append({'level': level, 'index': index, 'status': 'sufficient',
                        'addresses_question': True, 'coherent_unit': level != 'latent',
                        'individual_sufficient': level != 'latent',
                        'units': [{'evidence_id': eid, 'quote': quote, 'role': role}
                                  for eid in claim['evidence_ids']
                                  for quote, role in ((DEBT, 'short_term_obligations'), (CASH, 'operating_cash_generation'))
                                  if quote in by_id[eid]['text'] and (level == 'latent' or quote == DEBT)],
                        'joint_support': 'افزایش تعهدات و کاهش ایجاد وجه نقد، دو نشانه مکمل فشار احتمالی نقدینگی هستند.',
                        'justification': 'Synthetic mocked assessment; not independently validated financial research.',
                        'missing_evidence_requirements': [], 'contradiction_information': [], 'calculations': []})
            return {'question_status': 'sufficient' if found_cash and reviews else 'partially_sufficient',
                    'claims': reviews, 'missing_evidence_requirements': [] if found_cash else [GAP],
                    'contradiction_information': [], 'justification': 'Synthetic complementary-evidence demonstration.'}
        query = user_prompt.split('سؤال:\n', 1)[1].split('\nشواهد بازیابی‌شده:', 1)[0].strip()
        chunks = re.findall(r'\[(E\d+)\]\n(.*?)(?=\[E\d+\]\n|\Z)', user_prompt, flags=re.S)
        debt = next((eid for eid, text in chunks if DEBT in text), None)
        cash = next((eid for eid, text in chunks if CASH in text), None)
        result = {'query': query, 'explicit': [], 'implicit': [], 'latent': [],
                  'insufficient_evidence': False, 'notes': 'Synthetic mocked output only.'}
        if debt and cash:
            result['latent'] = [{'claim': 'هم‌زمانی افزایش بدهی کوتاه مدت و کاهش جریان نقد عملیاتی ممکن است نشانه فشار نقدینگی باشد.',
                'evidence_ids': list(dict.fromkeys([debt, cash])), 'basis': 'ترکیب تعهدات کوتاه مدت با توان ایجاد وجه نقد.',
                'confidence': 0.7, 'conceptual_labels': ['ریسک نقدینگی'],
                'disclaimer': 'این نتیجه یک استنباط است و اثبات درماندگی مالی نیست.'}]
        elif debt:
            result['explicit'] = [{'claim': DEBT, 'evidence_ids': [debt], 'basis': 'بیان مستقیم سند مصنوعی.',
                                   'confidence': 0.9, 'conceptual_labels': ['تعهدات مالی']}]
        else:
            result['insufficient_evidence'] = True
        return result


def run_demo(output):
    with tempfile.TemporaryDirectory(prefix='codal-egar-demo-') as tmp:
        root = Path(tmp)
        documents = root/'reports'
        documents.mkdir()
        (documents/'synthetic.md').write_text(f'# گزارش نمونه\n\n## تعهدات مالی\n\n{DEBT}\n\n## جریان نقد عملیاتی\n\n{CASH}\n', encoding='utf-8')
        config = load_config(environ={}, overrides={'project_root': root, 'top_k': 1,
            'embedding': {'backend': 'hash', 'model': 'smoke-hash-v1', 'tokenizer': 'whitespace',
                          'query_prefix': '', 'passage_prefix': ''},
            'retrieval': {'mode': 'hybrid'}, 'extraction': {'schema_version': 2},
            'egar': {'enabled': True, 'max_queries_per_iteration': 1}})
        chunk_documents(documents, config, metadata_by_id={'synthetic': {'company': 'شرکت نمونه', 'year': '1403'}})
        index_corpus(config)
        engine = DualRetrievalEngine(config=config)
        scope = RetrievalScope(companies=('شرکت نمونه',), years=('1403',))
        initial = engine.result_payload(engine.search(QUESTION, scope=scope))
        adaptive = extract_retrieval(initial, 'proposed', config, engine=engine, scope=scope,
                                     llm_client=SyntheticClient())
        single = extract_retrieval(initial, 'proposed', replace(config, egar=replace(config.egar, enabled=False)),
                                   llm_client=SyntheticClient())
        return save_json({'synthetic': True, 'scientific_result': False,
                          'services': 'Real FAISS + BM25/RRF, smoke hash embeddings, deterministic mocked generator/judge',
                          'initial_retrieval': initial, 'D': single, 'D+EGAR': adaptive}, output)
