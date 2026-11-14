from dataclasses import replace
import hashlib

import pytest

from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.index import index_corpus
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.retrieval.scope import RetrievalScope
from codal_rag.retrieval.sparse import BM25, reciprocal_rank_fusion, terms


def record(i, text):
    return {'document_id': 'd', 'chunk_id': str(i), 'text': text}


def test_persian_bm25_and_empty_query():
    bm = BM25({'0': record(0, 'جريان نقد عملياتي ۱۴۰۳'), '1': record(1, 'بدهی فروش')})
    assert bm.search('جریان نقد عملیاتی 1403', 5)[0]['vector_id'] == 0
    assert bm.search('ناموجود', 5) == []
    assert bm.search('نقد', 5, {'1'}) == []
    assert terms('كاهش مي\u200cيابد') == terms('کاهش می یابد')


def test_rrf_uses_ranks_not_raw_scores_and_deduplicates():
    a = {'rank': 1, 'score': 99999, 'vector_id': 0, 'chunk': record(0, 'a')}
    b = {'rank': 2, 'score': -100, 'vector_id': 1, 'chunk': record(1, 'b')}
    ranked = reciprocal_rank_fusion({'dense': [a, b], 'bm25': [b, b]}, 2)
    assert ranked[0]['chunk']['chunk_id'] == '1'
    assert ranked[0]['score'] == pytest.approx(1/62+1/61)
    assert ranked[0]['retrieval_ranks'] == {'dense': 2, 'bm25': 1}


@pytest.mark.parametrize('mode', ['dense', 'bm25', 'hybrid'])
def test_shared_retrieval_settings_scope_and_read_only(corpus, smoke_config, tokenizer, mode):
    cfg = replace(smoke_config, retrieval=replace(smoke_config.retrieval, mode=mode))
    chunk_documents(corpus, cfg, tokenizer=tokenizer, metadata_by_id={
        'alpha/report': {'company': 'الف', 'year': '1403', 'symbol': 'الف', 'financial_period': 'سالانه'},
        'beta/report': {'company': 'ب', 'year': '1402'}})
    index_corpus(cfg)
    paths = list(cfg.path('vector_db').rglob('*'))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    engine = DualRetrievalEngine(config=cfg)
    scoped = engine.search('فروش شرکت', scope=RetrievalScope(companies=('الف',), years=('۱۴۰۳',)))
    for method in ('baseline', 'proposed'):
        hits = getattr(scoped, method)
        assert hits and all(h['chunk']['document_id'] == 'alpha/report' for h in hits)
        assert engine.search_method('فروش', method, 4, scope=RetrievalScope(companies=('ناشناخته',))) == []
        hinted = engine.search_method('فروش', method, 2, scope=RetrievalScope(companies=('الف',)), section_hints=['nonexistent'])
        assert all(h['chunk']['company'] == 'الف' for h in hinted)
    assert engine.result_payload(scoped)['scope']['companies'] == ['الف']
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def test_coordinates_are_copied_only_to_known_structured_blocks(corpus, smoke_config, tokenizer):
    from codal_rag.parsing.markdown import parse_markdown
    from codal_rag.normalization.persian import normalize_markdown
    doc = corpus/'alpha'/'report.md'
    blocks = parse_markdown(normalize_markdown(doc.read_text()))
    locations = {str(b.block_id): {'page': 2, 'row': 3} for b in blocks}
    records = chunk_documents(corpus, smoke_config, tokenizer=tokenizer, metadata_by_id={
        'alpha/report': {'publication_date': '1404/01/01', 'block_locations': locations}})
    for c in records['proposed']:
        if c['document_id'] == 'alpha/report':
            assert c['source_locations'][0]['page'] == 2
    assert all('source_locations' not in c for c in records['baseline'])
