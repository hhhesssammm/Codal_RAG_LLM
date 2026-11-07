from dataclasses import replace
import numpy as np
import pytest

from codal_rag.indexing.embeddings import EmbeddingModel, HashEmbeddingModel
from codal_rag.indexing.faiss_store import FaissVectorStore
from codal_rag.indexing.representations import get_baseline_embedding_text, get_proposed_embedding_text
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.embed import embed_corpus, load_chunks
from codal_rag.pipelines.index import index_corpus
from codal_rag.retrieval.engine import DualRetrievalEngine
from codal_rag.storage.json_io import load_json, save_json
from tests.fakes import FakeSentenceModel


def test_prefixes_normalization_and_shape():
    backend = FakeSentenceModel()
    model = EmbeddingModel('intfloat/multilingual-e5-base', model=backend)
    assert model.encode_passages(['  متن فارسی ']).dtype == np.float32
    assert backend.calls[-1][0] == ['passage: متن فارسی']
    assert backend.calls[-1][1]['normalize_embeddings'] is True
    assert backend.calls[-1][1]['prompt'] == ''
    model.encode_query(' پرسش ')
    assert backend.calls[-1][0] == ['query: پرسش']
    assert model.encode_passages([]).shape == (0, 3)
    with pytest.raises(ValueError, match='context'):
        model.encode_passages(['word ' * 520])
    with pytest.raises(ValueError, match='context'):
        model.encode_query('word ' * 520)


def test_custom_prefixes():
    backend = FakeSentenceModel()
    model = EmbeddingModel('custom', query_prefix='', passage_prefix='', model=backend)
    model.encode_query('متن')
    assert backend.calls[-1][0] == ['متن']


def test_representations():
    record = {'content': 'body', 'heading_path': ['A', 'B']}
    assert get_baseline_embedding_text(record) == 'body'
    assert get_proposed_embedding_text(record) == 'A > B\n\nbody'
    assert get_proposed_embedding_text({**record, 'retrieval_text': 'exact'}) == 'exact'
    with pytest.raises(ValueError):
        get_baseline_embedding_text({})


def test_faiss_exact_roundtrip_and_append(tmp_path):
    store = FaissVectorStore(2)
    records = [{'chunk_id': 'a'}, {'chunk_id': 'b'}]
    store.add(np.array([[1,0], [0,1]]), records)
    store.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path)
    assert loaded.search([1,0], 10)[0] == {'rank': 1, 'score': 1.0, 'vector_id': 0, 'chunk': records[0]}
    loaded.add(np.array([[0.6,0.8]]), [{'chunk_id': 'c'}])
    assert loaded.next_id == 3
    assert loaded.search([0,1], 2)[1]['chunk']['chunk_id'] == 'c'


@pytest.mark.parametrize('vectors,records', [([[1,2,3]], [{}]), ([[float('nan'),0]], [{}]), ([[1,0]], []), ([1,0], [{}])])
def test_bad_vectors(vectors, records):
    with pytest.raises(ValueError):
        FaissVectorStore(2).add(np.array(vectors), records)


def test_empty_and_invalid_queries():
    store = FaissVectorStore(2)
    assert store.search([1,0]) == []
    for vector, k in [([1,0],0), ([1],1), ([[1,0],[0,1]],1), ([float('inf'),0],1)]:
        with pytest.raises(ValueError):
            store.search(vector,k)


@pytest.mark.parametrize('filename', ['metadata.json', 'db_info.json', 'config.json', 'index.faiss'])
def test_corrupt_index_rejected(tmp_path, filename):
    store = FaissVectorStore(2)
    store.add(np.array([[1,0]]), [{'chunk_id':'a'}])
    store.save(tmp_path)
    path = tmp_path / filename
    if path.suffix == '.json':
        payload = load_json(path)
        payload['unexpected'] = True
        if filename == 'db_info.json':
            payload['dimension'] = 3
        save_json(payload, path)
    else:
        path.write_bytes(b'broken')
    with pytest.raises(ValueError):
        FaissVectorStore.load(tmp_path)


def test_both_indexes_same_query_vector(corpus, smoke_config, tokenizer):
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    model = HashEmbeddingModel()
    counts = index_corpus(smoke_config, embedding_model=model)
    assert all(count > 0 for count in counts.values())
    engine = DualRetrievalEngine(config=smoke_config, embedding_model=model)
    calls = []
    original = model.encode_query
    model.encode_query = lambda text: calls.append(text) or original(text)
    result = engine.search('فروش شركت در ۱۴۰۳', 2)
    assert len(calls) == 1
    assert 'شرکت' in result.normalized_query and '1403' in result.normalized_query
    assert len(result.baseline) == len(result.proposed) == 2
    for hits in (result.baseline, result.proposed):
        assert [hit['rank'] for hit in hits] == [1,2]
        assert all(hit['chunk']['document_id'] for hit in hits)
    with pytest.raises(ValueError):
        engine.search(' ')
    with pytest.raises(ValueError):
        engine.search('متن', 0)


def test_cache_and_config_incompatibility(corpus, smoke_config, tokenizer):
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    embed_corpus(smoke_config)
    index_corpus(smoke_config, from_embeddings=True)
    bad = replace(smoke_config, embedding=replace(smoke_config.embedding, query_prefix='different'))
    with pytest.raises(ValueError, match='mismatch'):
        DualRetrievalEngine(config=bad)
    with pytest.raises(ValueError, match='dimension'):
        DualRetrievalEngine(config=smoke_config, embedding_model=HashEmbeddingModel(32))
    path = smoke_config.path('chunks','baseline','corpus.json')
    records = load_json(path)
    records[0]['content'] += ' new'
    save_json(records, path)
    with pytest.raises(ValueError, match='Stale'):
        index_corpus(smoke_config, from_embeddings=True)


def test_missing_index_and_duplicate_chunks(tmp_path, smoke_config):
    with pytest.raises(FileNotFoundError, match='index'):
        DualRetrievalEngine(config=smoke_config)
    path = save_json([{'chunk_id':'x','document_id':'d'}]*2, tmp_path/'chunks.json')
    with pytest.raises(ValueError, match='Duplicate'):
        load_chunks(path)


def test_same_document_ids_with_changed_contents_are_not_fair(corpus, smoke_config, tokenizer):
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    index_corpus(smoke_config)
    path = corpus / 'alpha' / 'report.md'
    path.write_text(path.read_text() + '\n\nNew material changes the corpus.\n')
    baseline = replace(smoke_config, mode='baseline')
    chunk_documents(corpus, baseline, tokenizer=tokenizer)
    index_corpus(baseline)
    with pytest.raises(ValueError, match='source contents'):
        DualRetrievalEngine(config=smoke_config)
