"""Explicit opt-in checks; excluded from the default offline test suite."""
import os
import pytest


@pytest.mark.external
def test_cached_multilingual_e5():
    if os.getenv('CODAL_RUN_MODEL_TESTS') != '1':
        pytest.skip('Set CODAL_RUN_MODEL_TESTS=1 after caching the model')
    from codal_rag.indexing.embeddings import EmbeddingModel
    model = EmbeddingModel('intfloat/multilingual-e5-base', local_files_only=True)
    assert model.encode_passages(['فروش شرکت کاهش یافت']).shape == (1, model.dimension)
    assert model.encode_query('فروش شرکت').shape == (1, model.dimension)


@pytest.mark.external
def test_live_provider():
    if os.getenv('CODAL_RUN_API_TESTS') != '1':
        pytest.skip('Set CODAL_RUN_API_TESTS=1 to authorize a live, potentially billable API request')
    from codal_rag.models import EvidenceItem
    from codal_rag.extraction.three_level import ThreeLevelExtractor
    evidence = EvidenceItem('E1',1,'demo_chunk','demo_doc','فروش شرکت کاهش یافته است.','paragraph',[])
    output = ThreeLevelExtractor().extract('فروش شرکت چه تغییری کرده است؟',[evidence])
    assert output['explicit']
