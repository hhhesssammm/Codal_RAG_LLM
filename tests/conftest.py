from pathlib import Path
import socket
import pytest

from codal_rag.config import load_config
from codal_rag.chunking.tokenization import WhitespaceTokenizer


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    if request.node.get_closest_marker('external'):
        return
    def blocked(*args, **kwargs):
        raise AssertionError('Default tests must not access external services')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    for key in ('CODAL_CONFIG', 'CODAL_PROJECT_ROOT', 'CODAL_LLM_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


@pytest.fixture
def smoke_config(tmp_path):
    return load_config(environ={}, overrides={
        'project_root': tmp_path, 'top_k': 3,
        'chunking': {'max_tokens': 45, 'baseline_overlap': 10},
        'embedding': {'backend': 'hash', 'model': 'smoke-hash-v1', 'tokenizer': 'whitespace',
                      'query_prefix': '', 'passage_prefix': ''}})


@pytest.fixture
def markdown():
    return (Path(__file__).parent / 'fixtures' / 'report.md').read_text(encoding='utf-8')


@pytest.fixture
def corpus(tmp_path, markdown):
    directory = tmp_path / 'documents'
    for company in ('alpha', 'beta'):
        path = directory / company / 'report.md'
        path.parent.mkdir(parents=True)
        path.write_text(markdown, encoding='utf-8')
    return directory
