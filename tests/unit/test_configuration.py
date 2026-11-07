import pytest
from codal_rag.config import load_config, ConfigurationError


def test_precedence_and_config_relative_root(tmp_path, monkeypatch):
    directory = tmp_path / 'settings'
    directory.mkdir()
    base = directory / 'base.yaml'
    local = directory / 'local.yaml'
    base.write_text('project_root: ..\ntop_k: 4\nembedding:\n  batch_size: 8\n')
    local.write_text('top_k: 6\n')
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config([base, local], environ={'CODAL_TOP_K': '7'}, overrides={'top_k': 9})
    assert cfg.project_root == tmp_path
    assert cfg.top_k == 9
    assert cfg.embedding.batch_size == 8
    assert cfg.path('chunks') == tmp_path / 'data' / 'chunks'


@pytest.mark.parametrize('overrides', [
    {'top_k': 0}, {'top_k': True}, {'mode': 'other'}, {'typo': 1},
    {'chunking': {'max_tokens': -1}}, {'chunking': {'baseline_overlap': 500}},
    {'embedding': {'normalize': False}}, {'embedding': {'batch_size': '16'}},
    {'llm': {'timeout': float('nan')}}, {'evaluation': {'answer_threshold': 2}},
    {'embedding': {'backend': 'hash'}}, {'evaluation': {'semantic_enabled': 'false'}},
])
def test_invalid_configuration(overrides):
    with pytest.raises(ConfigurationError):
        load_config(environ={}, overrides=overrides)


def test_secrets_never_serialized_or_in_repr():
    cfg = load_config(environ={'OPENAI_API_KEY': 'test-credential',
                              'CODAL_LLM_BASE_URL': 'https://example.test/v1?token=test-token'})
    assert cfg.llm.api_key == 'test-credential'
    assert 'test-credential' not in repr(cfg)
    assert 'test-credential' not in str(cfg.public_dict())
    assert 'test-token' not in str(cfg.public_dict())


def test_yaml_credentials_rejected_without_exposure(tmp_path):
    path = tmp_path / 'secret.yaml'
    path.write_text('llm:\n  api_key: do-not-print-this\n')
    with pytest.raises(ConfigurationError) as error:
        load_config([path], environ={})
    assert 'do-not-print-this' not in str(error.value)


@pytest.mark.parametrize('text', ['[1, 2]', 'top_k: [', 'llm: 3', 'project_root: 3'])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / 'config.yaml'
    path.write_text(text)
    with pytest.raises(ConfigurationError):
        load_config([path], environ={})


def test_env_boolean_and_invalid_env():
    assert load_config(environ={'CODAL_EVALUATION_SEMANTIC_ENABLED': 'true'}).evaluation.semantic_enabled
    with pytest.raises(ConfigurationError):
        load_config(environ={'CODAL_EMBEDDING_NORMALIZE': 'maybe'})
