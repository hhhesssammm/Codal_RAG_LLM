from pathlib import Path
import pytest

from codal_rag.storage.json_io import save_json
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.pipelines.index import index_corpus


def test_streamlit_initializes_without_data(tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('CODAL_PROJECT_ROOT', str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).parents[2]/'apps'/'streamlit_app.py')).run()
    assert not app.exception
    assert app.info


def test_streamlit_retrieval(corpus, smoke_config, tokenizer, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    index_corpus(smoke_config)
    config = smoke_config.public_dict()
    config['llm']['base_url'] = 'https://api.openai.com/v1'
    config_path = save_json(config, smoke_config.project_root/'smoke.yaml')
    monkeypatch.setenv('CODAL_CONFIG',str(config_path))
    app = AppTest.from_file(str(Path(__file__).parents[2]/'apps'/'streamlit_app.py')).run()
    assert not app.exception
    next(x for x in app.text_input if x.label == 'سؤال کاربر').set_value('فروش شرکت')
    next(x for x in app.button if x.label == 'جست‌وجو').click()
    app.run(timeout=10)
    assert not app.exception
    assert not app.error
    assert len(app.header) == 2
