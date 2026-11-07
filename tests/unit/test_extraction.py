from types import SimpleNamespace
import pytest

from codal_rag.config import LLMConfig
from codal_rag.llm.client import LLMClient, LLMError
from codal_rag.models import EvidenceItem
from codal_rag.extraction.evidence import retrieval_to_evidence, build_evidence_context
from codal_rag.extraction.three_level import ThreeLevelExtractor
from codal_rag.extraction.validation import validate_extraction
from codal_rag.extraction.citations import build_citation_view
from codal_rag.extraction.citation_validation import validate_citation_view
from codal_rag.pipelines.extract import extract_retrieval
from tests.fakes import FakeLLM


def evidence():
    return [EvidenceItem('E1',1,'c1','d','فروش کاهش یافت','paragraph',['گزارش','فروش'], source_file='report.md'),
            EvidenceItem('E2',2,'c2','d','هزینه افزایش یافت','paragraph',['گزارش','هزینه'])]


def response():
    claim = {'claim':'فروش کاهش یافت','evidence_ids':['E1'],'basis':'بیان مستقیم','confidence':0.9}
    return {'query':'چرا؟','explicit':[claim], 'implicit':[], 'latent':[],
            'insufficient_evidence':False, 'notes':''}


def test_evidence_metadata_and_citations():
    hits = [{'score':0.8,'chunk':{'chunk_id':'c','document_id':'d','content':'متن',
             'heading_path':['گزارش','فروش'], 'metadata':{'source_file':'a.md','company':'الف'}}}]
    items = retrieval_to_evidence(hits)
    assert items[0].source_file == 'a.md' and items[0].company == 'الف'
    assert 'document_id: d' in build_evidence_context(items)
    view = build_citation_view(response(), items, method='proposed')
    assert validate_citation_view(view)['valid']
    assert view['items'][0]['evidence'][0]['document_id'] == 'd'


def test_three_levels_and_prompt():
    result = response()
    result['implicit'] = [{**result['explicit'][0], 'claim':'استنباط مستقیم'}]
    result['latent'] = [{**result['explicit'][0], 'evidence_ids':['E1','E2'],
                          'disclaimer':'این نتیجه یک استنباط است.'}]
    llm = FakeLLM(result)
    assert ThreeLevelExtractor(llm).extract('چرا؟', evidence()) == result
    assert 'explicit' in llm.calls[0][0] and '[E1]' in llm.calls[0][1]


@pytest.mark.parametrize('mutation', [
    lambda r: r['explicit'][0].update(evidence_ids=['E999']),
    lambda r: r['explicit'][0].update(evidence_ids=['E1',{}]),
    lambda r: r['explicit'][0].update(evidence_ids=[]),
    lambda r: r['explicit'][0].update(confidence=float('nan')),
    lambda r: r['explicit'][0].update(confidence=True),
    lambda r: r['explicit'][0].update(basis=''),
    lambda r: r.update(query='different'),
    lambda r: r.update(implicit='bad'),
    lambda r: r.update(insufficient_evidence='true'),
])
def test_invalid_claims_fail_closed(mutation):
    result = response()
    mutation(result)
    with pytest.raises(ValueError):
        ThreeLevelExtractor(FakeLLM(result)).extract('چرا؟', evidence())


def test_latent_requires_two_distinct_evidence_and_disclaimer():
    result = response()
    result['latent'] = [dict(result['explicit'][0])]
    errors = validate_extraction(result, evidence())
    assert any('two distinct' in e for e in errors)
    assert any('disclaimer' in e for e in errors)


def test_empty_evidence_never_calls_provider():
    llm = FakeLLM()
    result = ThreeLevelExtractor(llm).extract('چرا؟', [])
    assert result['insufficient_evidence'] and not llm.calls
    assert ThreeLevelExtractor().extract('چرا؟', [])['explicit'] == []


@pytest.mark.parametrize('raw', ['', '[]', '{bad}', '{"a": NaN}', '```json\n{}', 'prose {"a":1}'])
def test_malformed_llm_json(raw):
    with pytest.raises(ValueError):
        LLMClient._parse_json(raw)


def test_json_fence_and_provider_error_redaction():
    assert LLMClient._parse_json('```json\n{"ok": true}\n```') == {'ok':True}
    class FailingAPI:
        def create(self, **kwargs):
            raise TimeoutError('sensitive-provider-message')
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=FailingAPI()))
    with pytest.raises(LLMError) as error:
        LLMClient(LLMConfig(model='test'), client=sdk).generate_json('system','user')
    assert 'sensitive-provider-message' not in str(error.value)


@pytest.mark.parametrize('finish,content', [('length','{}'), ('stop', None)])
def test_incomplete_provider_responses(finish, content):
    choice = SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[choice]))))
    with pytest.raises((ValueError, LLMError)):
        LLMClient(LLMConfig(model='test'), client=sdk).generate_json('system','user')


def test_citation_no_evidence_invalid():
    view = build_citation_view(response(), [])
    assert not validate_citation_view(view)['valid']
    assert not validate_citation_view({'items':[{'statement':'claim','evidence':[]} ]})['valid']


def test_extract_rejects_untraceable_hits(smoke_config):
    payload = {'query':'چرا؟', 'proposed':[{'chunk': {'text':'x'}}]}
    with pytest.raises(ValueError, match='document_id'):
        extract_retrieval(payload, 'proposed', smoke_config, llm_client=FakeLLM(response()))
