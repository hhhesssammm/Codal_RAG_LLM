from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import time

import pytest

from codal_rag.egar.analyzer import EvidenceSufficiencyAnalyzer
from codal_rag.egar.evidence import merge_hits, pack_hits, trace_sources
from codal_rag.egar.financial import number, validate_calculation
from codal_rag.extraction.evidence import retrieval_to_evidence
from codal_rag.pipelines.extract import extract_retrieval
from codal_rag.retrieval.scope import RetrievalScope, resolve_scope


def hit(text, chunk_id='c1', company='الف', year='1403', block=1):
    return {'rank': 1, 'score': 0.5, 'vector_id': block, 'chunk': {
        'document_id': 'report', 'chunk_id': chunk_id, 'text': text,
        'company': company, 'year': year, 'heading_path': ['گزارش', chunk_id],
        'source_file': 'report.md', 'source_block_ids': [block], 'chunk_type': 'text'}}


def claim(text='بدهی افزایش یافته است.', ids=None, **kwargs):
    return {'claim': text, 'evidence_ids': ids or ['E1'], 'basis': 'Source evidence',
            'confidence': 0.7, 'conceptual_labels': [],
            'disclaimer': 'این یک استنباط است.', **kwargs}


def extraction(level='explicit', items=None, query='q'):
    result = {'query': query, 'explicit': [], 'implicit': [], 'latent': [],
              'insufficient_evidence': False, 'notes': ''}
    result[level] = items if items is not None else [claim()]
    result['insufficient_evidence'] = not any(result[k] for k in ('explicit', 'implicit', 'latent'))
    return result


def review(ex, hits, status='sufficient', gaps=None):
    evs = {e.evidence_id: e for e in retrieval_to_evidence(hits)}
    return {'question_status': status, 'missing_evidence_requirements': gaps or [],
        'contradiction_information': [], 'justification': 'Fixture semantic assessment',
        'claims': [{'level': level, 'index': i, 'status': 'sufficient',
            'addresses_question': True, 'coherent_unit': level != 'latent',
            'individual_sufficient': level != 'latent', 'justification': 'Supported by these units',
            'joint_support': 'Increased obligations together with reduced cash generation indicate possible pressure.',
            'units': [{'evidence_id': eid, 'quote': evs[eid].text, 'role': evs[eid].chunk_id}
                      for eid in c['evidence_ids']],
            'missing_evidence_requirements': [], 'contradiction_information': [], 'calculations': []}
            for level in ('explicit', 'implicit', 'latent') for i, c in enumerate(ex[level])]}


class ScriptedClient:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class Engine:
    index_config = {}

    def __init__(self, initial, additions=()):
        self.initial, self.additions = initial, list(additions)
        self.calls = []

    def catalog(self, method):
        return [h['chunk'] for h in self.initial + [h for batch in self.additions if isinstance(batch, list) for h in batch]]

    def search_method(self, query, method, top_k, **kwargs):
        self.calls.append((query, method, top_k, kwargs))
        result = self.additions.pop(0) if self.additions else []
        if isinstance(result, Exception):
            raise result
        return deepcopy(result[:top_k])


def adaptive(config, hits, client, engine=None, **settings):
    cfg = replace(config, egar=replace(config.egar, enabled=True, **settings))
    return extract_retrieval({'original_query': 'q', 'proposed': hits}, 'proposed', cfg,
        engine=engine or Engine(hits), llm_client=client, scope=RetrievalScope(document_ids=('report',)))


def analyze(ex, hits, response=None, scope=None):
    evs = retrieval_to_evidence(hits)
    client = ScriptedClient(response or review(ex, hits))
    return EvidenceSufficiencyAnalyzer(client).analyze('q', ex, evs, scope=scope,
                                                      sources=trace_sources(evs, hits))


GAP = {'description': 'Cash generation is missing', 'query': 'جریان نقد عملیاتی', 'section_hints': []}


def test_explicit_sufficient_initial_avoids_retrieval(smoke_config):
    hits = [hit('بدهی افزایش یافته است.')]
    ex = extraction()
    client, engine = ScriptedClient(ex, review(ex, hits)), Engine(hits)
    before = deepcopy(hits)
    result = adaptive(smoke_config, hits, client, engine)
    assert not result['extraction']['insufficient_evidence']
    assert not result['egar']['activated'] and not engine.calls
    assert result['operational']['llm_calls'] == 2
    assert hits == before
    assert result['egar']['initial_retrieval'] == before


def test_implicit_single_coherent_table_and_arithmetic():
    hits = [hit('| قلم | 1403 (میلیون ریال) |\n|---|---|\n| فروش | 100 |\n| سود | 20 |')]
    ex = extraction('implicit', [claim('حاشیه سود 20 درصد است.')])
    judgment = review(ex, hits)
    calculation = {'operation': 'percentage', 'operands': [
        {'evidence_id': 'E1', 'quote': '| سود | 20 |', 'value': '20', 'unit': 'میلیون ریال', 'period': '1403'},
        {'evidence_id': 'E1', 'quote': '| فروش | 100 |', 'value': '100', 'unit': 'میلیون ریال', 'period': '1403'}],
        'result': '20', 'result_unit': 'درصد'}
    judgment['claims'][0]['calculations'] = [calculation]
    result = analyze(ex, hits, judgment)
    assert result.sufficiency_status == 'sufficient'
    assert result.claims[0].deterministic_checks[0]['computed_result'] == '20'
    calculation['result'] = '25'
    assert analyze(ex, hits, judgment).sufficiency_status == 'unverifiable'


def test_latent_complementary_units_and_retrieval_success(smoke_config):
    first = [hit('بدهی افزایش یافته است.')]
    second = hit('جریان نقد عملیاتی کاهش یافته است.', 'c2', block=2)
    initial, final = extraction(), extraction('latent', [claim('فشار نقدینگی ممکن است بیشتر شده باشد.', ['E1', 'E2'])])
    client = ScriptedClient(initial, review(initial, first, 'partially_sufficient', [GAP]),
                            final, review(final, first+[second]))
    engine = Engine(first, [[second]])
    result = adaptive(smoke_config, first, client, engine)
    assert result['egar']['stop_reason'] == 'sufficient'
    assert result['egar']['iteration_count'] == 1
    assert result['egar']['supplementary_queries'] == [GAP['query']]
    assert engine.calls[0][3]['scope'].document_ids == ('report',)
    assert result['extraction']['latent'][0]['source_trace']['E2']['source_block_ids'] == [2]


@pytest.mark.parametrize('addition', [[], [hit('بدهی افزایش یافته است.', 'overlap')]])
def test_missing_or_duplicate_additions_abstain(smoke_config, addition):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    result = adaptive(smoke_config, hits, ScriptedClient(ex, review(ex, hits, 'insufficient', [GAP])), Engine(hits, [addition]))
    assert result['extraction']['insufficient_evidence']
    assert not result['extraction']['latent']
    assert result['egar']['stop_reason'] == 'no_new_evidence'


def test_distinct_ids_or_table_parts_do_not_establish_latent_support():
    ex = extraction('latent', [claim('فشار نقدینگی ممکن است بیشتر باشد.', ['E1', 'E2'])])
    hits = [hit('بدهی افزایش یافته است.'), hit('بدهی افزایش یافته است.', 'other', block=2)]
    assert analyze(ex, hits).sufficiency_status == 'unverifiable'
    hits[1] = hit('جریان نقد عملیاتی کاهش یافته است.', 'part2', block=1)
    assert analyze(ex, hits).sufficiency_status == 'unverifiable'


@pytest.mark.parametrize('scope', [RetrievalScope(companies=('ب',)), RetrievalScope(years=('1402',))])
def test_wrong_company_or_year_rejected(scope):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    assert analyze(ex, hits, scope=scope).claims[0].sufficiency_status == 'unverifiable'


def test_contradictory_evidence_cannot_pass(smoke_config):
    hits, ex = [hit('بدهی افزایش یافته است.'), hit('بدهی کاهش یافته است.', 'c2', block=2)], extraction()
    verdict = review(ex, hits)
    verdict['contradiction_information'] = [{'evidence_ids': ['E1', 'E2'], 'description': 'Conflicting liability trends'}]
    result = adaptive(smoke_config, hits, ScriptedClient(ex, verdict))
    assert result['egar']['stop_reason'] == 'contradictory_evidence'
    assert result['extraction']['insufficient_evidence']


@pytest.mark.parametrize('text', ['افزایش بدهی باعث کاهش نقد شده است.', 'The company is bankrupt.'])
def test_unsupported_causality_and_distress_fail(text):
    hits = [hit('بدهی افزایش یافته است.'), hit('نقد کاهش یافته است.', 'c2', block=2)]
    ex = extraction('latent', [claim(text, ['E1', 'E2'])])
    assert analyze(ex, hits).sufficiency_status == 'unverifiable'


def test_unit_mismatch_and_numeral_signs():
    assert number('(۱٬۲۰۰)') == -1200
    assert number('−٢.٥') == number('-2.5')
    hits = [hit('بدهی 10 میلیارد ریال است.')]
    ex = extraction('explicit', [claim('بدهی 10 میلیون ریال است.')])
    assert analyze(ex, hits).sufficiency_status == 'unverifiable'
    with pytest.raises(ValueError):
        number('NaN')


@pytest.mark.parametrize('error', [RuntimeError('secret-provider-body'), ValueError('malformed JSON'), {'bad': 'schema'}])
def test_model_failure_and_malformed_output_fail_closed(smoke_config, error):
    result = adaptive(smoke_config, [hit('بدهی افزایش یافته است.')], ScriptedClient(error))
    assert result['extraction']['insufficient_evidence']
    assert result['egar']['stop_reason'] == 'service_or_validation_failure'
    assert 'secret-provider-body' not in json.dumps(result)


def test_retrieval_timeout_and_call_budget(smoke_config):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    engine = Engine(hits)
    def slow(*args, **kwargs):
        time.sleep(.03)
        return []
    engine.search_method = slow
    result = adaptive(smoke_config, hits, ScriptedClient(ex, review(ex, hits, 'insufficient', [GAP])),
                      engine, call_timeout_seconds=.01)
    assert result['egar']['stop_reason'] == 'budget_or_timeout'
    assert result['extraction']['insufficient_evidence']
    result = adaptive(smoke_config, hits, ScriptedClient(ex), max_llm_calls=1)
    assert result['operational']['llm_calls'] == 1 and result['extraction']['insufficient_evidence']


def test_maximum_iterations_and_repeated_queries(smoke_config):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    result = adaptive(smoke_config, hits, ScriptedClient(ex, review(ex, hits, 'insufficient', [GAP])), max_iterations=0)
    assert result['egar']['stop_reason'] == 'iteration_budget' and result['egar']['iteration_count'] == 0
    second = hit('شاهد نامرتبط جدید', 'c2', block=2)
    engine = Engine(hits, [[second]])
    client = ScriptedClient(ex, review(ex, hits, 'insufficient', [GAP, GAP]),
                            ex, review(ex, hits+[second], 'insufficient', [GAP]))
    result = adaptive(smoke_config, hits, client, engine)
    assert len(engine.calls) == 1 and result['egar']['stop_reason'] == 'no_new_targeted_queries'


def test_disabled_keeps_v1_no_judge_no_search(smoke_config):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    client = ScriptedClient(ex)
    result = extract_retrieval({'original_query': 'q', 'proposed': hits}, 'proposed', smoke_config, llm_client=client)
    assert result['schema_version'] == 1 and 'egar' not in result
    assert result['extraction'] == ex and len(client.calls) == 1


def test_packing_never_truncates_tables_and_preserves_coordinates():
    h = hit('x' * 7000)
    h['chunk']['source_locations'] = [{'page': 4, 'table': 2, 'row': 8}]
    selected, ev, _, _ = pack_hits([h], 9000)
    assert ev[0].text == 'x' * 7000
    assert trace_sources(ev, selected)['E1']['source_locations'] == [{'page': 4, 'table': 2, 'row': 8}]
    assert pack_hits([h], 300)[0] == []
    assert 'page' not in trace_sources(retrieval_to_evidence([hit('hello')]), [hit('hello')])['E1']


def test_merge_preserves_new_numbers_and_rejects_mutated_source():
    first, second = hit('سود 100 میلیون ریال'), hit('سود 120 میلیون ریال', 'c2')
    assert len(merge_hits([first], [second])[0]) == 2
    second['chunk']['chunk_id'] = 'c1'
    with pytest.raises(ValueError, match='changed'):
        merge_hits([first], [second])
    for new in ('loss 10', 'loss -100'):
        assert len(merge_hits([hit('loss 100')], [hit(new, 'c2')])[0]) == 2


def test_scope_ambiguity_explicit_multi_period_and_missing_metadata():
    catalog = [hit('a')['chunk'], {**hit('b', 'c2', year='1402')['chunk'], 'document_id': 'old'}]
    assert resolve_scope('این شرکت چطور است؟', catalog)[0] is None
    scope, _ = resolve_scope('الف در ۱۴۰۲ و ۱۴۰۳', catalog)
    assert set(scope.years) == {'1402', '1403'}
    scope = RetrievalScope(document_ids=('report', 'old'))
    assert resolve_scope('مقایسه', catalog, scope)[0] == scope
    assert not RetrievalScope(years=('1403',)).matches({'document_id': 'x'})


def test_unanchored_quotes_and_missing_reviews_fail():
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    verdict = review(ex, hits)
    verdict['claims'][0]['units'][0]['quote'] = 'Invented fact'
    assert analyze(ex, hits, verdict).sufficiency_status == 'unverifiable'
    verdict['claims'] = []
    with pytest.raises(ValueError, match='every candidate'):
        analyze(ex, hits, verdict)


def test_deterministic_analyzer_does_not_certify_semantic_inference():
    ev = retrieval_to_evidence([hit('بدهی افزایش یافته است.')])
    result = EvidenceSufficiencyAnalyzer().analyze('q', extraction(), ev)
    assert result.sufficiency_status == 'unverifiable'
    assert result.claims[0].validation_status == 'verbatim_only'


def test_numeric_conversion_and_period_alignment():
    ev = SimpleNamespace(text='1403: نقد 1 میلیارد ریال و بدهی 500 میلیون ریال', heading_path=[], year='1403')
    calc = {'operation': 'ratio', 'operands': [
        {'evidence_id': 'E1', 'quote': 'نقد 1 میلیارد ریال', 'value': '1', 'unit': 'میلیارد ریال', 'period': '1403'},
        {'evidence_id': 'E1', 'quote': 'بدهی 500 میلیون ریال', 'value': '500', 'unit': 'میلیون ریال', 'period': '1403'}],
        'result': '2', 'result_unit': 'ratio'}
    assert validate_calculation(calc, {'E1': ev})['valid']
    calc['operands'][0]['unit'] = 'میلیون ریال'
    with pytest.raises(ValueError, match='unit conflicts'):
        validate_calculation(calc, {'E1': ev})
    calc['operands'][0]['unit'] = 'میلیارد ریال'
    calc['operands'][1]['period'] = '1402'
    with pytest.raises(ValueError, match='period'):
        validate_calculation(calc, {'E1': ev})


def test_two_independent_source_paragraphs_inside_one_chunk_are_valid_v2():
    from codal_rag.extraction.source_units import annotate_units
    from codal_rag.extraction.validation import validate_extraction
    h = hit('بدهی افزایش یافته است. نقد عملیاتی کاهش یافته است.')
    annotate_units('بدهی افزایش یافته است.\n\nنقد عملیاتی کاهش یافته است.', [h['chunk']], 'baseline')
    ex = extraction('latent', [claim('فشار نقدینگی ممکن است افزایش یافته باشد.')])
    verdict = review(ex, [h])
    verdict['claims'][0]['units'] = [
        {'evidence_id': 'E1', 'quote': 'بدهی افزایش یافته است.', 'role': 'obligations'},
        {'evidence_id': 'E1', 'quote': 'نقد عملیاتی کاهش یافته است.', 'role': 'cash_generation'}]
    assert analyze(ex, [h], verdict).sufficiency_status == 'sufficient'
    evs = retrieval_to_evidence([h])
    assert not validate_extraction(ex, evs, allow_single_chunk_latent=True)
    assert validate_extraction(ex, evs)  # historical v1 still requires two IDs
    verdict['claims'][0]['individual_sufficient'] = True
    assert analyze(ex, [h], verdict).sufficiency_status == 'unverifiable'


def test_explicit_table_reading_binds_inherited_header_unit():
    h = hit('| مبلغ: میلیون ریال | 1403 |\n|---|---|\n| بدهی | 100 |')
    ex = extraction('explicit', [claim('بدهی 100 میلیون ریال است.')])
    verdict = review(ex, [h])
    verdict['claims'][0]['calculations'] = [{'operation': 'identity', 'operands': [
        {'evidence_id': 'E1', 'quote': '| بدهی | 100 |', 'value': '100',
         'unit': 'میلیون ریال', 'period': '1403'}], 'result': '100', 'result_unit': 'میلیون ریال'}]
    assert analyze(ex, [h], verdict).sufficiency_status == 'sufficient'
    verdict['claims'][0]['calculations'][0]['operands'][0]['unit'] = 'ریال'
    assert analyze(ex, [h], verdict).sufficiency_status == 'unverifiable'


def test_malformed_judge_response_and_candidate_limit_abstain(smoke_config):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    result = adaptive(smoke_config, hits, ScriptedClient(ex, {'status': 'yes'}))
    assert result['extraction']['insufficient_evidence'] and result['operational']['llm_calls'] == 2
    engine = Engine(hits, [[hit('نقد کمتر شده است.', 'c2')]])
    result = adaptive(smoke_config, hits, ScriptedClient(ex, review(ex, hits, 'insufficient', [GAP])),
                      engine, max_candidates=1)
    assert result['egar']['stop_reason'] == 'candidate_budget' and not engine.calls


def test_supplementary_scope_violation_and_original_query_repeat(smoke_config):
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    engine = Engine(hits, [[hit('نقد کمتر شده است.', 'c2', company='دیگر')]])
    cfg = replace(smoke_config, egar=replace(smoke_config.egar, enabled=True))
    result = extract_retrieval({'original_query': 'q', 'proposed': hits}, 'proposed', cfg,
        llm_client=ScriptedClient(ex, review(ex, hits, 'insufficient', [GAP])), engine=engine,
        scope=RetrievalScope(companies=('الف',), years=('1403',)))
    assert result['extraction']['insufficient_evidence']
    assert result['egar']['stop_reason'] == 'service_or_validation_failure'
    same = {**GAP, 'query': 'q'}
    result = adaptive(smoke_config, hits, ScriptedClient(ex, review(ex, hits, 'insufficient', [same])))
    assert result['egar']['stop_reason'] == 'no_new_targeted_queries'


def test_sufficient_judge_cannot_silently_override_generator_insufficiency():
    hits, ex = [hit('بدهی افزایش یافته است.')], extraction()
    ex['insufficient_evidence'] = True
    assert analyze(ex, hits).sufficiency_status == 'partially_sufficient'


def test_document_ids_are_exact_and_explicit_scope_cannot_change_named_company():
    assert not RetrievalScope(document_ids=('Report',)).matches({'document_id': 'report'})
    catalog = [hit('a', company='الف')['chunk'], {**hit('b', company='ب')['chunk'], 'document_id': 'other'}]
    scope, reason = resolve_scope('شرکت الف', catalog, RetrievalScope(document_ids=('other',)))
    assert scope is None and 'conflicts' in reason


def test_implicit_boundary_is_original_content_unit_not_chunk():
    ex = extraction('implicit', [claim('عملکرد عملیاتی تحت فشار است.', ['E1', 'E2'])])
    hits = [hit('فروش کمتر شده است.'), hit('هزینه بیشتر شده است.', 'c2', block=1)]
    assert analyze(ex, hits).sufficiency_status == 'sufficient'
    hits[1]['chunk']['source_block_ids'] = [2]
    assert analyze(ex, hits).sufficiency_status == 'unverifiable'


def test_multi_year_question_does_not_silently_mix_annual_and_interim_reports():
    catalog = [{**hit('data', year=year)['chunk'], 'document_id': f'{year}/{period}', 'financial_period': period}
               for year in ('1402', '1403') for period in ('سالانه', 'شش ماهه')]
    assert resolve_scope('الف در 1402 و 1403', catalog)[0] is None
    explicit = RetrievalScope(companies=('الف',), years=('1402', '1403'), periods=('سالانه',))
    assert resolve_scope('الف در 1402 و 1403', catalog, explicit)[0] == explicit
