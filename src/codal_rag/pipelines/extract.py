from __future__ import annotations

from dataclasses import replace

from codal_rag.config import load_config
from codal_rag.extraction.evidence import retrieval_to_evidence
from codal_rag.extraction.three_level import ThreeLevelExtractor
from codal_rag.extraction.citations import build_citation_view
from codal_rag.extraction.citation_validation import validate_citation_view
from codal_rag.llm.client import LLMClient
from codal_rag.storage.legacy import select_retrieval_results
from codal_rag.storage.json_io import save_json


def extract_retrieval(payload: dict, method: str, config=None, *, llm_client=None,
                      query: str | None = None, output=None, engine=None, scope=None,
                      judge_client=None) -> dict:
    if method not in {'baseline', 'proposed'}:
        raise ValueError('Extraction method must be baseline or proposed')
    config = config or load_config()
    saved_query = payload.get('original_query') or payload.get('query')
    query = query or saved_query
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Retrieval JSON has no query; supply the original query explicitly')
    if saved_query and query != saved_query:
        raise ValueError('Extraction query differs from the saved retrieval query')
    hits = select_retrieval_results(payload, method)
    for hit in hits:
        chunk = hit.get('chunk', {})
        if not isinstance(chunk, dict) or not chunk.get('chunk_id') or not chunk.get('document_id'):
            raise ValueError('Retrieval evidence must contain document_id and chunk_id')
    version = 2 if config.egar.enabled else config.extraction.schema_version
    extras = {}
    if version == 2:
        from codal_rag.egar.evidence import pack_hits, trace_sources
        hits, evidence, excluded, context_tokens = pack_hits(hits, config.extraction.max_evidence_tokens)
        extras.update(source_trace=trace_sources(evidence, hits), final_retrieval=hits,
                      context_budget={"limit": config.extraction.max_evidence_tokens,
                                      "used": context_tokens, "counter": "utf8_byte_upper_bound",
                                      "excluded_chunk_ids": excluded})
    else:
        evidence = retrieval_to_evidence(hits)
    client = llm_client
    if (evidence or config.egar.enabled) and client is None:
        llm_config = config.llm
        if config.egar.enabled or version == 2:
            # All actual retry attempts are counted by the per-question budget.
            llm_config = replace(llm_config, max_retries=0,
                                 timeout=min(llm_config.timeout, config.egar.call_timeout_seconds))
        client = LLMClient(llm_config)
    if config.egar.enabled:
        from codal_rag.egar.controller import run_adaptive
        from codal_rag.retrieval.engine import DualRetrievalEngine
        from codal_rag.retrieval.scope import RetrievalScope
        if scope is None and payload.get('scope'):
            scope = RetrievalScope(**payload['scope'])
        adaptive = run_adaptive(query, payload, method, config,
                                engine=engine or DualRetrievalEngine(config=config), client=client,
                                scope=scope, judge_client=judge_client)
        extraction, evidence = adaptive.pop('extraction'), adaptive.pop('evidence')
        extras.update(adaptive)
        extras['context_budget']['used'] = adaptive['operational']['final_evidence_tokens']
        assessments = adaptive['egar']['assessments']
        extras['context_budget']['excluded_chunk_ids'] = assessments[-1]['excluded_by_context_budget'] if assessments else []
    elif version == 2:
        from codal_rag.egar.runtime import BudgetedLLM, QuestionBudget
        from codal_rag.egar.evidence import content, token_upper_bound
        budget = QuestionBudget(config.egar)
        extraction = ThreeLevelExtractor(BudgetedLLM(client, budget), schema_version=2).extract(query, evidence)
        for level in ('explicit', 'implicit', 'latent'):
            for claim in extraction[level]:
                claim.update(inference_level=level, evidence_sufficiency_status='unassessed',
                             validation_status='structural_only',
                             source_trace={eid: extras['source_trace'][eid] for eid in claim['evidence_ids']})
        extras.update(egar={'enabled': False, 'activated': False, 'iteration_count': 0},
                      operational={**budget.metrics(), 'final_evidence_tokens': context_tokens,
                       'retrieved_token_count': sum(token_upper_bound(content(h)) for h in select_retrieval_results(payload, method)),
                       'additional_retrieval_token_upper_bound': 0,
                       'retrieved_token_count_basis': 'utf8_byte_upper_bound_not_provider_tokens'})
    else:
        extraction = ThreeLevelExtractor(client).extract(query, evidence)
    citations = build_citation_view(extraction, evidence, query=query, method=method)
    validation = validate_citation_view(citations)
    if not validation['valid']:
        raise ValueError('Citation validation failed: ' + '; '.join(validation['errors']))
    result = {'schema_version': version, 'query': query, 'method': method,
              'evidence': [e.to_dict() for e in evidence], 'extraction': extraction,
              'citation_view': citations, 'citation_validation': validation,
              'llm': {'model': config.llm.model, 'provider': config.llm.provider,
                      'mocked': llm_client is not None}, **extras}
    if output is not None:
        save_json(result, output)
    return result
