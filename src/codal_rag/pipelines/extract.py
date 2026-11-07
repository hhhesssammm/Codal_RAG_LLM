from __future__ import annotations

from codal_rag.config import load_config
from codal_rag.extraction.evidence import retrieval_to_evidence
from codal_rag.extraction.three_level import ThreeLevelExtractor
from codal_rag.extraction.citations import build_citation_view
from codal_rag.extraction.citation_validation import validate_citation_view
from codal_rag.llm.client import LLMClient
from codal_rag.storage.legacy import select_retrieval_results
from codal_rag.storage.json_io import save_json


def extract_retrieval(payload: dict, method: str, config=None, *, llm_client=None,
                      query: str | None = None, output=None) -> dict:
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
    evidence = retrieval_to_evidence(hits)
    client = llm_client
    if evidence and client is None:
        client = LLMClient(config.llm)
    extraction = ThreeLevelExtractor(client).extract(query, evidence)
    citations = build_citation_view(extraction, evidence, query=query, method=method)
    validation = validate_citation_view(citations)
    if not validation['valid']:
        raise ValueError('Citation validation failed: ' + '; '.join(validation['errors']))
    result = {'schema_version': 1, 'query': query, 'method': method,
              'evidence': [e.to_dict() for e in evidence], 'extraction': extraction,
              'citation_view': citations, 'citation_validation': validation,
              'llm': {'model': config.llm.model, 'provider': config.llm.provider,
                      'mocked': llm_client is not None}}
    if output is not None:
        save_json(result, output)
    return result
