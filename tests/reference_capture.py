"""Capture synthetic regression outputs from a credential-free Demo source snapshot.

python -m tests.reference_capture --reference ../Demo-reference --output tests/fixtures/demo_regression.json

Never imports or reads Demo config.py. Only missing plumbing symbols are injected;
algorithm function bodies are executed from the pinned reference source unchanged.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys
import types

import numpy as np
from codal_rag.chunking.tokenization import WhitespaceTokenizer
from codal_rag.models import Chunk

REFERENCE_COMMIT = '208bb64bdea92f919008eb5bba2ce24588a92daf'


def load_reference(root, path, namespace=None, only=None):
    source = (root / path).read_text(encoding='utf-8')
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(n.name == 'config' or n.name.startswith('src') for n in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and (node.module == 'config' or (node.module or '').startswith('src.')):
            continue
        if only and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if only and node.name not in only:
            continue
        body.append(node)
    tree.body = body
    name = '_demo_reference_' + Path(path).stem
    module = types.ModuleType(name)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module  # dataclass resolves its defining module here
    exec(compile(tree, str(root / path), 'exec'), module.__dict__)
    return module


def capture(reference_dir=None):
    if reference_dir:
        root = Path(reference_dir)
        norm = load_reference(root,'src/persian_normalizer.py')
        parser = load_reference(root,'src/markdown_parser.py')
        models = load_reference(root,'src/chunk_models.py')
        baseline = load_reference(root,'src/baseline_chunker.py',
            {'config':types.SimpleNamespace(MAX_CHUNK_TOKENS=500), 'Chunk':Chunk, 'TokenizerWrapper':WhitespaceTokenizer})
        structure = load_reference(root,'src/structure_chunker.py',
            {'DocumentMetadata':models.DocumentMetadata,'StructuredChunk':models.StructuredChunk,
             'MarkdownBlock':parser.MarkdownBlock, 'parse_markdown':parser.parse_markdown,
             'TokenizerWrapper':WhitespaceTokenizer})
        representation = load_reference(root,'run_indexing.py',only={'get_baseline_embedding_text','get_proposed_embedding_text'})
        lexical = retrieval = faithfulness = load_reference(root,'src/evaluation_metrics.py')
        evidence = load_reference(root,'src/evidence_context.py')
        citations = load_reference(root,'src/citation_builder.py')
        faiss_store = load_reference(root,'src/faiss_store.py')
        extractor = load_reference(root,'src/three_level_extractor.py',
            {'EvidenceItem':evidence.EvidenceItem, 'build_evidence_context':evidence.build_evidence_context,
             'LLMClient':object})
        dual = load_reference(root,'src/retrieval_engine.py',
            {'NormalizerConfig':norm.NormalizerConfig, 'normalize_plain_text':norm.normalize_plain_text,
             'EMBEDDING_MODEL':'smoke-hash-v1', 'EMBEDDING_BATCH_SIZE':16,
             'NORMALIZE_EMBEDDINGS':True, 'TOP_K':5})
    else:
        from codal_rag.normalization import persian as norm
        from codal_rag.parsing import markdown as parser
        from codal_rag import models
        from codal_rag.chunking import baseline, structure
        from codal_rag.indexing import representations as representation, faiss_store
        from codal_rag.evaluation import lexical, retrieval, faithfulness
        from codal_rag.extraction import evidence, citations
        from codal_rag.extraction import three_level as extractor
        from codal_rag.retrieval import engine as dual
    tokenizer = WhitespaceTokenizer()
    text = '# گزارش مديريت\n\n## فروش\n\nفروش شركت در سال ۱۴۰۳ کاهش یافته است.\n\n| سال | مبلغ |\n| --- | ---: |\n| ۱۴۰۳ | ۱٬۲۰۰ |\n\n## ریسک\n\n- هزینه مالی\n- نرخ ارز'
    normalized = norm.normalize_markdown(text)
    blocks = [asdict(b) for b in parser.parse_markdown(normalized)]
    fixed = [c.to_dict() for c in baseline.fixed_size_chunking(normalized,'synthetic',tokenizer,12,3)]
    # Baseline source lacks Chunk; compatibility shim injects the missing record type.
    structured = [c.to_dict() for c in structure.structure_aware_chunking(normalized,
        models.DocumentMetadata('synthetic',source_file='synthetic.md'),tokenizer,
        structure.StructureChunkerConfig(max_tokens=100))]
    hits = [{'rank':i+1,'score':1.0/(i+1), 'vector_id':i, 'chunk':chunk} for i,chunk in enumerate(structured)]
    ev = evidence.retrieval_to_evidence(hits)
    response = {'query':'فروش؟','explicit':[{'claim':'فروش کاهش یافته است.','evidence_ids':['E1'],
                  'confidence':0.9,'basis':'بیان مستقیم'}], 'implicit':[], 'latent':[],
                  'insufficient_evidence':False, 'notes':''}
    mock = types.SimpleNamespace(generate_json=lambda **_: response)
    extracted = extractor.ThreeLevelExtractor(mock).extract('فروش؟', ev)
    dual_result = dual.DualRetrievalResult('فروش؟', 'فروش؟', 3, hits, hits)
    store = faiss_store.FaissVectorStore(2)
    store.add(np.array([[1.,0.],[0.,1.]],dtype=np.float32),[{'chunk_id':'c1'},{'chunk_id':'c2'}])
    return {'reference_commit':REFERENCE_COMMIT, 'normalization':normalized,'blocks':blocks,
            'baseline':fixed, 'proposed':structured,
            'representations':{'baseline':representation.get_baseline_embedding_text(fixed[0]),
                               'proposed':representation.get_proposed_embedding_text(structured[0])},
            'evidence':[asdict(e) for e in ev], 'extraction':extracted, 'dual_retrieval':asdict(dual_result),
            'citations':citations.build_citation_view(response,ev,method='proposed'),
            'faiss':store.search(np.array([0.6,0.8],dtype=np.float32),2),
            'metrics':retrieval.calculate_retrieval_metrics(hits,[{'document_id':'synthetic','text':'فروش شرکت در سال 1403 کاهش یافته است.'}],k=3),
            'lexical_f1':lexical.token_f1('شركت فروش کاهش','شرکت فروش'),
            'citation_validity':faithfulness.calculate_citation_faithfulness(response,[asdict(e) for e in ev],'explicit')['score']}


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--reference', required=True, type=Path)
    args.add_argument('--output', required=True, type=Path)
    options = args.parse_args()
    options.output.write_text(json.dumps(capture(options.reference),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
