"""Installed CLI. All application imports and execution happen after argument parsing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codal_rag.config import load_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog='codal-rag', description='Persian document RAG experiments')
    result.add_argument('--config', action='append', default=[], metavar='YAML', help='Repeat in precedence order')
    result.add_argument('--root', type=Path, help='Absolute or shell-relative project root')
    result.add_argument('--mode', choices=['both', 'baseline', 'proposed'])
    result.add_argument('--top-k', type=int)
    result.add_argument('--retrieval-mode', choices=['dense', 'bm25', 'hybrid'])
    result.add_argument('--egar', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable Evidence-Gap-Guided Retrieval (default: disabled)')
    commands = result.add_subparsers(dest='command', required=True)
    commands.add_parser('config', help='Show effective configuration without secrets')
    normalize = commands.add_parser('normalize', help='Normalize a Markdown file or directory')
    normalize.add_argument('--input', required=True, type=Path)
    normalize.add_argument('--output', required=True, type=Path)
    for name in ('chunk', 'build'):
        item = commands.add_parser(name, help='Chunk documents' if name == 'chunk' else 'Chunk, embed, and index')
        item.add_argument('--input', required=True, type=Path)
        item.add_argument('--metadata', type=Path, help='JSON mapping document IDs to optional metadata')
    commands.add_parser('embed', help='Embed current chunk snapshots')
    index = commands.add_parser('index', help='Embed and build FAISS indexes')
    index.add_argument('--from-embeddings', action='store_true', help='Reuse validated embedding caches')
    retrieval = commands.add_parser('retrieve', help='Compare Baseline and Proposed retrieval')
    retrieval.add_argument('--query', required=True)
    retrieval.add_argument('--output', type=Path)
    extract = commands.add_parser('extract', help='Validate three-level extraction and citations')
    extract.add_argument('--retrieval', required=True, type=Path)
    extract.add_argument('--method', required=True, choices=['baseline', 'proposed'])
    extract.add_argument('--query')
    extract.add_argument('--output', required=True, type=Path)
    extract.add_argument('--response-file', type=Path, help='Explicitly replay a mock JSON response (no API call)')
    answer = commands.add_parser('answer', help='Retrieve and extract, optionally with EGAR')
    answer.add_argument('--query', required=True)
    answer.add_argument('--method', choices=['baseline', 'proposed'], default='proposed')
    answer.add_argument('--output', required=True, type=Path)
    for item in (retrieval, extract, answer):
        item.add_argument('--scope', type=Path, help='JSON scope: document_ids, companies, years, symbols, periods')
    template = commands.add_parser('eval-template', help='Create a 20/15/15 question annotation template')
    template.add_argument('--output', required=True, type=Path)
    batch = commands.add_parser('batch', help='Retrieve and extract both methods for a question dataset')
    batch.add_argument('--questions', required=True, type=Path)
    batch.add_argument('--output', type=Path)
    evaluation = commands.add_parser('evaluate', help='Evaluate saved question runs')
    evaluation.add_argument('--runs', type=Path)
    evaluation.add_argument('--output', type=Path)
    plan = commands.add_parser('codalbench-template', help='Create the unfilled approved 260-question plan')
    plan.add_argument('--output', required=True, type=Path)
    check = commands.add_parser('validate-benchmark', help='Validate CodalBench contracts and company splits')
    check.add_argument('--dataset', required=True, type=Path)
    check.add_argument('--full-protocol', action='store_true')
    experiment = commands.add_parser('experiment', help='Separate A-D and optional D+EGAR research runs')
    experiment.add_argument('--dataset', required=True, type=Path)
    experiment.add_argument('--arms', nargs='+', choices=['A','B','C','D','D+EGAR'], default=['A','B','C','D'])
    experiment.add_argument('--split', choices=['development','test'], default='development')
    experiment.add_argument('--allow-test', action='store_true')
    experiment.add_argument('--full-protocol', action='store_true')
    experiment.add_argument('--output', required=True, type=Path)
    research = commands.add_parser('research-evaluate', help='Evaluate coverage, adjudications and paired cluster bootstrap')
    research.add_argument('--runs', required=True, type=Path)
    research.add_argument('--dataset', required=True, type=Path)
    research.add_argument('--adjudications', type=Path)
    research.add_argument('--resamples', type=int, default=10000)
    research.add_argument('--output', required=True, type=Path)
    demo = commands.add_parser('egar-demo', help='Reproducible synthetic FAISS/hybrid run with mocked semantic services')
    demo.add_argument('--output', required=True, type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    overrides = {k: v for k, v in {'project_root': args.root, 'mode': args.mode, 'top_k': args.top_k}.items() if v is not None}
    if args.retrieval_mode is not None:
        overrides['retrieval'] = {'mode': args.retrieval_mode}
    if args.egar is not None:
        overrides['egar'] = {'enabled': args.egar}
    try:
        cfg = load_config(args.config, overrides=overrides)
        from codal_rag.storage.json_io import load_json
        scope = None
        if getattr(args, 'scope', None):
            from codal_rag.retrieval.scope import RetrievalScope
            scope = RetrievalScope(**load_json(args.scope))
        if args.command == 'config':
            print(json.dumps(cfg.public_dict(), ensure_ascii=False, indent=2))
        elif args.command == 'normalize':
            from codal_rag.pipelines.normalize import normalize_documents
            print(f"Normalized {len(normalize_documents(args.input, args.output))} documents")
        elif args.command in {'chunk', 'build'}:
            from codal_rag.pipelines.chunk import chunk_documents
            records = chunk_documents(args.input, cfg,
                            metadata_by_id=load_json(args.metadata) if args.metadata else None)
            print(json.dumps({m: len(r) for m, r in records.items()}))
            if args.command == 'build':
                from codal_rag.pipelines.index import index_corpus
                print(json.dumps(index_corpus(cfg)))
        elif args.command == 'embed':
            from codal_rag.pipelines.embed import embed_corpus
            print(json.dumps({m: len(a['records']) for m, a in embed_corpus(cfg).items()}))
        elif args.command == 'index':
            from codal_rag.pipelines.index import index_corpus
            print(json.dumps(index_corpus(cfg, from_embeddings=args.from_embeddings)))
        elif args.command == 'retrieve':
            from codal_rag.retrieval.engine import DualRetrievalEngine
            engine = DualRetrievalEngine(config=cfg)
            path = engine.save_result(engine.search(args.query, scope=scope), args.output)
            print(str(path))
        elif args.command == 'extract':
            from codal_rag.pipelines.extract import extract_retrieval
            client = None
            if args.response_file:
                response = load_json(args.response_file)
                class RecordedClient:
                    def generate_json(self, system_prompt, user_prompt):
                        return response
                client = RecordedClient()
            extract_retrieval(load_json(args.retrieval), args.method, cfg, query=args.query,
                              llm_client=client, output=args.output, scope=scope)
            print(str(args.output))
        elif args.command == 'answer':
            from codal_rag.retrieval.engine import DualRetrievalEngine
            from codal_rag.pipelines.extract import extract_retrieval
            engine = DualRetrievalEngine(config=cfg)
            payload = engine.result_payload(engine.search(args.query, scope=scope))
            extract_retrieval(payload, args.method, cfg, engine=engine, scope=scope, output=args.output)
            print(str(args.output))
        elif args.command == 'eval-template':
            from codal_rag.evaluation.dataset import create_template
            print(create_template(args.output))
        elif args.command == 'batch':
            from codal_rag.pipelines.batch import run_batch
            print(run_batch(args.questions, cfg, output_dir=args.output))
        elif args.command == 'evaluate':
            from codal_rag.pipelines.evaluate import evaluate_runs
            output = args.output or cfg.path('evaluation_results', 'report.json')
            evaluate_runs(args.runs or cfg.path('evaluation_runs'), cfg, output=output)
            print(str(output))
        elif args.command == 'codalbench-template':
            from codal_rag.evaluation.benchmark import create_benchmark_template
            print(create_benchmark_template(args.output))
        elif args.command == 'validate-benchmark':
            from codal_rag.evaluation.benchmark import load_benchmark
            data = load_benchmark(args.dataset, require_full=args.full_protocol)
            print(json.dumps({'valid': True, 'questions': len(data['questions']), 'documents': len(data['documents'])}))
        elif args.command == 'experiment':
            from codal_rag.pipelines.research import run_experiments
            print(run_experiments(args.dataset, cfg, arms=args.arms, split=args.split,
                allow_test=args.allow_test, require_full=args.full_protocol, output_dir=args.output))
        elif args.command == 'research-evaluate':
            from codal_rag.pipelines.research import evaluate_experiments
            evaluate_experiments(args.runs, args.dataset, adjudications_path=args.adjudications,
                                 resamples=args.resamples, seed=cfg.random_seed, output=args.output)
            print(str(args.output))
        elif args.command == 'egar-demo':
            from codal_rag.egar.demo import run_demo
            print(run_demo(args.output))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
