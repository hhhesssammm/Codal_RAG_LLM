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
    template = commands.add_parser('eval-template', help='Create a 20/15/15 question annotation template')
    template.add_argument('--output', required=True, type=Path)
    batch = commands.add_parser('batch', help='Retrieve and extract both methods for a question dataset')
    batch.add_argument('--questions', required=True, type=Path)
    batch.add_argument('--output', type=Path)
    evaluation = commands.add_parser('evaluate', help='Evaluate saved question runs')
    evaluation.add_argument('--runs', type=Path)
    evaluation.add_argument('--output', type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    overrides = {k: v for k, v in {'project_root': args.root, 'mode': args.mode, 'top_k': args.top_k}.items() if v is not None}
    try:
        cfg = load_config(args.config, overrides=overrides)
        from codal_rag.storage.json_io import load_json
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
            path = engine.save_result(engine.search(args.query), args.output)
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
                              llm_client=client, output=args.output)
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
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
