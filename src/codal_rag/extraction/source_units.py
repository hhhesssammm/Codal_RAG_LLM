"""Source-unit mapping independent of experimental chunk boundaries.

Only exact contiguous token spans shared with parsed original content are mapped.
No PDF coordinates, financial facts or missing text are inferred.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from collections import defaultdict

from codal_rag.chunking.baseline import markdown_to_plain_text
from codal_rag.parsing.markdown import parse_markdown


def annotate_units(markdown, records, method):
    blocks = [b for b in parse_markdown(markdown) if b.block_type != 'heading']
    references, postings = {}, defaultdict(set)
    for block in blocks:
        source = markdown_to_plain_text(block.content) if method == 'baseline' else block.content
        tokens = source.split()
        references[block.block_id] = (block, tokens)
        width = min(3, len(tokens))
        if width:
            for i in range(len(tokens)-width+1):
                postings[tuple(tokens[i:i+width])].add(block.block_id)
    for chunk in records:
        text = chunk.get('text') or chunk.get('content') or ''
        tokens = text.split()
        units = []
        candidate_ids = set()
        for width in (1, 2, 3):
            for i in range(len(tokens)-width+1):
                candidate_ids.update(postings.get(tuple(tokens[i:i+width]), ()))
        for block_id in sorted(candidate_ids):
            block, reference = references[block_id]
            if method == 'proposed' and block.block_id not in chunk.get('source_block_ids', []):
                continue
            matching = SequenceMatcher(None, tokens, reference, autojunk=False).get_matching_blocks()
            for span in matching:
                # Trivial shared words do not establish content-unit provenance.
                if span.size < min(3, len(reference)) or not span.size:
                    continue
                units.append({'source_unit_id': f"{chunk['document_id']}:block:{block.block_id}",
                              'source_block_id': block.block_id, 'unit_type': block.block_type,
                              'heading_path': list(block.heading_path),
                              'text': ' '.join(tokens[span.a:span.a+span.size]),
                              'chunk_token_span': [span.a, span.a+span.size],
                              'source_unit_token_span': [span.b, span.b+span.size],
                              'coordinates': 'whitespace_tokens_in_normalized_markdown_not_pdf'})
        if units:
            chunk['source_units'] = units
