from dataclasses import replace
import pytest

from codal_rag.chunking.structure import StructureChunkerConfig, structure_aware_chunking, split_block
from codal_rag.models import DocumentMetadata, StructuredChunk
from codal_rag.parsing.markdown import MarkdownBlock
from codal_rag.pipelines.chunk import chunk_documents
from codal_rag.storage.json_io import load_json, save_json


def test_paragraph_and_text_blocks(tokenizer):
    for block_type in ('paragraph', 'text'):
        block = MarkdownBlock(block_id=0, block_type=block_type, content='a b c. d e f.', heading_path=[])
        assert split_block(block, tokenizer, 4) == ['a b c.', 'd e f.']


def test_heading_budget(tokenizer):
    chunks = structure_aware_chunking('# Long heading\n\n' + 'word ' * 25,
        DocumentMetadata('doc'), tokenizer, StructureChunkerConfig(max_tokens=10))
    assert len(chunks) > 2
    assert all(c.token_count <= 10 and c.heading_path == ['Long heading'] for c in chunks)
    assert sum(c.text.split().count('word') for c in chunks) == 25


def test_large_table_repeats_headers(tokenizer):
    markdown = '# Report\n\n| Year | Sales |\n| --- | --- |\n' + '\n'.join(f'| {2000+i} | {i} |' for i in range(8))
    chunks = structure_aware_chunking(markdown, DocumentMetadata('doc'), tokenizer,
                                      StructureChunkerConfig(max_tokens=25))
    assert len(chunks) > 1
    assert all(c.chunk_type == 'table' and c.text.startswith('| Year | Sales |\n| --- | --- |') for c in chunks)
    assert all(c.token_count <= 25 for c in chunks)
    assert [c.table_part for c in chunks] == list(range(1, len(chunks)+1))
    for i in range(8):
        assert sum(f'| {2000+i} | {i} |' in c.text for c in chunks) == 1


def test_wide_row_fails_without_truncation(tokenizer):
    markdown = '| Name | Value |\n| --- | --- |\n| ' + 'long ' * 30 + '| 1 |'
    with pytest.raises(ValueError, match='table header'):
        structure_aware_chunking(markdown, DocumentMetadata('doc'), tokenizer,
                                  StructureChunkerConfig(max_tokens=20))


def test_deep_heading_and_empty(tokenizer):
    assert structure_aware_chunking('', DocumentMetadata('d'), tokenizer) == []
    with pytest.raises(ValueError, match='Heading'):
        structure_aware_chunking('# heading too long\ntext', DocumentMetadata('d'), tokenizer,
                                  StructureChunkerConfig(max_tokens=2))


def test_multiple_docs_metadata_and_snapshot(corpus, smoke_config, tokenizer):
    result = chunk_documents(corpus, smoke_config, tokenizer=tokenizer,
                            metadata_by_id={'alpha/report': {'company': 'الف', 'year': '1403'}})
    assert set(result) == {'baseline','proposed'}
    for records in result.values():
        assert {r['document_id'] for r in records} == {'alpha/report', 'beta/report'}
        assert len({r['chunk_id'] for r in records}) == len(records)
        assert all(r['company'] == 'الف' for r in records if r['document_id'] == 'alpha/report')
        assert all(r['source_file'].endswith('report.md') for r in records)
    record = result['proposed'][0]
    assert StructuredChunk(**record).to_dict() == record
    (corpus / 'beta' / 'report.md').unlink()
    chunk_documents(corpus, smoke_config, tokenizer=tokenizer)
    saved = load_json(smoke_config.path('chunks','proposed','corpus.json'))
    assert {r['document_id'] for r in saved} == {'alpha/report'}


def test_single_method(corpus, smoke_config, tokenizer):
    cfg = replace(smoke_config, mode='baseline')
    assert set(chunk_documents(corpus, cfg, tokenizer=tokenizer)) == {'baseline'}


def test_json_atomic_rejects_nan(tmp_path):
    path = save_json({'good': 1}, tmp_path / 'file.json')
    with pytest.raises(ValueError):
        save_json({'bad': float('nan')}, path)
    assert load_json(path) == {'good': 1}


def test_oversized_header_only_table(tokenizer):
    from codal_rag.chunking.structure import split_table_block
    with pytest.raises(ValueError, match='header'):
        split_table_block('| Header | Value |\n| --- | --- |', tokenizer, 4)


@pytest.mark.parametrize('metadata', [[], {'alpha/report': []}, {'alpha/report': {'year': 1403}}, {'alpha/report': {'unexpected': 'value'}}])
def test_invalid_document_metadata(corpus, smoke_config, tokenizer, metadata):
    with pytest.raises(ValueError, match='metadata|Metadata'):
        chunk_documents(corpus, smoke_config, tokenizer=tokenizer, metadata_by_id=metadata)
