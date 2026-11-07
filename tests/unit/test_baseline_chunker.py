import pytest
from codal_rag.chunking.baseline import fixed_size_chunking, markdown_to_plain_text


def test_fixed_windows_and_overlap(tokenizer):
    chunks = fixed_size_chunking('one two three four five six seven', 'doc', tokenizer, 4, 1)
    assert [c.content for c in chunks] == ['one two three four', 'four five six seven']
    assert [c.token_count for c in chunks] == [4, 4]
    assert chunks[0].to_dict()['chunk_id'] == 'doc_baseline_0000'
    assert chunks[0].document_id == 'doc'


@pytest.mark.parametrize('size,overlap', [(0, 0), (-1, 0), (4, 4), (4, -1)])
def test_invalid_limits(tokenizer, size, overlap):
    with pytest.raises(ValueError):
        fixed_size_chunking('text', 'doc', tokenizer, size, overlap)


def test_empty_and_plain_markdown(tokenizer):
    assert fixed_size_chunking('', 'doc', tokenizer) == []
    assert markdown_to_plain_text('# Title\n- **item**\n[link](https://example.test)') == 'Title\nitem\nlink'
