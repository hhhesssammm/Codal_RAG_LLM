from src.chunk_models import (
    DocumentMetadata,
)

from src.structure_chunker import (
    StructureChunkerConfig,
    structure_aware_chunking,
)

from src.tokenizer_utils import (
    TokenizerWrapper,
)


MARKDOWN = """
# گزارش تفسیری مدیریت

## وضعیت فروش

در سال مالی جاری مقدار فروش شرکت کاهش یافته است.

### دلایل کاهش فروش

کاهش فروش عمدتاً ناشی از محدودیت تأمین مواد اولیه بوده است.

### جدول فروش

| سال | فروش |
| --- | ---: |
| 1402 | 100 |
| 1403 | 80 |
""".strip()


metadata = DocumentMetadata(
    document_id="test_report",
    company="شرکت الف",
    year="1403",
    document_type="گزارش تفسیری مدیریت",
)


tokenizer = TokenizerWrapper()


config = StructureChunkerConfig(
    max_tokens=500
)


def get_chunks():

    return structure_aware_chunking(
        markdown=MARKDOWN,
        metadata=metadata,
        tokenizer=tokenizer,
        config=config,
    )


def test_chunks_created():

    chunks = get_chunks()

    assert len(chunks) >= 3


def test_sales_section():

    chunks = get_chunks()

    chunk = next(
        chunk
        for chunk in chunks
        if (
            "مقدار فروش شرکت کاهش"
            in chunk.text
        )
    )

    assert chunk.heading_path == [
        "گزارش تفسیری مدیریت",
        "وضعیت فروش",
    ]


def test_reason_subsection():

    chunks = get_chunks()

    chunk = next(
        chunk
        for chunk in chunks
        if (
            "محدودیت تأمین مواد اولیه"
            in chunk.text
        )
    )

    assert chunk.heading_path == [
        "گزارش تفسیری مدیریت",
        "وضعیت فروش",
        "دلایل کاهش فروش",
    ]

    assert chunk.section == (
        "وضعیت فروش"
    )

    assert chunk.subsection == (
        "دلایل کاهش فروش"
    )


def test_table_is_independent():

    chunks = get_chunks()

    tables = [
        chunk
        for chunk in chunks
        if chunk.chunk_type == "table"
    ]

    assert len(tables) == 1

    table = tables[0]

    assert "| 1402 | 100 |" in table.text

    assert "| 1403 | 80 |" in table.text


def test_table_path():

    chunks = get_chunks()

    table = next(
        chunk
        for chunk in chunks
        if chunk.chunk_type == "table"
    )

    assert table.heading_path == [
        "گزارش تفسیری مدیریت",
        "وضعیت فروش",
        "جدول فروش",
    ]


def test_company_metadata():

    chunks = get_chunks()

    for chunk in chunks:

        assert chunk.company == (
            "شرکت الف"
        )

        assert chunk.year == "1403"


def test_path_added_to_retrieval_text():

    chunks = get_chunks()

    target = next(
        chunk
        for chunk in chunks
        if (
            "محدودیت تأمین مواد اولیه"
            in chunk.text
        )
    )

    assert (
        "گزارش تفسیری مدیریت"
        in target.retrieval_text
    )

    assert (
        "دلایل کاهش فروش"
        in target.retrieval_text
    )
def test_different_sections_are_not_merged():

    markdown = """
# گزارش تفسیری مدیریت

## فروش

فروش شرکت افزایش یافته است.

## ریسک‌ها

ریسک نرخ ارز افزایش یافته است.
""".strip()

    chunks = structure_aware_chunking(
        markdown=markdown,
        metadata=metadata,
        tokenizer=tokenizer,
        config=config,
    )

    sales_chunk = next(
        c
        for c in chunks
        if "فروش شرکت" in c.text
    )

    risk_chunk = next(
        c
        for c in chunks
        if "ریسک نرخ ارز" in c.text
    )

    assert (
        sales_chunk.chunk_id
        != risk_chunk.chunk_id
    )

    assert (
        sales_chunk.heading_path
        != risk_chunk.heading_path
    )
def test_table_not_mixed_with_text():

    chunks = get_chunks()

    table = next(
        c
        for c in chunks
        if c.chunk_type == "table"
    )

    assert (
        "کاهش فروش عمدتاً"
        not in table.text
    )

    assert table.text.startswith(
        "| سال"
    )