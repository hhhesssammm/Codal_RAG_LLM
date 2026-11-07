from codal_rag.parsing.markdown import (
    parse_markdown,
)


SAMPLE = """
# گزارش تفسیری مدیریت

## فروش

فروش شرکت افزایش یافته است.

### فروش داخلی

| سال | مبلغ |
| --- | ---: |
| 1402 | 1200 |
| 1403 | 1500 |

- بازار داخلی
- بازار صادراتی

## ریسک‌ها

ریسک نرخ ارز افزایش یافته است.
""".strip()


def test_parser():

    blocks = parse_markdown(
        SAMPLE
    )

    types = [
        block.block_type
        for block in blocks
    ]

    assert "heading" in types
    assert "paragraph" in types
    assert "table" in types
    assert "list" in types


def test_table_heading_path():

    blocks = parse_markdown(
        SAMPLE
    )

    table = next(
        block
        for block in blocks
        if block.block_type == "table"
    )

    assert table.heading_path == [
        "گزارش تفسیری مدیریت",
        "فروش",
        "فروش داخلی",
    ]


def test_risk_section():

    blocks = parse_markdown(
        SAMPLE
    )

    risk_paragraph = next(
        block
        for block in blocks
        if (
            block.block_type
            == "paragraph"
            and "نرخ ارز"
            in block.content
        )
    )

    assert risk_paragraph.heading_path == [
        "گزارش تفسیری مدیریت",
        "ریسک‌ها",
    ]