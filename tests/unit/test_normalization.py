from src.persian_normalizer import (
    NormalizerConfig,
    normalize_markdown,
    assert_markdown_structure_preserved,
)


def test_persian_characters():

    text = "شركت مديريت مالي"

    result = normalize_markdown(
        text
    )

    assert result == "شرکت مدیریت مالی"


def test_digits():

    config = NormalizerConfig(
        digit_style="ascii"
    )

    text = "سال ۱۴۰۳"

    result = normalize_markdown(
        text,
        config
    )

    assert result == "سال 1403"


def test_date():

    text = "تاریخ: ۱۴۰۳-۶-۵"

    result = normalize_markdown(
        text
    )

    assert result == "تاریخ: 1403/06/05"


def test_heading_preservation():

    text = """
# گزارش مديريت
## فروش شركت
### بازار داخلي
""".strip()

    normalized = normalize_markdown(
        text
    )

    assert "# گزارش مدیریت" in normalized

    assert "## فروش شرکت" in normalized

    assert "### بازار داخلی" in normalized

    assert_markdown_structure_preserved(
        text,
        normalized
    )


def test_table_preservation():

    text = """
| سال | مبلغ |
| --- | ---: |
| ۱۴۰۲ | ۱٬۲۰۰ میلیون ريال |
| ۱۴۰۳ | ۱٬۵۰۰ میلیون ريال |
""".strip()

    normalized = normalize_markdown(
        text
    )

    assert "| --- | ---: |" in normalized

    assert "1402" in normalized

    assert "1,200 میلیون ریال" in normalized

    assert_markdown_structure_preserved(
        text,
        normalized
    )