import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


DigitStyle = Literal["ascii", "persian", "keep"]


@dataclass
class NormalizerConfig:
    digit_style: DigitStyle = "ascii"

    normalize_dates: bool = True
    normalize_financial_units: bool = True
    normalize_half_space: bool = True
    remove_diacritics: bool = True
    normalize_whitespace: bool = True


# ---------------------------------------------------------
# Character mappings
# ---------------------------------------------------------

ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ۀ": "هٔ",
    "ة": "ه",
})

PERSIAN_TO_ASCII_DIGITS = str.maketrans({
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",

    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
})

ASCII_TO_PERSIAN_DIGITS = str.maketrans({
    "0": "۰",
    "1": "۱",
    "2": "۲",
    "3": "۳",
    "4": "۴",
    "5": "۵",
    "6": "۶",
    "7": "۷",
    "8": "۸",
    "9": "۹",
})


# ---------------------------------------------------------
# Unicode
# ---------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    NFC is safer than NFKC for documents whose formatting
    and symbols should be preserved.
    """
    text = unicodedata.normalize("NFC", text)

    # Remove invisible directional/control characters
    # but DO NOT remove ZWNJ (\u200c)
    text = text.replace("\u200e", "")
    text = text.replace("\u200f", "")
    text = text.replace("\ufeff", "")

    # Normalize Arabic punctuation used in numbers
    text = text.replace("٬", ",")
    text = text.replace("٫", ".")

    # Normalize different minus characters
    text = text.replace("−", "-")
    text = text.replace("–", "-")

    return text


# ---------------------------------------------------------
# Persian characters
# ---------------------------------------------------------

def normalize_persian_characters(text: str) -> str:
    text = text.translate(ARABIC_TO_PERSIAN)

    # Arabic tatweel
    text = text.replace("ـ", "")

    return text


# ---------------------------------------------------------
# Diacritics
# ---------------------------------------------------------

DIACRITICS_PATTERN = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def remove_diacritics(text: str) -> str:
    return DIACRITICS_PATTERN.sub("", text)


# ---------------------------------------------------------
# Digits
# ---------------------------------------------------------

def normalize_digits(
    text: str,
    style: DigitStyle = "ascii",
) -> str:

    if style == "ascii":
        return text.translate(PERSIAN_TO_ASCII_DIGITS)

    if style == "persian":
        text = text.translate(PERSIAN_TO_ASCII_DIGITS)
        return text.translate(ASCII_TO_PERSIAN_DIGITS)

    return text


# ---------------------------------------------------------
# Half-space
# ---------------------------------------------------------

def normalize_zwnj(text: str) -> str:
    """
    Conservative ZWNJ normalization.

    Example:
    می رود     -> می‌رود
    نمی شود    -> نمی‌شود
    شرکت ها    -> شرکت‌ها
    گزارش های  -> گزارش‌های
    """

    # Multiple ZWNJ -> one
    text = re.sub(r"\u200c+", "\u200c", text)

    # Remove spaces around existing ZWNJ
    text = re.sub(r"\s*\u200c\s*", "\u200c", text)

    # می رود / می شود / می باشد
    text = re.sub(
        r"\b(ن?می)\s+([آ-ی])",
        r"\1‌\2",
        text
    )

    # Common plural suffix
    text = re.sub(
        r"([آ-ی])\s+(ها|های)\b",
        r"\1‌\2",
        text
    )

    return text


# ---------------------------------------------------------
# Dates
# ---------------------------------------------------------

DATE_PATTERN = re.compile(
    r"(?<!\d)"
    r"((?:13|14|19|20)\d{2})"
    r"\s*[-./]\s*"
    r"(\d{1,2})"
    r"\s*[-./]\s*"
    r"(\d{1,2})"
    r"(?!\d)"
)


def normalize_date_formats(text: str) -> str:
    """
    Does NOT convert Jalali <-> Gregorian.

    Only canonicalizes the textual format.

    Examples:
        1402-1-5
        1402.01.05
        1402/1/05

    become:
        1402/01/05
    """

    def repl(match):
        year = match.group(1)
        month = int(match.group(2))
        day = int(match.group(3))

        return f"{year}/{month:02d}/{day:02d}"

    return DATE_PATTERN.sub(repl, text)


# ---------------------------------------------------------
# Financial units
# ---------------------------------------------------------

def normalize_financial_units(text: str) -> str:

    replacements = [
        (
            r"میلیارد\s*ریال",
            "میلیارد ریال"
        ),
        (
            r"میلیون\s*ریال",
            "میلیون ریال"
        ),
        (
            r"هزار\s*ریال",
            "هزار ریال"
        ),
        (
            r"میلیارد\s*تومان",
            "میلیارد تومان"
        ),
        (
            r"میلیون\s*تومان",
            "میلیون تومان"
        ),
        (
            r"هزار\s*تومان",
            "هزار تومان"
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


# ---------------------------------------------------------
# Whitespace
# ---------------------------------------------------------

def normalize_text_whitespace(text: str) -> str:
    """
    Only normalize internal horizontal whitespace.
    Newlines are handled at document level.
    """

    text = text.replace("\t", " ")

    # Multiple spaces -> one space
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Protection for inline code / URL
# ---------------------------------------------------------

def protect_fragments(text: str):

    protected = []

    patterns = [
        r"`[^`\n]+`",
        r"https?://[^\s)]+",
    ]

    def replacement(match):
        index = len(protected)
        protected.append(match.group(0))
        return f"__PROTECTED_{index}__"

    for pattern in patterns:
        text = re.sub(pattern, replacement, text)

    return text, protected


def restore_fragments(text: str, protected):

    for i, fragment in enumerate(protected):
        text = text.replace(
            f"__PROTECTED_{i}__",
            fragment
        )

    return text


# ---------------------------------------------------------
# Plain text normalization
# ---------------------------------------------------------

def normalize_plain_text(
    text: str,
    config: NormalizerConfig
) -> str:

    if not text:
        return text

    text, protected = protect_fragments(text)

    text = normalize_unicode(text)
    text = normalize_persian_characters(text)

    if config.remove_diacritics:
        text = remove_diacritics(text)

    text = normalize_digits(
        text,
        style=config.digit_style
    )

    if config.normalize_half_space:
        text = normalize_zwnj(text)

    if config.normalize_dates:
        text = normalize_date_formats(text)

    if config.normalize_financial_units:
        text = normalize_financial_units(text)

    if config.normalize_whitespace:
        text = normalize_text_whitespace(text)

    text = restore_fragments(text, protected)

    return text


# ---------------------------------------------------------
# Markdown detection
# ---------------------------------------------------------

HEADING_PATTERN = re.compile(
    r"^(\s{0,3}#{1,6})([ \t]+)(.*)$"
)

LIST_PATTERN = re.compile(
    r"^(\s*)((?:[-+*])|(?:\d+[.)]))([ \t]+)(.*)$"
)

BLOCKQUOTE_PATTERN = re.compile(
    r"^(\s*>+\s?)(.*)$"
)

TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?"
    r"\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+"
    r"\|?\s*$"
)


# ---------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------

def normalize_table_row(
    line: str,
    config: NormalizerConfig
) -> str:

    # Keep alignment row untouched
    if TABLE_SEPARATOR_PATTERN.match(line):
        return line

    # Preserve exact pipe structure
    cells = re.split(r"(?<!\\)\|", line)

    normalized_cells = []

    for cell in cells:

        # Preserve surrounding whitespace approximately
        leading = len(cell) - len(cell.lstrip(" "))
        trailing = len(cell) - len(cell.rstrip(" "))

        core = cell.strip()

        if core:
            core = normalize_plain_text(
                core,
                config
            )

        left = " " if leading > 0 else ""
        right = " " if trailing > 0 else ""

        normalized_cells.append(
            left + core + right
        )

    return "|".join(normalized_cells)


def looks_like_table_row(line: str) -> bool:
    return line.count("|") >= 2


# ---------------------------------------------------------
# Line-aware normalization
# ---------------------------------------------------------

def normalize_markdown_line(
    line: str,
    config: NormalizerConfig
) -> str:

    if not line.strip():
        return line

    # -------------------------------------------------
    # Heading
    # -------------------------------------------------

    heading = HEADING_PATTERN.match(line)

    if heading:

        prefix = heading.group(1)
        separator = heading.group(2)
        content = heading.group(3)

        content = normalize_plain_text(
            content,
            config
        )

        return prefix + separator + content

    # -------------------------------------------------
    # Markdown table
    # -------------------------------------------------

    if looks_like_table_row(line):
        return normalize_table_row(
            line,
            config
        )

    # -------------------------------------------------
    # Lists
    # -------------------------------------------------

    item = LIST_PATTERN.match(line)

    if item:

        indent = item.group(1)
        marker = item.group(2)
        separator = item.group(3)
        content = item.group(4)

        content = normalize_plain_text(
            content,
            config
        )

        return (
            indent
            + marker
            + separator
            + content
        )

    # -------------------------------------------------
    # Blockquote
    # -------------------------------------------------

    quote = BLOCKQUOTE_PATTERN.match(line)

    if quote:

        marker = quote.group(1)
        content = quote.group(2)

        content = normalize_plain_text(
            content,
            config
        )

        return marker + content

    # -------------------------------------------------
    # Normal paragraph
    # -------------------------------------------------

    leading_spaces = (
        len(line)
        - len(line.lstrip(" "))
    )

    prefix = line[:leading_spaces]

    content = line[leading_spaces:]

    content = normalize_plain_text(
        content,
        config
    )

    return prefix + content


# ---------------------------------------------------------
# Whole Markdown document
# ---------------------------------------------------------

def normalize_markdown(
    markdown: str,
    config: NormalizerConfig | None = None
) -> str:

    if config is None:
        config = NormalizerConfig()

    lines = markdown.splitlines()

    result = []

    inside_code_block = False
    fence_marker = None

    for line in lines:

        stripped = line.strip()

        # -----------------------------
        # fenced code block
        # -----------------------------

        if stripped.startswith("```"):

            if not inside_code_block:
                inside_code_block = True
                fence_marker = "```"

            elif fence_marker == "```":
                inside_code_block = False
                fence_marker = None

            result.append(line)
            continue

        if stripped.startswith("~~~"):

            if not inside_code_block:
                inside_code_block = True
                fence_marker = "~~~"

            elif fence_marker == "~~~":
                inside_code_block = False
                fence_marker = None

            result.append(line)
            continue

        # Do not modify content inside code blocks
        if inside_code_block:
            result.append(line)
            continue

        normalized = normalize_markdown_line(
            line,
            config
        )

        result.append(normalized)

    return "\n".join(result)
def markdown_structure_signature(markdown: str) -> dict:

    headings = []
    table_rows = []
    table_separators = []
    list_items = []
    code_fences = []

    lines = markdown.splitlines()

    for line_number, line in enumerate(lines):

        # Headings
        m = HEADING_PATTERN.match(line)

        if m:
            headings.append({
                "line": line_number,
                "level": len(m.group(1).strip())
            })

        # Tables
        if looks_like_table_row(line):
            table_rows.append({
                "line": line_number,
                "pipes": line.count("|")
            })

        if TABLE_SEPARATOR_PATTERN.match(line):
            table_separators.append(line_number)

        # Lists
        if LIST_PATTERN.match(line):
            list_items.append(line_number)

        # Code blocks
        stripped = line.strip()

        if (
            stripped.startswith("```")
            or stripped.startswith("~~~")
        ):
            code_fences.append(line_number)

    return {
        "headings": headings,
        "table_rows": table_rows,
        "table_separators": table_separators,
        "list_items": list_items,
        "code_fences": code_fences,
    }


def assert_markdown_structure_preserved(
    before: str,
    after: str
):

    before_signature = markdown_structure_signature(
        before
    )

    after_signature = markdown_structure_signature(
        after
    )

    if before_signature != after_signature:

        raise ValueError(
            "Markdown structure was changed during normalization.\n"
            f"BEFORE:\n{before_signature}\n\n"
            f"AFTER:\n{after_signature}"
        )