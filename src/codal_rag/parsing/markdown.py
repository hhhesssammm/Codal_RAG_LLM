from dataclasses import dataclass, asdict, field
from typing import Optional

import re


# =========================================================
# Data model
# =========================================================

@dataclass
class MarkdownBlock:

    block_id: int

    block_type: str

    content: str

    heading_path: list[str] = field(
        default_factory=list
    )

    heading_level: Optional[int] = None


    def to_dict(self):
        return asdict(self)


# =========================================================
# Patterns
# =========================================================

HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})[ \t]+(.+?)\s*$"
)


LIST_PATTERN = re.compile(
    r"^\s*"
    r"(?:[-+*]|\d+[.)])"
    r"\s+"
)


BLOCKQUOTE_PATTERN = re.compile(
    r"^\s*>\s?"
)


TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?"
    r"\s*:?-{3,}:?\s*"
    r"(?:\|\s*:?-{3,}:?\s*)+"
    r"\|?\s*$"
)


# =========================================================
# Heading
# =========================================================

def parse_heading(
    line: str
) -> Optional[tuple[int, str]]:

    match = HEADING_PATTERN.match(line)

    if not match:
        return None

    level = len(
        match.group(1)
    )

    title = match.group(2).strip()

    return level, title


# =========================================================
# Table helpers
# =========================================================

def is_table_row(
    line: str
) -> bool:

    stripped = line.strip()

    if not stripped:
        return False

    # at least two pipes
    return stripped.count("|") >= 2


def is_table_separator(
    line: str
) -> bool:

    return bool(
        TABLE_SEPARATOR_PATTERN.match(line)
    )


def is_table_start(
    lines: list[str],
    index: int
) -> bool:

    """
    Standard Markdown table:

    | A | B |
    |---|---|
    | 1 | 2 |

    We detect a table only when the current
    row is followed by a separator row.
    """

    if index + 1 >= len(lines):
        return False

    current = lines[index]
    next_line = lines[index + 1]

    return (
        is_table_row(current)
        and is_table_separator(next_line)
    )


# =========================================================
# Lists
# =========================================================

def is_list_item(
    line: str
) -> bool:

    return bool(
        LIST_PATTERN.match(line)
    )


# =========================================================
# Parser
# =========================================================

def parse_markdown(
    markdown: str
) -> list[MarkdownBlock]:

    lines = markdown.splitlines()

    blocks: list[MarkdownBlock] = []

    # heading level -> title
    heading_stack: dict[int, str] = {}

    block_id = 0
    index = 0

    paragraph_buffer: list[str] = []


    # -----------------------------------------------------
    # Current hierarchical path
    # -----------------------------------------------------

    def current_heading_path() -> list[str]:

        return [
            heading_stack[level]
            for level in sorted(
                heading_stack.keys()
            )
        ]


    # -----------------------------------------------------
    # Paragraph flush
    # -----------------------------------------------------

    def flush_paragraph():

        nonlocal block_id
        nonlocal paragraph_buffer

        if not paragraph_buffer:
            return

        text = "\n".join(
            paragraph_buffer
        ).strip()

        if text:

            blocks.append(
                MarkdownBlock(
                    block_id=block_id,
                    block_type="paragraph",
                    content=text,
                    heading_path=(
                        current_heading_path()
                    ),
                )
            )

            block_id += 1

        paragraph_buffer = []


    # =====================================================
    # Main loop
    # =====================================================

    while index < len(lines):

        line = lines[index]

        stripped = line.strip()

        # -------------------------------------------------
        # Empty
        # -------------------------------------------------

        if not stripped:

            flush_paragraph()

            index += 1

            continue

        # -------------------------------------------------
        # Heading
        # -------------------------------------------------

        heading = parse_heading(line)

        if heading is not None:

            flush_paragraph()

            level, title = heading

            # Delete current heading and all its children.
            #
            # Example:
            #
            # H1 A
            # H2 B
            # H3 C
            #
            # if new H2 D arrives:
            # remove B and C

            levels_to_delete = [
                old_level
                for old_level
                in heading_stack.keys()
                if old_level >= level
            ]

            for old_level in levels_to_delete:
                del heading_stack[old_level]

            heading_stack[level] = title

            blocks.append(
                MarkdownBlock(
                    block_id=block_id,
                    block_type="heading",
                    content=title,
                    heading_path=(
                        current_heading_path()
                    ),
                    heading_level=level,
                )
            )

            block_id += 1
            index += 1

            continue

        # -------------------------------------------------
        # Fenced code
        # -------------------------------------------------

        if (
            stripped.startswith("```")
            or stripped.startswith("~~~")
        ):

            flush_paragraph()

            fence = (
                "```"
                if stripped.startswith("```")
                else "~~~"
            )

            code_lines = [
                line
            ]

            index += 1

            while index < len(lines):

                code_line = lines[index]

                code_lines.append(
                    code_line
                )

                if (
                    code_line
                    .strip()
                    .startswith(fence)
                ):

                    index += 1
                    break

                index += 1

            blocks.append(
                MarkdownBlock(
                    block_id=block_id,
                    block_type="code",
                    content="\n".join(
                        code_lines
                    ),
                    heading_path=(
                        current_heading_path()
                    ),
                )
            )

            block_id += 1

            continue

        # -------------------------------------------------
        # Markdown table
        # -------------------------------------------------

        if is_table_start(
            lines,
            index
        ):

            flush_paragraph()

            table_lines = [
                lines[index],
                lines[index + 1],
            ]

            index += 2

            while index < len(lines):

                possible_row = lines[index]

                if not is_table_row(
                    possible_row
                ):
                    break

                table_lines.append(
                    possible_row
                )

                index += 1

            blocks.append(
                MarkdownBlock(
                    block_id=block_id,
                    block_type="table",
                    content="\n".join(
                        table_lines
                    ),
                    heading_path=(
                        current_heading_path()
                    ),
                )
            )

            block_id += 1

            continue

        # -------------------------------------------------
        # List
        # -------------------------------------------------

        if is_list_item(line):

            flush_paragraph()

            list_lines = []

            while index < len(lines):

                list_line = lines[index]

                if not is_list_item(
                    list_line
                ):
                    break

                list_lines.append(
                    list_line
                )

                index += 1

            blocks.append(
                MarkdownBlock(
                    block_id=block_id,
                    block_type="list",
                    content="\n".join(
                        list_lines
                    ),
                    heading_path=(
                        current_heading_path()
                    ),
                )
            )

            block_id += 1

            continue

        # -------------------------------------------------
        # Blockquote
        # -------------------------------------------------

        if BLOCKQUOTE_PATTERN.match(line):

            flush_paragraph()

            quote_lines = []

            while index < len(lines):

                quote_line = lines[index]

                if not BLOCKQUOTE_PATTERN.match(
                    quote_line
                ):
                    break

                quote_lines.append(
                    quote_line
                )

                index += 1

            blocks.append(
                MarkdownBlock(
                    block_id=block_id,
                    block_type="blockquote",
                    content="\n".join(
                        quote_lines
                    ),
                    heading_path=(
                        current_heading_path()
                    ),
                )
            )

            block_id += 1

            continue

        # -------------------------------------------------
        # Ordinary paragraph
        # -------------------------------------------------

        paragraph_buffer.append(
            line
        )

        index += 1


    flush_paragraph()

    return blocks