# src/structure_chunker.py

from dataclasses import dataclass

from codal_rag.models import (
    DocumentMetadata,
    StructuredChunk,
)

from codal_rag.parsing.markdown import (
    MarkdownBlock,
    parse_markdown,
)

from codal_rag.chunking.tokenization import (
    TokenizerWrapper,
)


# =========================================================
# Configuration
# =========================================================

@dataclass
class StructureChunkerConfig:

    pipeline_name: str = "structure_aware"

    max_tokens: int = 500

    # Prevent creation of many very tiny text chunks.
    min_merge_tokens: int = 40

    # Include hierarchy in retrieval / embedding text.
    include_path_in_retrieval_text: bool = True

    # Separator shown to embedding model.
    path_separator: str = " > "

    def __post_init__(self):
        if self.max_tokens <= 0 or self.min_merge_tokens < 0:
            raise ValueError("Chunk token budget must be positive; merge budget nonnegative")


# =========================================================
# Heading utilities
# =========================================================

def heading_path_to_string(
    heading_path: list[str],
    separator: str = " > ",
) -> str:

    return separator.join(
        heading_path
    )


def get_section(
    heading_path: list[str]
) -> str | None:

    """
    Demo convention:

    [
        "گزارش تفسیری مدیریت",
        "وضعیت فروش",
        "دلایل کاهش فروش",
    ]

    section = "وضعیت فروش"

    If only one heading exists,
    that heading is used as section.
    """

    if len(heading_path) >= 2:
        return heading_path[1]

    if len(heading_path) == 1:
        return heading_path[0]

    return None


def get_subsection(
    heading_path: list[str]
) -> str | None:

    if len(heading_path) >= 3:
        return heading_path[-1]

    return None


# =========================================================
# Retrieval text
# =========================================================

def build_retrieval_text(
    text: str,
    heading_path: list[str],
    config: StructureChunkerConfig,
) -> str:

    if (
        not config.include_path_in_retrieval_text
        or not heading_path
    ):
        return text

    path = heading_path_to_string(
        heading_path,
        separator=config.path_separator,
    )

    return (
        f"{path}\n\n"
        f"{text}"
    )


# =========================================================
# Generic token split
# =========================================================

def token_split(
    text: str,
    tokenizer: TokenizerWrapper,
    max_tokens: int,
) -> list[str]:

    if max_tokens <= 0:
        raise ValueError("No token budget remains after structural headings")

    token_ids = tokenizer.encode(
        text
    )

    if len(token_ids) <= max_tokens:

        return [
            text
        ]

    parts = []

    start = 0

    while start < len(token_ids):

        end = min(
            start + max_tokens,
            len(token_ids),
        )

        part = tokenizer.decode(
            token_ids[start:end]
        ).strip()

        if part:

            parts.append(
                part
            )

        start = end

    return parts


# =========================================================
# Merge tiny pieces
# =========================================================

def merge_small_pieces(
    pieces: list[str],
    tokenizer: TokenizerWrapper,
    max_tokens: int,
    min_tokens: int,
) -> list[str]:

    """
    Merge very small pieces with the previous piece
    whenever the merged result stays below max_tokens.

    Example:

        300 tokens
        25 tokens

    with:

        min_tokens = 40
        max_tokens = 500

    becomes:

        325 tokens
    """

    if (
        len(pieces) <= 1
        or min_tokens <= 0
    ):
        return pieces

    merged: list[str] = []

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        piece_token_count = tokenizer.count(
            piece
        )

        if (
            merged
            and piece_token_count < min_tokens
        ):

            candidate = (
                merged[-1]
                + "\n\n"
                + piece
            )

            if (
                tokenizer.count(candidate)
                <= max_tokens
            ):

                merged[-1] = candidate

                continue

        merged.append(
            piece
        )

    return merged


# =========================================================
# Sentence splitting
# =========================================================

def split_sentences(
    text: str
) -> list[str]:

    """
    Simple Persian-friendly sentence splitter.

    It recognizes:

        .
        !
        ؟

    The rule is intentionally conservative because
    financial text contains numbers and abbreviations.
    """

    import re

    parts = re.split(
        r"(?<=[.!؟])\s+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# =========================================================
# Paragraph-aware text splitting
# =========================================================

def split_text_block(
    text: str,
    tokenizer: TokenizerWrapper,
    max_tokens: int,
) -> list[str]:

    """
    Splitting priority:

        1. Paragraph / line boundary
        2. Sentence boundary
        3. Token boundary as last resort
    """

    if (
        tokenizer.count(text)
        <= max_tokens
    ):

        return [
            text
        ]

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    result: list[str] = []

    current: list[str] = []

    for paragraph in paragraphs:

        candidate = "\n".join(
            current
            + [paragraph]
        )

        # -------------------------------------------------
        # Current buffer would overflow
        # -------------------------------------------------

        if (
            current
            and tokenizer.count(candidate)
            > max_tokens
        ):

            current_text = "\n".join(
                current
            ).strip()

            if current_text:

                result.append(
                    current_text
                )

            current = []

        # -------------------------------------------------
        # Paragraph itself fits
        # -------------------------------------------------

        if (
            tokenizer.count(paragraph)
            <= max_tokens
        ):

            current.append(
                paragraph
            )

            continue

        # -------------------------------------------------
        # Paragraph itself is too large
        # Try sentence-aware splitting
        # -------------------------------------------------

        sentence_parts = split_sentences(
            paragraph
        )

        sentence_buffer: list[str] = []

        for sentence in sentence_parts:

            sentence_candidate = " ".join(
                sentence_buffer
                + [sentence]
            )

            if (
                sentence_buffer
                and tokenizer.count(
                    sentence_candidate
                )
                > max_tokens
            ):

                buffered_text = " ".join(
                    sentence_buffer
                ).strip()

                if buffered_text:

                    result.append(
                        buffered_text
                    )

                sentence_buffer = []

            # ---------------------------------------------
            # Sentence itself fits
            # ---------------------------------------------

            if (
                tokenizer.count(sentence)
                <= max_tokens
            ):

                sentence_buffer.append(
                    sentence
                )

            # ---------------------------------------------
            # Sentence itself is too large
            # Last-resort token split
            # ---------------------------------------------

            else:

                if sentence_buffer:

                    buffered_text = " ".join(
                        sentence_buffer
                    ).strip()

                    if buffered_text:

                        result.append(
                            buffered_text
                        )

                    sentence_buffer = []

                result.extend(
                    token_split(
                        text=sentence,
                        tokenizer=tokenizer,
                        max_tokens=max_tokens,
                    )
                )

        if sentence_buffer:

            buffered_text = " ".join(
                sentence_buffer
            ).strip()

            if buffered_text:

                result.append(
                    buffered_text
                )

    if current:

        current_text = "\n".join(
            current
        ).strip()

        if current_text:

            result.append(
                current_text
            )

    return [
        part
        for part in result
        if part.strip()
    ]


# =========================================================
# List-aware splitting
# =========================================================

def split_list_block(
    text: str,
    tokenizer: TokenizerWrapper,
    max_tokens: int,
) -> list[str]:

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    if (
        tokenizer.count(text)
        <= max_tokens
    ):

        return [
            text
        ]

    chunks: list[str] = []

    current_lines: list[str] = []

    for line in lines:

        candidate = "\n".join(
            current_lines
            + [line]
        )

        # -------------------------------------------------
        # Candidate exceeds max_tokens
        # -------------------------------------------------

        if (
            current_lines
            and tokenizer.count(candidate)
            > max_tokens
        ):

            current_text = "\n".join(
                current_lines
            ).strip()

            if current_text:

                chunks.append(
                    current_text
                )

            current_lines = []

        # -------------------------------------------------
        # Single list item itself too large
        # -------------------------------------------------

        if (
            tokenizer.count(line)
            > max_tokens
        ):

            if current_lines:

                current_text = "\n".join(
                    current_lines
                ).strip()

                if current_text:

                    chunks.append(
                        current_text
                    )

                current_lines = []

            chunks.extend(
                token_split(
                    text=line,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                )
            )

        else:

            current_lines.append(
                line
            )

    if current_lines:

        current_text = "\n".join(
            current_lines
        ).strip()

        if current_text:

            chunks.append(
                current_text
            )

    return chunks


# =========================================================
# Table-aware splitting
# =========================================================

def split_table_block(
    table_text: str,
    tokenizer: TokenizerWrapper,
    max_tokens: int,
) -> list[str]:

    """
    Table-aware strategy:

    Small table:
        keep complete table.

    Large table:
        split only on row boundaries.

    Every table part repeats:
        header
        separator

    A row is not blindly token-split because that
    could destroy relationships between columns.
    """

    lines = [
        line
        for line in table_text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    # -----------------------------------------------------
    # Complete table already fits
    # -----------------------------------------------------

    if (
        tokenizer.count(table_text)
        <= max_tokens
    ):

        return [
            table_text
        ]

    # -----------------------------------------------------
    # Standard markdown table:
    #
    # line 0 = header
    # line 1 = separator
    # line 2+ = rows
    # -----------------------------------------------------

    if len(lines) < 3:
        raise ValueError("A table header exceeds the token budget; increase max_tokens")

    header = lines[0]

    separator = lines[1]

    rows = lines[2:]

    table_prefix = (
        f"{header}\n"
        f"{separator}"
    )

    parts: list[str] = []

    current_rows: list[str] = []

    for row in rows:

        candidate_rows = (
            current_rows
            + [row]
        )

        candidate = (
            table_prefix
            + "\n"
            + "\n".join(
                candidate_rows
            )
        )

        # -------------------------------------------------
        # Candidate still fits
        # -------------------------------------------------

        if (
            tokenizer.count(candidate)
            <= max_tokens
        ):

            current_rows.append(
                row
            )

            continue

        # -------------------------------------------------
        # Flush previous table rows
        # -------------------------------------------------

        if current_rows:

            part = (
                table_prefix
                + "\n"
                + "\n".join(
                    current_rows
                )
            )

            parts.append(
                part
            )

            current_rows = []

        # -------------------------------------------------
        # Check the current row by itself
        # -------------------------------------------------

        single_row_table = (
            table_prefix
            + "\n"
            + row
        )

        if (
            tokenizer.count(
                single_row_table
            )
            <= max_tokens
        ):

            current_rows = [
                row
            ]

        else:

            # ---------------------------------------------
            # Very large row.
            #
            # Keep it intact rather than breaking columns.
            # This may exceed max_tokens intentionally.
            # ---------------------------------------------

            raise ValueError(
                "A table header plus one row exceeds the token budget. "
                "Increase chunking.max_tokens or preprocess the wide table; rows are never truncated."
            )

    # -----------------------------------------------------
    # Flush remaining rows
    # -----------------------------------------------------

    if current_rows:

        part = (
            table_prefix
            + "\n"
            + "\n".join(
                current_rows
            )
        )

        parts.append(
            part
        )

    return parts


# =========================================================
# Block splitting dispatcher
# =========================================================

def split_block(
    block: MarkdownBlock,
    tokenizer: TokenizerWrapper,
    max_tokens: int,
) -> list[str]:

    # -----------------------------------------------------
    # Table
    # -----------------------------------------------------

    if block.block_type == "table":

        return split_table_block(
            table_text=block.content,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )

    # -----------------------------------------------------
    # List
    # -----------------------------------------------------

    if block.block_type == "list":

        return split_list_block(
            text=block.content,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )

    # -----------------------------------------------------
    # Normal text
    # -----------------------------------------------------

    if block.block_type in {"text", "paragraph"}:
        return split_text_block(
            text=block.content,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )

    # -----------------------------------------------------
    # Other types:
    #
    # code
    # blockquote
    # etc.
    # -----------------------------------------------------

    return token_split(
        text=block.content,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )


# =========================================================
# Chunk factory
# =========================================================

def make_chunk(
    *,
    index: int,
    metadata: DocumentMetadata,
    chunk_type: str,
    text: str,
    heading_path: list[str],
    source_block_ids: list[int],
    tokenizer: TokenizerWrapper,
    config: StructureChunkerConfig,
    table_part: int | None = None,
    table_parts: int | None = None,
) -> StructuredChunk:

    retrieval_text = build_retrieval_text(
        text=text,
        heading_path=heading_path,
        config=config,
    )

    if tokenizer.count(retrieval_text) > config.max_tokens:
        raise ValueError("Chunk including its heading exceeds max_tokens; increase the budget")

    return StructuredChunk(

        chunk_id=(
            f"{metadata.document_id}"
            f"_S{index:05d}"
        ),

        document_id=(
            metadata.document_id
        ),

        pipeline=(
            config.pipeline_name
        ),

        company=(
            metadata.company
        ),

        year=(
            metadata.year
        ),

        document_type=(
            metadata.document_type
        ),

        chunk_type=(
            chunk_type
        ),

        text=(
            text
        ),

        retrieval_text=(
            retrieval_text
        ),

        heading_path=(
            heading_path.copy()
        ),

        section=(
            get_section(
                heading_path
            )
        ),

        subsection=(
            get_subsection(
                heading_path
            )
        ),

        token_count=(
            tokenizer.count(
                retrieval_text
            )
        ),

        source_block_ids=(
            source_block_ids.copy()
        ),

        source_file=(
            metadata.source_file
        ),

        table_part=(
            table_part
        ),

        table_parts=(
            table_parts
        ),
    )


# =========================================================
# Main Structure-Aware Chunker
# =========================================================

def structure_aware_chunking(
    markdown: str,
    metadata: DocumentMetadata,
    tokenizer: TokenizerWrapper,
    config: StructureChunkerConfig
    | None = None,
) -> list[StructuredChunk]:

    if config is None:

        config = (
            StructureChunkerConfig()
        )

    blocks = parse_markdown(
        markdown
    )

    chunks: list[
        StructuredChunk
    ] = []

    chunk_index = 0

    # -----------------------------------------------------
    # Buffer for non-table blocks
    # -----------------------------------------------------

    buffer_texts: list[str] = []

    buffer_types: list[str] = []

    buffer_block_ids: list[int] = []

    buffer_path: list[str] = []


    # =====================================================
    # Buffer flush
    # =====================================================

    def flush_buffer():

        nonlocal chunk_index

        nonlocal buffer_texts
        nonlocal buffer_types
        nonlocal buffer_block_ids
        nonlocal buffer_path

        if not buffer_texts:
            return

        text = "\n\n".join(
            buffer_texts
        ).strip()

        if not text:

            buffer_texts = []
            buffer_types = []
            buffer_block_ids = []
            buffer_path = []

            return

        unique_types = set(
            buffer_types
        )

        if len(unique_types) == 1:

            chunk_type = next(
                iter(
                    unique_types
                )
            )

        else:

            chunk_type = "mixed"

        chunk = make_chunk(
            index=chunk_index,
            metadata=metadata,
            chunk_type=chunk_type,
            text=text,
            heading_path=buffer_path,
            source_block_ids=(
                buffer_block_ids
            ),
            tokenizer=tokenizer,
            config=config,
        )

        chunks.append(
            chunk
        )

        chunk_index += 1

        buffer_texts = []
        buffer_types = []
        buffer_block_ids = []
        buffer_path = []


    # =====================================================
    # Main block loop
    # =====================================================

    for block in blocks:

        # -------------------------------------------------
        # Heading
        #
        # Heading itself is not a standalone chunk.
        # Its hierarchy is already attached to following
        # blocks through heading_path.
        # -------------------------------------------------

        if (
            block.block_type
            == "heading"
        ):

            continue

        # Reserve space for the structural path before splitting content.
        prefix = build_retrieval_text("", block.heading_path, config)
        content_budget = config.max_tokens - tokenizer.count(prefix)
        if prefix:
            content_budget -= 2  # conservative allowance for tokenizer boundary effects
        if content_budget <= 0:
            raise ValueError("Heading path exhausts chunk token budget; increase max_tokens")

        # =================================================
        # TABLE
        # =================================================

        if (
            block.block_type
            == "table"
        ):

            # ---------------------------------------------
            # Never merge tables with narrative text.
            # ---------------------------------------------

            flush_buffer()

            table_parts = (
                split_table_block(
                    table_text=(
                        block.content
                    ),
                    tokenizer=tokenizer,
                    max_tokens=(
                        content_budget
                    ),
                )
            )

            total_parts = len(
                table_parts
            )

            for (
                part_index,
                table_part_text,
            ) in enumerate(
                table_parts,
                start=1,
            ):

                chunk = make_chunk(
                    index=chunk_index,
                    metadata=metadata,
                    chunk_type="table",
                    text=(
                        table_part_text
                    ),
                    heading_path=(
                        block.heading_path
                    ),
                    source_block_ids=[
                        block.block_id
                    ],
                    tokenizer=tokenizer,
                    config=config,
                    table_part=(
                        part_index
                        if total_parts > 1
                        else None
                    ),
                    table_parts=(
                        total_parts
                        if total_parts > 1
                        else None
                    ),
                )

                chunks.append(
                    chunk
                )

                chunk_index += 1

            continue

        # =================================================
        # OTHER BLOCK TYPES
        # =================================================

        pieces = split_block(
            block=block,
            tokenizer=tokenizer,
            max_tokens=(
                content_budget
            ),
        )

        # -------------------------------------------------
        # Avoid tiny pieces when possible.
        # -------------------------------------------------

        pieces = merge_small_pieces(
            pieces=pieces,
            tokenizer=tokenizer,
            max_tokens=(
                content_budget
            ),
            min_tokens=(
                config.min_merge_tokens
            ),
        )

        # =================================================
        # Piece loop
        # =================================================

        for piece in pieces:

            piece = piece.strip()

            if not piece:
                continue

            # ---------------------------------------------
            # Never mix different heading paths.
            # ---------------------------------------------

            if (
                buffer_texts
                and block.heading_path
                != buffer_path
            ):

                flush_buffer()

            # ---------------------------------------------
            # Start new structural buffer
            # ---------------------------------------------

            if not buffer_texts:

                buffer_path = (
                    block.heading_path.copy()
                )

            # ---------------------------------------------
            # Candidate merge
            # ---------------------------------------------

            candidate_text = "\n\n".join(
                buffer_texts
                + [piece]
            )

            candidate_retrieval_text = (
                build_retrieval_text(
                    text=(
                        candidate_text
                    ),
                    heading_path=(
                        buffer_path
                    ),
                    config=config,
                )
            )

            candidate_tokens = (
                tokenizer.count(
                    candidate_retrieval_text
                )
            )

            # ---------------------------------------------
            # Candidate still fits
            # ---------------------------------------------

            if (
                candidate_tokens
                <= config.max_tokens
            ):

                buffer_texts.append(
                    piece
                )

                buffer_types.append(
                    block.block_type
                )

                buffer_block_ids.append(
                    block.block_id
                )

                continue

            # ---------------------------------------------
            # Current buffer is full
            # ---------------------------------------------

            flush_buffer()

            buffer_path = (
                block.heading_path.copy()
            )

            buffer_texts = [
                piece
            ]

            buffer_types = [
                block.block_type
            ]

            buffer_block_ids = [
                block.block_id
            ]

    # =====================================================
    # Final flush
    # =====================================================

    flush_buffer()

    return chunks