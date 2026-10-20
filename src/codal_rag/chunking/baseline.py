import re
import config
from src.chunk_models import Chunk
from src.tokenizer_utils import TokenizerWrapper


def markdown_to_plain_text(
    markdown: str
) -> str:

    text = markdown

    # -------------------------
    # Remove fenced code marker
    # -------------------------

    text = re.sub(
        r"```[^\n]*",
        "",
        text
    )

    text = text.replace(
        "```",
        ""
    )

    # -------------------------
    # Headings
    # -------------------------

    text = re.sub(
        r"^\s*#{1,6}\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # -------------------------
    # Blockquote
    # -------------------------

    text = re.sub(
        r"^\s*>\s?",
        "",
        text,
        flags=re.MULTILINE
    )

    # -------------------------
    # Lists
    # -------------------------

    text = re.sub(
        r"^\s*[-+*]\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"^\s*\d+[.)]\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # -------------------------
    # Table alignment
    # -------------------------

    text = re.sub(
        r"^\s*\|?"
        r"\s*:?-{3,}:?\s*"
        r"(?:\|\s*:?-{3,}:?\s*)+"
        r"\|?\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # -------------------------
    # Table pipes -> space
    # -------------------------

    text = text.replace(
        "|",
        " "
    )

    # -------------------------
    # Markdown emphasis
    # -------------------------

    text = re.sub(
        r"[*_~]{1,3}",
        "",
        text
    )

    # -------------------------
    # Markdown links
    # -------------------------

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # -------------------------
    # Multiple spaces
    # -------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # -------------------------
    # Excess blank lines
    # -------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()
def fixed_size_chunking(
    markdown: str,
    document_id: str,
    tokenizer: TokenizerWrapper,
    chunk_size: int = config.MAX_CHUNK_TOKENS,
    overlap: int = 100
) -> list[Chunk]:

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    plain_text = markdown_to_plain_text(
        markdown
    )

    tokens = tokenizer.encode(
        plain_text
    )

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(tokens):

        end = min(
            start + chunk_size,
            len(tokens)
        )

        token_slice = tokens[
            start:end
        ]

        text = tokenizer.decode(
            token_slice
        ).strip()

        if text:

            chunk = Chunk(
                chunk_id=(
                    f"{document_id}"
                    f"_baseline_"
                    f"{chunk_index:04d}"
                ),
                document_id=document_id,
                pipeline="baseline",
                content=text,
                chunk_type="fixed",
                token_count=len(
                    token_slice
                ),
                heading_path=[],
                block_index=chunk_index
            )

            chunks.append(
                chunk
            )

            chunk_index += 1

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks