import json
import config
from pathlib import Path

from src.tokenizer_utils import (
    TokenizerWrapper,
)

from src.baseline_chunker import (
    fixed_size_chunking,
)

from src.structure_chunker import (
    structure_aware_chunking,
)

import json

from pathlib import Path

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


INPUT_DIR = Path(
    "data/normalized_markdown"
)

BASELINE_DIR = Path(
    "data/chunks/baseline"
)

PROPOSED_DIR = Path(
    "data/chunks/proposed"
)


def save_chunks(
    chunks,
    path: Path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = [
        chunk.to_dict()
        for chunk in chunks
    ]

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def process_document(
    path: Path,
    tokenizer: TokenizerWrapper
):

    document_id = path.stem

    markdown = path.read_text(
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print(
        f"Document: {document_id}"
    )

    # ========================================
    # SYSTEM A
    # ========================================

    baseline_chunks = (
        fixed_size_chunking(
            markdown=markdown,
            document_id=document_id,
            tokenizer=tokenizer,
            chunk_size=config.MAX_CHUNK_TOKENS,
            overlap=100
        )
    )

    # ========================================
    # SYSTEM B
    # ========================================

    proposed_chunks = (
        structure_aware_chunking(
            markdown=markdown,
            document_id=document_id,
            tokenizer=tokenizer,
            max_tokens=config.MAX_CHUNK_TOKENS
        )
    )

    print(
        f"Baseline chunks: "
        f"{len(baseline_chunks)}"
    )

    print(
        f"Proposed chunks: "
        f"{len(proposed_chunks)}"
    )

    # ========================================
    # Save
    # ========================================

    save_chunks(
        baseline_chunks,
        BASELINE_DIR
        / f"{document_id}.json"
    )

    save_chunks(
        proposed_chunks,
        PROPOSED_DIR
        / f"{document_id}.json"
    )




# =========================================================
# Paths
# =========================================================

INPUT_FILE = Path(
    "data/normalized_markdown/foolad/1.md"
)

OUTPUT_FILE = Path(
    "data/chunks/proposed/1.json"
)


# =========================================================
# Document metadata
# =========================================================

metadata = DocumentMetadata(

    document_id="1",

    company="شرکت الف",

    year="1403",

    document_type="گزارش تفسیری مدیریت",

    source_file=INPUT_FILE.name,
)


# =========================================================
# Configuration
# =========================================================

config = StructureChunkerConfig(

    max_tokens=500,

    include_path_in_retrieval_text=True,

    path_separator=" > ",
)


# =========================================================
# Main
# =========================================================

def main():

    print(
        "Loading tokenizer..."
    )

    tokenizer = TokenizerWrapper()

    files = list(
        INPUT_DIR.rglob(
            "*.md"
        )
    )

    print(
        f"{len(files)} documents found."
    )

    for path in files:

        process_document(
            path,
            tokenizer
        )

    markdown = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    chunks = structure_aware_chunking(
        markdown=markdown,
        metadata=metadata,
        tokenizer=tokenizer,
        config=config,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            [
                chunk.to_dict()
                for chunk in chunks
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)

    print(
        f"Created chunks: {len(chunks)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 70)

    for chunk in chunks:

        print()

        print(
            f"ID: {chunk.chunk_id}"
        )

        print(
            f"TYPE: {chunk.chunk_type}"
        )

        print(
            "PATH:",
            " > ".join(
                chunk.heading_path
            )
        )

        print(
            f"TOKENS: {chunk.token_count}"
        )

        print("-" * 60)

        print(
            chunk.text[:500]
        )






if __name__ == "__main__":
    main()