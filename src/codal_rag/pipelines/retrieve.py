import shutil

from pathlib import Path

from src.retrieval_engine import (
    DualRetrievalEngine,
)


# =========================================================
# Helpers
# =========================================================

def get_chunk_text(
    chunk: dict
) -> str:

    return (
        chunk.get("text")
        or chunk.get("content")
        or ""
    )


def print_result(
    result: dict,
    pipeline: str,
):

    chunk = result["chunk"]

    print()
    print("-" * 70)

    print(
        f"Rank: {result['rank']}"
    )

    print(
        f"Similarity: "
        f"{result['score']:.4f}"
    )

    print(
        "Chunk ID:",
        chunk.get(
            "chunk_id",
            "-"
        ),
    )

    print(
        "Document:",
        chunk.get(
            "document_id",
            "-"
        ),
    )

    chunk_type = (
        chunk.get("chunk_type")
        or chunk.get("type")
        or "-"
    )

    print(
        "Type:",
        chunk_type
    )

    # -----------------------------------------------------
    # Structural metadata only where available
    # -----------------------------------------------------

    heading_path = (
        chunk.get(
            "heading_path"
        )
        or []
    )

    if heading_path:

        print(
            "Path:",
            " > ".join(
                heading_path
            )
        )

    if chunk.get("company"):

        print(
            "Company:",
            chunk["company"]
        )

    if chunk.get("year"):

        print(
            "Year:",
            chunk["year"]
        )

    print("-" * 70)

    print(
        get_chunk_text(
            chunk
        )
    )


# =========================================================
# Main
# =========================================================

def main():

    engine = (
        DualRetrievalEngine()
    )

    print()

    query = input(
        "سؤال را وارد کنید: "
    ).strip()

    if not query:

        query = (
            "علت کاهش فروش شرکت "
            "در سال ۱۴۰۳ چیست؟"
        )

    result = engine.search(
        query=query,
        top_k=5,
    )

    print()
    print("=" * 80)

    print(
        "Original Query:"
    )

    print(
        result.original_query
    )

    print()

    print(
        "Normalized Query:"
    )

    print(
        result.normalized_query
    )

    # =====================================================
    # BASELINE
    # =====================================================

    print()
    print()
    print("=" * 80)
    print(
        "SYSTEM A — BASELINE RETRIEVAL"
    )
    print("=" * 80)

    for item in result.baseline:

        print_result(
            item,
            pipeline="baseline",
        )

    # =====================================================
    # PROPOSED
    # =====================================================

    print()
    print()
    print("=" * 80)
    print(
        "SYSTEM B — MARKDOWN-AWARE RETRIEVAL"
    )
    print("=" * 80)

    for item in result.proposed:

        print_result(
            item,
            pipeline="proposed",
        )

    # =====================================================
    # Save
    # =====================================================

    output_path = Path(
        engine.save_result(
            result
        )
    )

    # -----------------------------------------------------
    # Update latest_comparison.json
    # -----------------------------------------------------

    latest_path = (
        output_path.parent
        / "latest_comparison.json"
    )

    shutil.copyfile(
        output_path,
        latest_path,
    )

    print()

    print(
        f"Saved to: {output_path}"
    )

    print(
        f"Latest comparison updated: "
        f"{latest_path}"
    )


if __name__ == "__main__":
    main()