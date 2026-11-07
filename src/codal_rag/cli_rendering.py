



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
