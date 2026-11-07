"""Baseline content and Proposed structural context for embedding."""

def get_baseline_embedding_text(
    chunk: dict
) -> str:

    # Support both versions of your previous
    # baseline chunk format.

    text = (
        chunk.get("text")
        or chunk.get("content")
    )

    if not text:

        raise ValueError(
            "Baseline chunk does not "
            "contain text/content."
        )

    return text.strip()


# =========================================================
# Proposed embedding representation
# =========================================================

def get_proposed_embedding_text(
    chunk: dict
) -> str:

    # Preferred representation from Stage 4

    retrieval_text = (
        chunk.get(
            "retrieval_text"
        )
    )

    if retrieval_text:

        return retrieval_text.strip()

    # Fallback only for older chunk files

    text = (
        chunk.get("text")
        or chunk.get("content")
    )

    if not text:

        raise ValueError(
            "Proposed chunk does not contain "
            "retrieval_text/text/content."
        )

    heading_path = (
        chunk.get(
            "heading_path"
        )
        or []
    )

    if heading_path:

        path = " > ".join(
            heading_path
        )

        return (
            f"{path}\n\n{text}"
        )

    return text.strip()


# =========================================================
# Index one pipeline
# =========================================================
