def load_chunks(
    directory: Path
) -> list[dict]:

    records = []

    files = sorted(
        directory.rglob("*.json")
    )

    print(
        f"Found {len(files)} files "
        f"in {directory}"
    )

    for path in files:

        with path.open(
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):

            records.extend(
                data
            )

        elif isinstance(data, dict):

            records.append(
                data
            )

        else:

            raise ValueError(
                f"Invalid JSON structure: "
                f"{path}"
            )

    return records


# =========================================================
# Baseline embedding representation
# =========================================================

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