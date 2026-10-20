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

def index_pipeline(
    *,
    name: str,
    chunks: list[dict],
    output_directory: Path,
    embedding_model: EmbeddingModel,
    proposed: bool,
):

    print()
    print("=" * 70)

    print(
        f"INDEXING PIPELINE: "
        f"{name}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Select embedding representation
    # -----------------------------------------------------

    if proposed:

        texts = [

            get_proposed_embedding_text(
                chunk
            )

            for chunk in chunks
        ]

    else:

        texts = [

            get_baseline_embedding_text(
                chunk
            )

            for chunk in chunks
        ]

    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    embeddings = (
        embedding_model
        .encode_passages(
            texts=texts,
            show_progress_bar=True,
        )
    )

    print(
        "Embedding matrix:",
        embeddings.shape,
    )

    # -----------------------------------------------------
    # Add audit information
    # -----------------------------------------------------

    records = []

    for chunk, embedding_text in zip(
        chunks,
        texts,
    ):

        record = dict(
            chunk
        )

        # Save what was actually embedded.
        #
        # This is very useful during evaluation
        # and debugging.

        record[
            "embedding_text"
        ] = embedding_text

        record[
            "embedding_model"
        ] = EMBEDDING_MODEL

        record[
            "pipeline"
        ] = name

        records.append(
            record
        )

    # -----------------------------------------------------
    # Vector Store
    # -----------------------------------------------------

    store = FaissVectorStore(
        dimension=(
            embedding_model.dimension
        )
    )

    store.add(
        embeddings=embeddings,
        records=records,
    )

    # -----------------------------------------------------
    # Save configuration
    # -----------------------------------------------------

    experiment_config = {

        "pipeline": name,

        "embedding_model": (
            EMBEDDING_MODEL
        ),

        "embedding_dimension": (
            embedding_model.dimension
        ),

        "normalize_embeddings": (
            NORMALIZE_EMBEDDINGS
        ),

        "query_prefix": "query: ",

        "passage_prefix": (
            "passage: "
        ),

        "similarity": (
            "cosine_via_normalized_inner_product"
        ),

        "faiss_index": (
            "IndexFlatIP + IndexIDMap2"
        ),

        "number_of_chunks": (
            len(chunks)
        ),
    }

    store.save(
        directory=(
            output_directory
        ),
        config=(
            experiment_config
        ),
    )

    print(
        f"Saved vectors: "
        f"{store.size}"
    )

    print(
        f"Saved to: "
        f"{output_directory}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print()
    print(
        "Loading ONE shared "
        "embedding model..."
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # ONE model object is created and then reused
    # for both pipelines.
    # -----------------------------------------------------

    embedding_model = EmbeddingModel(

        model_name=(
            EMBEDDING_MODEL
        ),

        batch_size=(
            EMBEDDING_BATCH_SIZE
        ),

        normalize_embeddings=(
            NORMALIZE_EMBEDDINGS
        ),
    )

    # -----------------------------------------------------
    # Load chunks
    # -----------------------------------------------------

    baseline_chunks = load_chunks(
        BASELINE_CHUNKS_DIR
    )

    proposed_chunks = load_chunks(
        PROPOSED_CHUNKS_DIR
    )

    if not baseline_chunks:

        raise RuntimeError(
            "No baseline chunks found."
        )

    if not proposed_chunks:

        raise RuntimeError(
            "No proposed chunks found."
        )

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------

    index_pipeline(

        name="baseline",

        chunks=baseline_chunks,

        output_directory=(
            BASELINE_DB_DIR
        ),

        embedding_model=(
            embedding_model
        ),

        proposed=False,
    )

    # -----------------------------------------------------
    # PROPOSED
    # -----------------------------------------------------

    index_pipeline(

        name="proposed",

        chunks=proposed_chunks,

        output_directory=(
            PROPOSED_DB_DIR
        ),

        embedding_model=(
            embedding_model
        ),

        proposed=True,
    )

    print()
    print("=" * 70)
    print(
        "INDEXING COMPLETED"
    )
    print("=" * 70)

    print(
        "Same embedding model:"
    )

    print(
        f"  {EMBEDDING_MODEL}"
    )

    print()

    print(
        f"Baseline DB: "
        f"{BASELINE_DB_DIR}"
    )

    print(
        f"Proposed DB: "
        f"{PROPOSED_DB_DIR}"
    )


if __name__ == "__main__":
    main()