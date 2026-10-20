from __future__ import annotations

import numpy as np

from sentence_transformers import (
    SentenceTransformer,
)


# =========================================================
# Embedding Model
# =========================================================

class EmbeddingModel:

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ):

        self.model_name = model_name

        self.batch_size = batch_size

        self.normalize_embeddings = (
            normalize_embeddings
        )

        print(
            f"Loading embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        self.max_seq_length = (
            self.model.max_seq_length
        )

        print(
            f"Embedding dimension: "
            f"{self.dimension}"
        )

        print(
            f"Max sequence length: "
            f"{self.max_seq_length}"
        )


    # =====================================================
    # E5 formatting
    # =====================================================

    @staticmethod
    def prepare_passage(
        text: str
    ) -> str:

        text = text.strip()

        return (
            f"passage: {text}"
        )


    @staticmethod
    def prepare_query(
        query: str
    ) -> str:

        query = query.strip()

        return (
            f"query: {query}"
        )


    # =====================================================
    # Token length inspection
    # =====================================================

    def count_model_tokens(
        self,
        text: str
    ) -> int:

        token_ids = (
            self.model.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
        )

        return len(token_ids)


    def validate_passages(
        self,
        texts: list[str]
    ) -> None:

        """
        Detect accidental over-length chunks.

        This is important because silent truncation
        would make the experiment less transparent.
        """

        oversized = []

        for index, text in enumerate(texts):

            prepared = self.prepare_passage(
                text
            )

            token_count = (
                self.count_model_tokens(
                    prepared
                )
            )

            if (
                token_count
                > self.max_seq_length
            ):

                oversized.append(
                    (
                        index,
                        token_count,
                    )
                )

        if oversized:

            print()
            print(
                "WARNING: Some embedding inputs "
                "exceed model context length."
            )

            for (
                index,
                token_count,
            ) in oversized[:10]:

                print(
                    f"  chunk={index} "
                    f"tokens={token_count}"
                )

            print(
                "These inputs may be truncated "
                "by the embedding model."
            )

            print()


    # =====================================================
    # Passage embeddings
    # =====================================================

    def encode_passages(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype=np.float32,
            )

        self.validate_passages(
            texts
        )

        prepared_texts = [

            self.prepare_passage(
                text
            )

            for text in texts
        ]

        embeddings = self.model.encode(

            prepared_texts,

            batch_size=(
                self.batch_size
            ),

            show_progress_bar=(
                show_progress_bar
            ),

            convert_to_numpy=True,

            normalize_embeddings=(
                self.normalize_embeddings
            ),
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        return np.ascontiguousarray(
            embeddings
        )


    # =====================================================
    # Query embedding
    # =====================================================

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        prepared_query = (
            self.prepare_query(
                query
            )
        )

        embedding = self.model.encode(

            [prepared_query],

            batch_size=1,

            show_progress_bar=False,

            convert_to_numpy=True,

            normalize_embeddings=(
                self.normalize_embeddings
            ),
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        return np.ascontiguousarray(
            embedding
        )