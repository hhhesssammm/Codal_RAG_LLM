from __future__ import annotations

import json

from pathlib import Path

import faiss
import numpy as np


# =========================================================
# FAISS Vector Store
# =========================================================

class FaissVectorStore:

    def __init__(
        self,
        dimension: int,
    ):

        self.dimension = dimension

        # -------------------------------------------------
        # Exact inner-product search.
        #
        # Since embeddings are normalized,
        # inner product corresponds to cosine similarity.
        # -------------------------------------------------

        base_index = (
            faiss.IndexFlatIP(
                dimension
            )
        )

        self.index = (
            faiss.IndexIDMap2(
                base_index
            )
        )

        self.metadata: dict[
            str,
            dict
        ] = {}

        self.next_id = 0


    # =====================================================
    # Number of indexed vectors
    # =====================================================

    @property
    def size(self) -> int:

        return int(
            self.index.ntotal
        )


    # =====================================================
    # Add vectors
    # =====================================================

    def add(
        self,
        embeddings: np.ndarray,
        records: list[dict],
    ) -> None:

        if len(embeddings) != len(records):

            raise ValueError(
                "Number of embeddings and "
                "metadata records must match."
            )

        if len(records) == 0:
            return

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        embeddings = (
            np.ascontiguousarray(
                embeddings
            )
        )

        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a "
                "2-dimensional matrix."
            )

        if (
            embeddings.shape[1]
            != self.dimension
        ):

            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        start_id = self.next_id

        end_id = (
            start_id
            + len(records)
        )

        vector_ids = np.arange(
            start_id,
            end_id,
            dtype=np.int64,
        )

        self.index.add_with_ids(
            embeddings,
            vector_ids,
        )

        for vector_id, record in zip(
            vector_ids,
            records,
        ):

            self.metadata[
                str(int(vector_id))
            ] = record

        self.next_id = end_id


    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:

        if self.size == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:

            query_embedding = (
                query_embedding.reshape(
                    1,
                    -1,
                )
            )

        query_embedding = (
            np.ascontiguousarray(
                query_embedding
            )
        )

        actual_k = min(
            top_k,
            self.size,
        )

        scores, vector_ids = (
            self.index.search(
                query_embedding,
                actual_k,
            )
        )

        results = []

        for rank, (
            score,
            vector_id,
        ) in enumerate(
            zip(
                scores[0],
                vector_ids[0],
            ),
            start=1,
        ):

            vector_id = int(
                vector_id
            )

            if vector_id < 0:
                continue

            record = self.metadata.get(
                str(vector_id)
            )

            if record is None:
                continue

            results.append(
                {
                    "rank": rank,

                    "score": float(
                        score
                    ),

                    "vector_id": (
                        vector_id
                    ),

                    "chunk": record,
                }
            )

        return results


    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        directory: str | Path,
        config: dict | None = None,
    ) -> None:

        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # FAISS index

        index_path = (
            directory
            / "index.faiss"
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        # Metadata

        metadata_path = (
            directory
            / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # DB information

        database_info = {
            "dimension": self.dimension,
            "size": self.size,
            "next_id": self.next_id,
        }

        info_path = (
            directory
            / "db_info.json"
        )

        with info_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                database_info,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Experimental config

        if config is not None:

            config_path = (
                directory
                / "config.json"
            )

            with config_path.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    config,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


    # =====================================================
    # Load
    # =====================================================

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FaissVectorStore":

        directory = Path(
            directory
        )

        index_path = (
            directory
            / "index.faiss"
        )

        metadata_path = (
            directory
            / "metadata.json"
        )

        info_path = (
            directory
            / "db_info.json"
        )

        if not index_path.exists():

            raise FileNotFoundError(
                index_path
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                metadata_path
            )

        if not info_path.exists():

            raise FileNotFoundError(
                info_path
            )

        with info_path.open(
            encoding="utf-8",
        ) as f:

            info = json.load(f)

        obj = cls(
            dimension=(
                info["dimension"]
            )
        )

        obj.index = faiss.read_index(
            str(index_path)
        )

        with metadata_path.open(
            encoding="utf-8",
        ) as f:

            obj.metadata = json.load(
                f
            )

        obj.next_id = (
            info.get(
                "next_id",
                obj.index.ntotal,
            )
        )

        return obj