from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    TOP_K,
)

from src.embedding_model import (
    EmbeddingModel,
)

from src.faiss_store import (
    FaissVectorStore,
)

from src.persian_normalizer import (
    NormalizerConfig,
    normalize_plain_text,
)


# =========================================================
# Paths
# =========================================================

BASELINE_DB_DIR = Path(
    "data/vector_db/baseline"
)

PROPOSED_DB_DIR = Path(
    "data/vector_db/proposed"
)

RESULTS_DIR = Path(
    "data/retrieval_results"
)


# =========================================================
# Query normalization
# =========================================================

QUERY_NORMALIZER_CONFIG = NormalizerConfig(

    digit_style="ascii",

    normalize_dates=True,

    normalize_financial_units=True,

    normalize_half_space=True,

    remove_diacritics=True,

    normalize_whitespace=True,
)


# =========================================================
# Retrieval result
# =========================================================

@dataclass
class DualRetrievalResult:

    original_query: str

    normalized_query: str

    top_k: int

    baseline: list[dict]

    proposed: list[dict]


# =========================================================
# Retrieval Engine
# =========================================================

class DualRetrievalEngine:

    def __init__(
        self,
        baseline_db_dir: Path = BASELINE_DB_DIR,
        proposed_db_dir: Path = PROPOSED_DB_DIR,
    ):


        print("=" * 70)
        print("Initializing Retrieval Engine")
        print("=" * 70)

        # -------------------------------------------------
        # ONE embedding model
        # -------------------------------------------------

        print(
            "Loading shared embedding model..."
        )

        self.embedding_model = EmbeddingModel(

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

        # -------------------------------------------------
        # Validate index configurations
        # -------------------------------------------------

        print(
            "Checking index configurations..."
        )

        baseline_config = (
            self.read_index_config(
                baseline_db_dir
            )
        )

        proposed_config = (
            self.read_index_config(
                proposed_db_dir
            )
        )

        baseline_model = (
            baseline_config.get(
                "embedding_model"
            )
        )

        proposed_model = (
            proposed_config.get(
                "embedding_model"
            )
        )

        if baseline_model is None:
            raise ValueError(
                "Baseline config does not contain "
                "'embedding_model'."
            )

        if proposed_model is None:
            raise ValueError(
                "Proposed config does not contain "
                "'embedding_model'."
            )

        if (
                baseline_model
                != proposed_model
        ):
            raise ValueError(
                "Baseline and Proposed indexes were "
                "built with different embedding models. "
                f"Baseline={baseline_model}, "
                f"Proposed={proposed_model}"
            )

        if (
                baseline_model
                != EMBEDDING_MODEL
        ):
            raise ValueError(
                "Current embedding model does not "
                "match the indexed vectors. "
                f"Index={baseline_model}, "
                f"Current={EMBEDDING_MODEL}"
            )

        # -------------------------------------------------
        # Load Baseline FAISS
        # -------------------------------------------------

        print(
            "Loading baseline FAISS index..."
        )

        self.baseline_store = (
            FaissVectorStore.load(
                baseline_db_dir
            )
        )

        # -------------------------------------------------
        # Load Proposed FAISS
        # -------------------------------------------------

        print(
            "Loading proposed FAISS index..."
        )

        self.proposed_store = (
            FaissVectorStore.load(
                proposed_db_dir
            )
        )

        # =====================================================
        # Read index config
        # =====================================================

        @staticmethod
        def read_index_config(
                directory: Path
        ) -> dict:
            path = (
                    directory
                    / "config.json"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing index config: {path}"
                )

            with path.open(
                    encoding="utf-8"
            ) as f:
                return json.load(f)
        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        self._validate_indexes()

        print()
        print(
            f"Baseline vectors: "
            f"{self.baseline_store.size}"
        )

        print(
            f"Proposed vectors: "
            f"{self.proposed_store.size}"
        )

        print()
        print(
            "Retrieval Engine ready."
        )


    # =====================================================
    # Validation
    # =====================================================

    def _validate_indexes(
        self
    ) -> None:

        model_dimension = (
            self.embedding_model.dimension
        )

        baseline_dimension = (
            self.baseline_store.dimension
        )

        proposed_dimension = (
            self.proposed_store.dimension
        )

        if (
            baseline_dimension
            != model_dimension
        ):

            raise ValueError(
                "Baseline index dimension "
                "does not match embedding model. "
                f"Index={baseline_dimension}, "
                f"Model={model_dimension}"
            )

        if (
            proposed_dimension
            != model_dimension
        ):

            raise ValueError(
                "Proposed index dimension "
                "does not match embedding model. "
                f"Index={proposed_dimension}, "
                f"Model={model_dimension}"
            )

        if (
            baseline_dimension
            != proposed_dimension
        ):

            raise ValueError(
                "Baseline and Proposed indexes "
                "have different dimensions."
            )


    # =====================================================
    # Query normalization
    # =====================================================

    @staticmethod
    def normalize_query(
        query: str
    ) -> str:

        return normalize_plain_text(

            query,

            QUERY_NORMALIZER_CONFIG,
        )


    # =====================================================
    # Main retrieval
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> DualRetrievalResult:

        if not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        # -------------------------------------------------
        # Normalize ONCE
        # -------------------------------------------------

        normalized_query = (
            self.normalize_query(
                query
            )
        )

        # -------------------------------------------------
        # Embed ONCE
        # -------------------------------------------------

        query_embedding = (
            self.embedding_model
            .encode_query(
                normalized_query
            )
        )

        # -------------------------------------------------
        # EXACT SAME VECTOR -> BASELINE
        # -------------------------------------------------

        baseline_results = (
            self.baseline_store.search(

                query_embedding=(
                    query_embedding
                ),

                top_k=top_k,
            )
        )

        # -------------------------------------------------
        # EXACT SAME VECTOR -> PROPOSED
        # -------------------------------------------------

        proposed_results = (
            self.proposed_store.search(

                query_embedding=(
                    query_embedding
                ),

                top_k=top_k,
            )
        )

        return DualRetrievalResult(

            original_query=query,

            normalized_query=(
                normalized_query
            ),

            top_k=top_k,

            baseline=(
                baseline_results
            ),

            proposed=(
                proposed_results
            ),
        )


    # =====================================================
    # compare
    # =====================================================
    @staticmethod
    def read_index_config(
            directory: Path
    ) -> dict:

        path = (
                directory
                / "config.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing index config: {path}"
            )

        with path.open(
                encoding="utf-8"
        ) as f:
            return json.load(f)
    # =====================================================
    # Save experiment
    # =====================================================

    def save_result(
        self,
        result: DualRetrievalResult,
    ) -> Path:

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        output_path = (

            RESULTS_DIR

            / (
                f"retrieval_"
                f"{timestamp}.json"
            )
        )

        data = {

            "timestamp": timestamp,

            "original_query": (
                result.original_query
            ),

            "normalized_query": (
                result.normalized_query
            ),

            "embedding_model": (
                EMBEDDING_MODEL
            ),

            "top_k": (
                result.top_k
            ),

            "baseline": (
                result.baseline
            ),

            "proposed": (
                result.proposed
            ),
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output_path


