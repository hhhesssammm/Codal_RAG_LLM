# ============================================
# Semantic correctness
# ============================================


class SemanticEvaluator:
    """
    Semantic Similarity با SentenceTransformer.

    این بخش اختیاری است.
    اگر sentence-transformers نصب نباشد،
    metric با مقدار None گزارش می‌شود.
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    ):

        self.model_name = (
            model_name
        )

        self.model = None

        self.available = False

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

            self.model = (
                SentenceTransformer(
                    model_name
                )
            )

            self.available = True

        except Exception as exc:

            print(
                "WARNING: Semantic evaluator "
                "is unavailable:"
            )

            print(
                exc
            )

    def similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float | None:

        if not self.available:
            return None

        import numpy as np

        embeddings = (
            self.model.encode(
                [
                    text_a,
                    text_b,
                ],
                normalize_embeddings=True,
            )
        )

        score = float(
            np.dot(
                embeddings[0],
                embeddings[1],
            )
        )

        return score

    def best_similarity(
        self,
        expected_answer: str,
        candidate_answers: list[str],
    ) -> dict:

        if not self.available:

            return {
                "score": None,
                "best_answer": None,
            }

        best_score = -1.0
        best_answer = None

        for answer in candidate_answers:

            score = self.similarity(
                expected_answer,
                answer,
            )

            if (
                score is not None
                and score > best_score
            ):

                best_score = score
                best_answer = answer

        if best_score < 0:

            best_score = None

        return {
            "score": best_score,
            "best_answer": best_answer,
        }