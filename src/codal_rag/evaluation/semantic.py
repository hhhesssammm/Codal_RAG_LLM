"""Optional embedding similarity, never a factual entailment judgment."""
import numpy as np


class SemanticEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 *, enabled: bool = False, model=None):
        self.model_name = model_name
        self.model = model
        self.reason = None
        if model is None and enabled:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.reason = "Semantic model unavailable; install models extra and cache the specified model"
        elif model is None:
            self.reason = "Semantic evaluation disabled"
        self.available = self.model is not None

    def similarity(self, text_a: str, text_b: str) -> float | None:
        if not self.available:
            return None
        vectors = np.asarray(self.model.encode([text_a, text_b], normalize_embeddings=True))
        score = float(np.dot(vectors[0], vectors[1]))
        if not np.isfinite(score):
            raise ValueError("Semantic model returned a non-finite score")
        return float(np.clip(score, -1, 1))

    def best_similarity(self, expected_answer: str, candidate_answers: list[str]) -> dict:
        if not self.available:
            return {"score": None, "best_answer": None}
        if not candidate_answers:
            return {"score": 0.0, "best_answer": None}
        scored = [(self.similarity(expected_answer, answer), answer) for answer in candidate_answers]
        score, answer = max(scored, key=lambda item: item[0])
        return {"score": score, "best_answer": answer}
