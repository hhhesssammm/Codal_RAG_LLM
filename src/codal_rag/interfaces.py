"""Small dependency-injection contracts for deterministic tests and alternative services."""
from typing import Protocol
import numpy as np


class Embedder(Protocol):
    model_name: str
    dimension: int

    def encode_passages(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray: ...
    def encode_query(self, query: str) -> np.ndarray: ...


class JSONGenerator(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict: ...
