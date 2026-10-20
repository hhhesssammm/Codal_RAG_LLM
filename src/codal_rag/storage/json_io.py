from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from src.evaluation_metrics import (
    SemanticEvaluator,
    answer_correctness,
    calculate_citation_faithfulness,
    calculate_retrieval_metrics,
    calculate_semantic_faithfulness,
    safe_mean,
)

def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)




