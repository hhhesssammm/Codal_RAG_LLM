import math

# ============================================
# Mean helper
# ============================================


def safe_mean(
    values: list[float | None],
) -> float | None:

    valid = [
        value
        for value in values
        if value is not None
        and not (
            isinstance(value, float)
            and math.isnan(value)
        )
    ]

    if not valid:
        return None

    return (
        sum(valid)
        / len(valid)
    )

# ============================================
# Aggregate
# ============================================


def aggregate_method(
    detailed_results: list[dict],
    method: str,
    top_k: int = 5,
) -> dict:

    method_results = [
        item[method]
        for item in detailed_results
    ]

    precision_values = [
        item["retrieval"][
            f"precision@{top_k}"
        ]
        for item in method_results
    ]

    recall_values = [
        item["retrieval"][
            f"recall@{top_k}"
        ]
        for item in method_results
    ]

    mrr_values = [
        item["retrieval"][
            "mrr"
        ]
        for item in method_results
    ]

    evidence_hit_values = [
        item["retrieval"][
            "evidence_hit"
        ]
        for item in method_results
    ]

    answer_values = [
        item["answer"][
            "answer_correctness"
        ]
        for item in method_results
    ]

    semantic_values = [
        item["answer"][
            "semantic_correctness"
        ]
        for item in method_results
    ]

    citation_values = [
        item["faithfulness"][
            "citation_faithfulness"
        ][
            "score"
        ]
        for item in method_results
    ]

    semantic_faithfulness = [
        item["faithfulness"][
            "semantic_faithfulness"
        ][
            "score"
        ]
        for item in method_results
    ]

    answer_accuracy_values = [
        1.0
        if item["answer"][
            "answer_correct"
        ]
        else 0.0
        for item in method_results
    ]

    semantic_accuracy_values = [
        (
            1.0
            if item["answer"][
                "semantic_correct"
            ]
            else 0.0
        )
        if item["answer"][
            "semantic_correct"
        ]
        is not None
        else None
        for item in method_results
    ]

    return {

        "questions": len(
            method_results
        ),

        "retrieval": {

            f"precision@{top_k}": (
                safe_mean(
                    precision_values
                )
            ),

            f"recall@{top_k}": (
                safe_mean(
                    recall_values
                )
            ),

            "mrr": (
                safe_mean(
                    mrr_values
                )
            ),

            "evidence_hit_rate": (
                safe_mean(
                    evidence_hit_values
                )
            ),
        },

        "answer": {

            "mean_answer_correctness": (
                safe_mean(
                    answer_values
                )
            ),

            "answer_accuracy": (
                safe_mean(
                    answer_accuracy_values
                )
            ),

            "mean_semantic_correctness": (
                safe_mean(
                    semantic_values
                )
            ),

            "semantic_accuracy": (
                safe_mean(
                    semantic_accuracy_values
                )
            ),
        },

        "faithfulness": {

            "citation_faithfulness": (
                safe_mean(
                    citation_values
                )
            ),

            "semantic_faithfulness": (
                safe_mean(
                    semantic_faithfulness
                )
            ),
        },
    }


def aggregate_by_type(
    detailed_results: list[dict],
    top_k: int = 5,
) -> dict:

    output = {}

    for question_type in (
        "explicit",
        "implicit",
        "latent",
    ):

        subset = [
            item
            for item in detailed_results
            if item[
                "type"
            ]
            == question_type
        ]

        if not subset:
            continue

        output[
            question_type
        ] = {

            "baseline": (
                aggregate_method(
                    subset,
                    "baseline", top_k,
                )
            ),

            "proposed": (
                aggregate_method(
                    subset,
                    "proposed", top_k,
                )
            ),
        }

    return output