from __future__ import annotations


from difflib import SequenceMatcher
from codal_rag.evaluation.lexical import normalize_text, tokenize



# ============================================
# Evidence matching
# ============================================


def evidence_match_score(
    gold_text: str,
    retrieved_text: str,
) -> float:
    """
    میزان پوشش یک Gold Evidence توسط chunk بازیابی‌شده.

    خروجی بین صفر و یک است.

    از چند معیار استفاده می‌شود:
    1. substring
    2. token recall
    3. SequenceMatcher

    در ارزیابی اصلی، Gold Evidence بهتر است
    یک یا چند جمله مشخص از گزارش باشد.
    """

    gold = normalize_text(
        gold_text
    )

    retrieved = normalize_text(
        retrieved_text
    )

    if not gold or not retrieved:
        return 0.0

    # ----------------------------------------
    # Exact containment
    # ----------------------------------------

    if gold in retrieved:
        return 1.0

    if (
        retrieved in gold
        and len(retrieved) >= 40
    ):
        return 1.0

    # ----------------------------------------
    # Token recall
    # ----------------------------------------

    gold_tokens = set(
        tokenize(gold)
    )

    retrieved_tokens = set(
        tokenize(retrieved)
    )

    if not gold_tokens:
        token_recall = 0.0

    else:

        token_recall = (
            len(
                gold_tokens
                & retrieved_tokens
            )
            / len(gold_tokens)
        )

    # ----------------------------------------
    # Character similarity
    # ----------------------------------------

    sequence_score = (
        SequenceMatcher(
            None,
            gold,
            retrieved,
        ).ratio()
    )

    # token recall برای chunkهای طولانی
    # معیار مناسب‌تری است.
    return max(
        token_recall,
        sequence_score,
    )



def is_relevant_chunk(
    chunk: dict,
    gold_evidence: list[dict],
    threshold: float = 0.70,
) -> tuple[bool, list[int]]:
    """
    مشخص می‌کند آیا chunk بازیابی‌شده
    حداقل یکی از Gold Evidenceها را پوشش می‌دهد.

    همچنین index شاهدهای match شده را برمی‌گرداند.
    """

    chunk_text = (
        chunk.get("text")
        or chunk.get("content")
        or ""
    )

    chunk_document_id = str(
        chunk.get(
            "document_id",
            ""
        )
    )

    matched_gold = []

    for index, gold in enumerate(
        gold_evidence
    ):

        gold_text = gold.get(
            "text",
            "",
        )

        gold_document_id = str(
            gold.get(
                "document_id",
                ""
            )
        )

        # اگر document_id برای Gold تعریف شده،
        # باید سند نیز یکسان باشد.
        if (
            gold_document_id
            and gold_document_id
            != chunk_document_id
        ):
            continue

        score = evidence_match_score(
            gold_text=gold_text,
            retrieved_text=chunk_text,
        )

        if score >= threshold:

            matched_gold.append(
                index
            )

    return (
        len(matched_gold) > 0,
        matched_gold,
    )

# ============================================
# Retrieval metrics
# ============================================


def calculate_retrieval_metrics(
    retrieval_results: list[dict],
    gold_evidence: list[dict],
    k: int = 5,
    evidence_threshold: float = 0.70,
) -> dict:
    """
    محاسبه:

    Precision@K
    Recall@K
    MRR
    Evidence Hit Rate
    """

    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if not 0 < evidence_threshold <= 1:
        raise ValueError("evidence_threshold must be in (0, 1]")
    top_results = (
        retrieval_results[:k]
    )

    if not gold_evidence:

        return {
            f"precision@{k}": None,
            f"recall@{k}": None,
            "mrr": None,
            "evidence_hit": None,
            "matched_gold_count": 0,
            "gold_evidence_count": 0,
        }

    relevant_retrieved_count = 0

    matched_gold_indexes = set()

    first_relevant_rank = None

    retrieval_details = []

    for rank, result in enumerate(
        top_results,
        start=1,
    ):

        chunk = result.get(
            "chunk",
            result,
        )

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        relevant, matched = (
            is_relevant_chunk(
                chunk=chunk,
                gold_evidence=gold_evidence,
                threshold=evidence_threshold,
            )
        )

        if relevant:

            relevant_retrieved_count += 1

            matched_gold_indexes.update(
                matched
            )

            if first_relevant_rank is None:

                first_relevant_rank = rank

        retrieval_details.append(
            {
                "rank": rank,

                "chunk_id": chunk.get(
                    "chunk_id"
                ),

                "document_id": chunk.get(
                    "document_id"
                ),

                "relevant": relevant,

                "matched_gold_indexes": (
                    matched
                ),
            }
        )

    denominator_precision = max(
        len(top_results),
        1,
    )

    precision = (
        relevant_retrieved_count
        / denominator_precision
    )

    recall = (
        len(matched_gold_indexes)
        / len(gold_evidence)
    )

    if first_relevant_rank is None:

        mrr = 0.0

    else:

        mrr = (
            1.0
            / first_relevant_rank
        )

    evidence_hit = (
        1.0
        if matched_gold_indexes
        else 0.0
    )

    return {
        f"precision@{k}": precision,

        f"recall@{k}": recall,

        "mrr": mrr,

        "evidence_hit": evidence_hit,

        "first_relevant_rank": (
            first_relevant_rank
        ),

        "matched_gold_count": len(
            matched_gold_indexes
        ),

        "gold_evidence_count": len(
            gold_evidence
        ),

        "retrieval_details": (
            retrieval_details
        ),
    }

