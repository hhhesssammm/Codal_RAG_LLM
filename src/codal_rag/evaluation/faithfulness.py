# ============================================
# Faithfulness
# ============================================


def calculate_citation_faithfulness(
    extraction: dict,
    evidence_items: list[dict],
    question_type: str,
) -> dict:
    """
    Faithfulness ساختاری.

    بررسی می‌کند claimهایی که مدل تولید کرده،
    واقعاً Evidence ID معتبر داشته باشند.

    توجه:
    این معیار اثبات کامل معنایی support نیست.
    بلکه Citation Grounding را اندازه می‌گیرد.
    """

    available_ids = {
        item.get("evidence_id")
        for item in evidence_items
        if item.get("evidence_id")
    }

    claims = extraction.get(
        question_type,
        [],
    )

    if not claims:

        return {
            "score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
        }

    supported_claims = 0

    details = []

    for claim in claims:

        evidence_ids = (
            claim.get(
                "evidence_ids"
            )
            or []
        )

        resolved = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id
            in available_ids
        ]

        supported = (
            len(resolved) > 0
        )

        if supported:

            supported_claims += 1

        details.append(
            {
                "claim": claim.get(
                    "claim"
                ),

                "evidence_ids": (
                    evidence_ids
                ),

                "resolved_ids": (
                    resolved
                ),

                "supported": supported,
            }
        )

    score = (
        supported_claims
        / len(claims)
    )

    return {
        "score": score,

        "supported_claims": (
            supported_claims
        ),

        "total_claims": len(
            claims
        ),

        "details": details,
    }


def calculate_semantic_faithfulness(
    extraction: dict,
    evidence_items: list[dict],
    question_type: str,
    semantic_evaluator: SemanticEvaluator,
    threshold: float = 0.50,
) -> dict:
    """
    یک proxy معنایی برای Faithfulness.

    هر Claim فقط با Evidenceهایی مقایسه می‌شود
    که خودش cite کرده است.

    اگر شباهت Claim با حداقل یکی از Evidenceهای
    cite شده از threshold بالاتر باشد،
    claim پشتیبانی‌شده در نظر گرفته می‌شود.

    این معیار جایگزین ارزیابی انسانی یا NLI نیست؛
    برای دمو و مقایسه اولیه کاربرد دارد.
    """

    if not semantic_evaluator.available:

        return {
            "score": None,
            "supported_claims": None,
            "total_claims": None,
        }

    evidence_index = {
        item.get("evidence_id"): item
        for item in evidence_items
        if item.get("evidence_id")
    }

    claims = extraction.get(
        question_type,
        [],
    )

    if not claims:

        return {
            "score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
        }

    supported_claims = 0

    details = []

    for claim_item in claims:

        claim = (
            claim_item.get(
                "claim"
            )
            or ""
        )

        evidence_ids = (
            claim_item.get(
                "evidence_ids"
            )
            or []
        )

        best_score = 0.0
        best_evidence = None

        for evidence_id in evidence_ids:

            evidence = evidence_index.get(
                evidence_id
            )

            if evidence is None:
                continue

            evidence_text = (
                evidence.get(
                    "text"
                )
                or ""
            )

            score = (
                semantic_evaluator.similarity(
                    claim,
                    evidence_text,
                )
            )

            if (
                score is not None
                and score > best_score
            ):

                best_score = score

                best_evidence = (
                    evidence_id
                )

        supported = (
            best_score >= threshold
        )

        if supported:
            supported_claims += 1

        details.append(
            {
                "claim": claim,

                "best_evidence_id": (
                    best_evidence
                ),

                "similarity": (
                    best_score
                ),

                "supported": (
                    supported
                ),
            }
        )

    return {
        "score": (
            supported_claims
            / len(claims)
        ),

        "supported_claims": (
            supported_claims
        ),

        "total_claims": len(
            claims
        ),

        "threshold": threshold,

        "details": details,
    }
