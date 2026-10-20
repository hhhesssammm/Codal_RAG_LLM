from __future__ import annotations


def validate_citation_view(
    citation_view: dict,
) -> dict:

    errors = []
    warnings = []

    method = citation_view.get(
        "method"
    )

    items = citation_view.get(
        "items",
        [],
    )

    if not isinstance(items, list):

        return {
            "valid": False,
            "errors": [
                "citation_view.items must be a list."
            ],
            "warnings": [],
        }

    for item in items:

        finding_id = (
            item.get("finding_id")
            or "UNKNOWN"
        )

        statement = item.get(
            "statement"
        )

        if not statement:

            errors.append(
                f"{finding_id}: "
                "missing statement."
            )

        evidence = item.get(
            "evidence",
            [],
        )

        if not evidence:

            warnings.append(
                f"{finding_id}: "
                "no evidence attached."
            )

        for ev in evidence:

            evidence_id = (
                ev.get("evidence_id")
            )

            if not ev.get(
                "resolved",
                False,
            ):

                errors.append(
                    f"{finding_id}: "
                    f"{evidence_id} "
                    "could not be resolved."
                )

                continue

            if not ev.get(
                "document_id"
            ):

                warnings.append(
                    f"{finding_id}/"
                    f"{evidence_id}: "
                    "document_id is missing."
                )

            if not ev.get(
                "chunk_id"
            ):

                warnings.append(
                    f"{finding_id}/"
                    f"{evidence_id}: "
                    "chunk_id is missing."
                )

            heading_path = ev.get(
                "heading_path",
                [],
            )

            # برای روش پیشنهادی انتظار داریم
            # ساختار Markdown قابل ردیابی باشد.
            if (
                method == "proposed"
                and not heading_path
            ):

                warnings.append(
                    f"{finding_id}/"
                    f"{evidence_id}: "
                    "heading_path is empty "
                    "for proposed method."
                )

    return {
        "valid": (
            len(errors) == 0
        ),

        "errors": errors,

        "warnings": warnings,

        "error_count": len(
            errors
        ),

        "warning_count": len(
            warnings
        ),
    }