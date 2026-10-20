from __future__ import annotations

from typing import Any

from src.evidence_context import (
    EvidenceItem,
)


LEVEL_NAMES = (
    "explicit",
    "implicit",
    "latent",
)


def validate_extraction(
    extraction: dict[str, Any],
    evidence_items: list[EvidenceItem],
) -> list[str]:
    """
    ساختار خروجی LLM و ارجاع به Evidenceها را بررسی می‌کند.

    توجه:
    این Validator صحت معنایی claim را تضمین نمی‌کند.
    فقط اعتبار ساختاری و ارجاع به شواهد را بررسی می‌کند.
    """

    errors: list[str] = []

    if not isinstance(
        extraction,
        dict,
    ):

        return [
            "Extraction output must be a dictionary."
        ]

    valid_evidence_ids = {
        evidence.evidence_id
        for evidence in evidence_items
    }

    for level_name in LEVEL_NAMES:

        items = extraction.get(
            level_name
        )

        if items is None:

            errors.append(
                f"Missing field: {level_name}"
            )

            continue

        if not isinstance(
            items,
            list,
        ):

            errors.append(
                f"{level_name} must be a list."
            )

            continue

        for index, item in enumerate(
            items,
            start=1,
        ):

            prefix = (
                f"{level_name}[{index}]"
            )

            if not isinstance(
                item,
                dict,
            ):

                errors.append(
                    f"{prefix} must be an object."
                )

                continue

            claim = item.get(
                "claim"
            )

            if (
                not isinstance(claim, str)
                or not claim.strip()
            ):

                errors.append(
                    f"{prefix}.claim is empty."
                )

            evidence_ids = item.get(
                "evidence_ids"
            )

            if not isinstance(
                evidence_ids,
                list,
            ):

                errors.append(
                    f"{prefix}.evidence_ids "
                    "must be a list."
                )

                continue

            if not evidence_ids:

                errors.append(
                    f"{prefix} has no evidence."
                )

            for evidence_id in evidence_ids:

                if evidence_id not in (
                    valid_evidence_ids
                ):

                    errors.append(
                        f"{prefix} references "
                        f"unknown evidence_id: "
                        f"{evidence_id}"
                    )

            confidence = item.get(
                "confidence"
            )

            if confidence is None:

                errors.append(
                    f"{prefix}.confidence "
                    "is missing."
                )

            else:

                try:
                    confidence_value = float(
                        confidence
                    )

                    if not (
                        0 <= confidence_value <= 1
                    ):

                        errors.append(
                            f"{prefix}.confidence "
                            "must be between 0 and 1."
                        )

                except (
                    TypeError,
                    ValueError,
                ):

                    errors.append(
                        f"{prefix}.confidence "
                        "must be numeric."
                    )

            basis = item.get(
                "basis"
            )

            if (
                not isinstance(basis, str)
                or not basis.strip()
            ):

                errors.append(
                    f"{prefix}.basis is empty."
                )

            if level_name == "latent":

                unique_ids = set(
                    evidence_ids
                )

                if len(unique_ids) < 2:

                    errors.append(
                        f"{prefix} must use "
                        "at least two distinct "
                        "evidence items."
                    )

                disclaimer = item.get(
                    "disclaimer"
                )

                if (
                    not isinstance(
                        disclaimer,
                        str,
                    )
                    or not disclaimer.strip()
                ):

                    errors.append(
                        f"{prefix}.disclaimer "
                        "is missing."
                    )

                elif "استنباط" not in disclaimer:

                    errors.append(
                        f"{prefix}.disclaimer "
                        "must explicitly indicate "
                        "that the claim is an inference."
                    )

    insufficient = extraction.get(
        "insufficient_evidence"
    )

    if not isinstance(
        insufficient,
        bool,
    ):

        errors.append(
            "insufficient_evidence must be boolean."
        )

    return errors


def assert_valid_extraction(
    extraction: dict[str, Any],
    evidence_items: list[EvidenceItem],
) -> None:

    errors = validate_extraction(
        extraction,
        evidence_items,
    )

    if errors:

        message = "\n".join(
            f"- {error}"
            for error in errors
        )

        raise ValueError(
            "Invalid extraction output:\n"
            + message
        )