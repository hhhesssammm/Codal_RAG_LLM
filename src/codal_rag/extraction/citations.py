from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


LEVEL_LABELS = {
    "explicit": "اطلاعات صریح",
    "implicit": "اطلاعات ضمنی",
    "latent": "اطلاعات نهفته / مفهومی",
}


LEVEL_ALIASES = {
    "explicit": (
        "explicit",
        "explicit_information",
    ),
    "implicit": (
        "implicit",
        "implicit_information",
    ),
    "latent": (
        "latent",
        "hidden",
        "conceptual",
        "latent_information",
        "latent_conceptual",
    ),
}


TEXT_KEYS = (
    "statement",
    "text",
    "claim",
    "finding",
    "result",
    "analysis",
)


EVIDENCE_KEYS = (
    "evidence_ids",
    "evidence",
    "citations",
    "supporting_evidence",
)


CONCEPT_KEYS = (
    "concept",
    "topic",
    "label",
)


CONFIDENCE_KEYS = (
    "confidence",
    "confidence_level",
)


def _to_dict(obj: Any) -> dict:
    """
    EvidenceItem یا dict را به dict تبدیل می‌کند.
    """

    if isinstance(obj, dict):
        return obj

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    raise TypeError(
        f"Unsupported object type: {type(obj)}"
    )


def _first_value(
    data: dict,
    keys: tuple[str, ...],
    default=None,
):
    """
    اولین کلید موجود را برمی‌گرداند.
    """

    for key in keys:
        value = data.get(key)

        if value is not None:
            return value

    return default


def _normalize_evidence_ids(
    value: Any,
) -> list[str]:
    """
    Evidence IDها را همیشه به list[str] تبدیل می‌کند.
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        evidence_id = (
            value.get("evidence_id")
            or value.get("id")
        )

        if evidence_id:
            return [str(evidence_id)]

        return []

    if isinstance(value, list):
        result = []

        for item in value:

            if isinstance(item, str):
                result.append(item)

            elif isinstance(item, dict):
                evidence_id = (
                    item.get("evidence_id")
                    or item.get("id")
                )

                if evidence_id:
                    result.append(
                        str(evidence_id)
                    )

        return result

    return []


def _normalize_confidence(
    value: Any,
) -> str | None:
    """
    Confidence را به high / medium / low تبدیل می‌کند.
    """

    if value is None:
        return None

    value = str(value).strip().lower()

    mapping = {
        "high": "high",
        "medium": "medium",
        "low": "low",

        "بالا": "high",
        "زیاد": "high",

        "متوسط": "medium",

        "پایین": "low",
        "کم": "low",
    }

    return mapping.get(
        value,
        value,
    )


def _make_snippet(
    text: str | None,
    max_chars: int = 600,
) -> str:
    """
    برای نمایش دمو، متن Evidence را کوتاه می‌کند.
    """

    if not text:
        return ""

    clean_text = " ".join(
        str(text).split()
    )

    if len(clean_text) <= max_chars:
        return clean_text

    return (
        clean_text[:max_chars].rstrip()
        + " ..."
    )


def _get_level_items(
    extraction_result: dict,
    level: str,
) -> list:
    """
    خروجی سه‌سطحی را حتی اگر داخل results یا extraction
    قرار گرفته باشد پیدا می‌کند.
    """

    containers = [
        extraction_result,
    ]

    for key in (
        "results",
        "extraction",
        "output",
        "levels",
    ):

        nested = extraction_result.get(key)

        if isinstance(nested, dict):
            containers.append(nested)

    aliases = LEVEL_ALIASES[level]

    for container in containers:

        for alias in aliases:

            items = container.get(alias)

            if isinstance(items, list):
                return items

    return []


def _build_source_label(
    evidence: dict,
) -> str:
    """
    نام قابل نمایش منبع را فقط از metadata واقعی می‌سازد.
    چیزی را حدس نمی‌زند.
    """

    document_title = (
        evidence.get("document_title")
    )

    report_type = (
        evidence.get("report_type")
    )

    source_file = (
        evidence.get("source_file")
    )

    document_id = (
        evidence.get("document_id")
    )

    year = (
        evidence.get("year")
    )

    base = (
        document_title
        or report_type
        or source_file
    )

    if not base:

        if document_id:
            base = f"سند {document_id}"

        else:
            base = "سند نامشخص"

    if (
        year
        and str(year) not in str(base)
    ):
        base = (
            f"{base} - {year}"
        )

    return str(base)


def _build_evidence_payload(
    evidence_id: str,
    evidence_index: dict[str, dict],
) -> dict:
    """
    یک Evidence ID را به Citation کامل تبدیل می‌کند.
    """

    evidence = evidence_index.get(
        evidence_id
    )

    if evidence is None:

        return {
            "evidence_id": evidence_id,
            "resolved": False,
            "error": (
                "Evidence ID was not found "
                "in evidence context."
            ),
        }

    heading_path = (
        evidence.get("heading_path")
        or []
    )

    if not isinstance(
        heading_path,
        list,
    ):
        heading_path = [
            str(heading_path)
        ]

    heading_path = [
        str(item)
        for item in heading_path
        if item
    ]

    section = (
        evidence.get("section")
    )

    subsection = (
        evidence.get("subsection")
    )

    if (
        not section
        and heading_path
    ):
        section = heading_path[-1]

    text = evidence.get("text")

    return {
        "evidence_id": evidence_id,

        "resolved": True,

        "rank": evidence.get(
            "rank"
        ),

        "chunk_id": evidence.get(
            "chunk_id"
        ),

        "document_id": evidence.get(
            "document_id"
        ),

        "company": evidence.get(
            "company"
        ),

        "year": evidence.get(
            "year"
        ),

        "section": section,

        "subsection": subsection,

        "heading_path": heading_path,

        "path_display": (
            " > ".join(heading_path)
            if heading_path
            else None
        ),

        "chunk_type": evidence.get(
            "chunk_type"
        ),

        "retrieval_score": evidence.get(
            "retrieval_score"
        ),

        "source_file": evidence.get(
            "source_file"
        ),

        "document_title": evidence.get(
            "document_title"
        ),

        "report_type": evidence.get(
            "report_type"
        ),

        "source_label": (
            _build_source_label(
                evidence
            )
        ),

        "snippet": _make_snippet(
            text
        ),

        "full_text": text,
    }


def build_citation_view(
    extraction_result: dict,
    evidence_items: list,
    query: str | None = None,
    method: str | None = None,
) -> dict:
    """
    خروجی استخراج سه‌سطحی را به یک ساختار
    Citation / Evidence قابل نمایش تبدیل می‌کند.
    """

    evidence_index = {}

    for evidence_item in evidence_items:

        evidence = _to_dict(
            evidence_item
        )

        evidence_id = evidence.get(
            "evidence_id"
        )

        if evidence_id:

            evidence_index[
                str(evidence_id)
            ] = evidence

    citation_items = []

    finding_counter = 1

    for level in (
        "explicit",
        "implicit",
        "latent",
    ):

        level_items = _get_level_items(
            extraction_result,
            level,
        )

        for raw_item in level_items:

            if isinstance(
                raw_item,
                str,
            ):
                item = {
                    "statement": raw_item
                }

            elif isinstance(
                raw_item,
                dict,
            ):
                item = raw_item

            else:
                continue

            statement = _first_value(
                item,
                TEXT_KEYS,
            )

            if not statement:
                continue

            evidence_value = (
                _first_value(
                    item,
                    EVIDENCE_KEYS,
                    [],
                )
            )

            evidence_ids = (
                _normalize_evidence_ids(
                    evidence_value
                )
            )

            concept = _first_value(
                item,
                CONCEPT_KEYS,
            )

            if concept:
                concept_source = (
                    "extractor"
                )

            else:
                # فقط fallback نمایشی است.
                # هیچ مفهوم جدیدی ساخته نمی‌شود.
                concept = statement

                concept_source = (
                    "statement_fallback"
                )

            confidence = (
                _normalize_confidence(
                    _first_value(
                        item,
                        CONFIDENCE_KEYS,
                    )
                )
            )

            evidence_payloads = [
                _build_evidence_payload(
                    evidence_id,
                    evidence_index,
                )
                for evidence_id
                in evidence_ids
            ]

            citation_items.append(
                {
                    "finding_id": (
                        f"F{finding_counter}"
                    ),

                    "level": level,

                    "information_type": (
                        LEVEL_LABELS[level]
                    ),

                    "concept": concept,

                    "concept_source": (
                        concept_source
                    ),

                    "statement": statement,

                    "confidence": (
                        confidence
                    ),

                    "evidence_ids": (
                        evidence_ids
                    ),

                    "evidence_count": len(
                        evidence_payloads
                    ),

                    "evidence": (
                        evidence_payloads
                    ),
                }
            )

            finding_counter += 1

    return {
        "query": (
            query
            or extraction_result.get(
                "query"
            )
        ),

        "method": (
            method
            or extraction_result.get(
                "method"
            )
        ),

        "traceability_relation": (
            "Concept -> Evidence -> "
            "Section -> Document"
        ),

        "total_findings": len(
            citation_items
        ),

        "items": citation_items,
    }