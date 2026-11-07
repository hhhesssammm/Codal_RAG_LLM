from __future__ import annotations

from typing import Optional, Any


from codal_rag.models import EvidenceItem


def _get_value(
    chunk: dict,
    metadata: dict,
    *keys: str,
    default=None,
):
    """
    مقدار موردنظر را ابتدا از خود chunk
    و سپس از metadata استخراج می‌کند.

    این کار باعث می‌شود اگر نسخه‌های مختلف
    chunker اطلاعات را در مکان‌های متفاوتی
    ذخیره کرده باشند، کد همچنان کار کند.
    """

    for key in keys:

        if key in chunk:

            value = chunk.get(
                key
            )

            if value is not None:
                return value

        if key in metadata:

            value = metadata.get(
                key
            )

            if value is not None:
                return value

    return default


def _normalize_optional_string(
    value: Any,
) -> Optional[str]:
    """
    مقدار اختیاری metadata را به رشته تمیز
    تبدیل می‌کند.

    اگر مقدار None یا رشته خالی باشد،
    None برگردانده می‌شود.
    """

    if value is None:
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    return value


def _normalize_heading_path(
    heading_path: Any,
) -> list[str]:
    """
    heading_path را همیشه به list[str]
    استاندارد تبدیل می‌کند.
    """

    if heading_path is None:
        return []

    if isinstance(
        heading_path,
        str,
    ):

        heading_path = [
            heading_path
        ]

    if not isinstance(
        heading_path,
        list,
    ):
        return []

    normalized = []

    for item in heading_path:

        if item is None:
            continue

        value = str(
            item
        ).strip()

        if value:
            normalized.append(
                value
            )

    return normalized


def _normalize_retrieval_score(
    result: dict,
) -> Optional[float]:
    """
    امتیاز Retrieval را از چند نام احتمالی
    استخراج کرده و به float تبدیل می‌کند.
    """

    possible_keys = (
        "retrieval_score",
        "score",
        "similarity_score",
        "distance_score",
    )

    retrieval_score = None

    for key in possible_keys:

        value = result.get(
            key
        )

        if value is not None:

            retrieval_score = value
            break

    if retrieval_score is None:
        return None

    try:

        return float(
            retrieval_score
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def retrieval_to_evidence(
    retrieval_results: list[dict[str, Any]],
) -> list[EvidenceItem]:
    """
    خروجی Retriever را به EvidenceItem تبدیل می‌کند.

    ساختار اصلی مورد انتظار:

    {
        "score": 0.82,

        "chunk": {

            "chunk_id": "...",

            "document_id": "...",

            "text": "...",

            "chunk_type": "...",

            "heading_path": [...],

            "metadata": {
                "company": "...",
                "year": "...",
                "source_file": "...",
                "document_title": "...",
                "report_type": "..."
            }
        }
    }

    نکته:
    وجود همه metadataها اجباری نیست.
    در صورت نبود اطلاعات، مقدار None ذخیره می‌شود
    و سیستم چیزی را حدس نمی‌زند.
    """

    evidence_items: list[
        EvidenceItem
    ] = []

    for index, result in enumerate(
        retrieval_results,
        start=1,
    ):

        # -----------------------------
        # کنترل ساختار نتیجه Retrieval
        # -----------------------------

        if not isinstance(
            result,
            dict,
        ):
            continue

        chunk = result.get(
            "chunk"
        )

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        # -----------------------------
        # استخراج metadata
        # -----------------------------

        chunk_metadata = (
            chunk.get(
                "metadata"
            )
            or {}
        )

        if not isinstance(
            chunk_metadata,
            dict,
        ):
            chunk_metadata = {}

        result_metadata = (
            result.get(
                "metadata"
            )
            or {}
        )

        if not isinstance(
            result_metadata,
            dict,
        ):
            result_metadata = {}

        # metadata موجود در chunk اولویت دارد.
        metadata = {
            **result_metadata,
            **chunk_metadata,
        }

        # -----------------------------
        # استخراج متن chunk
        # -----------------------------

        text = (
            chunk.get(
                "text"
            )
            or chunk.get(
                "content"
            )
            or chunk.get(
                "page_content"
            )
            or ""
        )

        text = str(
            text
        ).strip()

        # Evidence بدون متن قابل استفاده نیست.
        if not text:
            continue

        # -----------------------------
        # heading_path
        # -----------------------------

        heading_path = (
            _get_value(
                chunk,
                metadata,
                "heading_path",
                "headings",
                default=[],
            )
        )

        heading_path = (
            _normalize_heading_path(
                heading_path
            )
        )

        # -----------------------------
        # Retrieval score
        # -----------------------------

        retrieval_score = (
            _normalize_retrieval_score(
                result
            )
        )

        # -----------------------------
        # Chunk ID
        # -----------------------------

        chunk_id = _get_value(
            chunk,
            metadata,
            "chunk_id",
            "id",
            default=(
                f"chunk_{index}"
            ),
        )

        # -----------------------------
        # Document ID
        # -----------------------------

        document_id = _get_value(
            chunk,
            metadata,
            "document_id",
            "doc_id",
            default=None,
        )

        # اگر document_id وجود نداشت،
        # از اطلاعات فایل به عنوان fallback استفاده می‌شود.
        if document_id is None:

            document_id = (
                _get_value(
                    chunk,
                    metadata,
                    "source",
                    "source_file",
                    "file_name",
                    "filename",
                    default=(
                        "unknown_document"
                    ),
                )
            )

        # -----------------------------
        # اطلاعات شرکت
        # -----------------------------

        company = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "company",
                    "company_name",
                )
            )
        )

        # -----------------------------
        # سال گزارش
        # -----------------------------

        year = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "year",
                    "fiscal_year",
                    "report_year",
                )
            )
        )

        # -----------------------------
        # Section
        # -----------------------------

        section = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "section",
                    "section_title",
                )
            )
        )

        # -----------------------------
        # Subsection
        # -----------------------------

        subsection = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "subsection",
                    "subsection_title",
                )
            )
        )

        # -----------------------------
        # Source File
        # -----------------------------

        source_file = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "source_file",
                    "file_name",
                    "filename",
                    "source",
                    "file_path",
                    "source_path",
                )
            )
        )

        # -----------------------------
        # Document Title
        # -----------------------------

        document_title = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "document_title",
                    "report_title",
                    "title",
                )
            )
        )

        # -----------------------------
        # Report Type
        # -----------------------------

        report_type = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "report_type",
                    "document_type",
                    "doc_type",
                )
            )
        )

        # -----------------------------
        # Chunk Type
        # -----------------------------

        chunk_type = (
            _normalize_optional_string(
                _get_value(
                    chunk,
                    metadata,
                    "chunk_type",
                    "type",
                    default="text",
                )
            )
            or "text"
        )

        # -----------------------------
        # ساخت EvidenceItem
        # -----------------------------

        evidence = EvidenceItem(

            evidence_id=(
                f"E{index}"
            ),

            rank=index,

            chunk_id=str(
                chunk_id
            ),

            document_id=str(
                document_id
            ),

            text=text,

            chunk_type=(
                chunk_type
            ),

            heading_path=(
                heading_path
            ),

            company=company,

            year=year,

            section=section,

            subsection=subsection,

            retrieval_score=(
                retrieval_score
            ),

            source_file=(
                source_file
            ),

            document_title=(
                document_title
            ),

            report_type=(
                report_type
            ),
        )

        evidence_items.append(
            evidence
        )

    return evidence_items


def build_evidence_context(
    evidence_items: list[EvidenceItem],
    max_chars_per_evidence: int = 6000,
) -> str:
    """
    Evidenceها را به یک Context ساختاریافته
    و قابل خواندن برای LLM تبدیل می‌کند.

    اطلاعات ساختاری Markdown مانند:

    - heading_path
    - section
    - subsection

    همراه با اطلاعات منبع مانند:

    - source_file
    - document_title
    - report_type

    وارد Context می‌شوند.

    این metadataها به LLM کمک می‌کنند بفهمد
    هر شاهد متعلق به کدام بخش و کدام سند است.

    نکته مهم:
    LLM نباید خودش source یا heading را حدس بزند.
    این اطلاعات مستقیماً از metadata واقعی chunk
    وارد Context می‌شوند.
    """

    blocks: list[str] = []

    for evidence in evidence_items:

        # -----------------------------
        # محدودسازی اندازه متن
        # -----------------------------

        text = evidence.text

        if (
            len(text)
            > max_chars_per_evidence
        ):

            text = (
                text[
                    :max_chars_per_evidence
                ]
                + "\n...[TRUNCATED]"
            )

        # -----------------------------
        # تبدیل مسیر Markdown به رشته
        # -----------------------------

        heading = " > ".join(
            evidence.heading_path
        )

        # -----------------------------
        # اطلاعات پایه Evidence
        # -----------------------------

        lines = [

            f"[{evidence.evidence_id}]",

            f"rank: {evidence.rank}",

            (
                "document_id: "
                f"{evidence.document_id}"
            ),

            (
                "chunk_id: "
                f"{evidence.chunk_id}"
            ),

            (
                "chunk_type: "
                f"{evidence.chunk_type}"
            ),
        ]

        # -----------------------------
        # اطلاعات سند
        # -----------------------------

        if evidence.company:

            lines.append(
                "company: "
                f"{evidence.company}"
            )

        if evidence.year:

            lines.append(
                "year: "
                f"{evidence.year}"
            )

        if evidence.report_type:

            lines.append(
                "report_type: "
                f"{evidence.report_type}"
            )

        if evidence.document_title:

            lines.append(
                "document_title: "
                f"{evidence.document_title}"
            )

        if evidence.source_file:

            lines.append(
                "source_file: "
                f"{evidence.source_file}"
            )

        # -----------------------------
        # اطلاعات ساختاری Markdown
        # -----------------------------

        if evidence.section:

            lines.append(
                "section: "
                f"{evidence.section}"
            )

        if evidence.subsection:

            lines.append(
                "subsection: "
                f"{evidence.subsection}"
            )

        if heading:

            lines.append(
                "heading_path: "
                f"{heading}"
            )

        # -----------------------------
        # امتیاز Retrieval
        # -----------------------------

        if (
            evidence.retrieval_score
            is not None
        ):

            lines.append(
                "retrieval_score: "
                f"{evidence.retrieval_score:.6f}"
            )

        # -----------------------------
        # متن اصلی Evidence
        # -----------------------------

        lines.extend(
            [
                "text:",
                text,
            ]
        )

        blocks.append(
            "\n".join(
                lines
            )
        )

    # -----------------------------
    # ترکیب Evidenceها
    # -----------------------------

    return (
        "\n\n"
        "--------------------"
        "\n\n"
    ).join(
        blocks
    )