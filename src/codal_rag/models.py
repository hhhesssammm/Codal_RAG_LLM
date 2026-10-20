from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional


from dataclasses import dataclass, asdict
from typing import Optional, Any

@dataclass
class DocumentMetadata:
    document_id: str
    company: Optional[str] = None
    year: Optional[str] = None
    document_type: Optional[str] = None
    source_file: Optional[str] = None


@dataclass
class StructuredChunk:
    chunk_id: str

    document_id: str

    pipeline: str

    company: Optional[str]
    year: Optional[str]
    document_type: Optional[str]

    chunk_type: str



    text: str
    retrieval_text: str

    heading_path: list[str] = field(
        default_factory=list
    )

    section: Optional[str] = None
    subsection: Optional[str] = None

    token_count: int = 0

    source_block_ids: list[int] = field(
        default_factory=list
    )

    source_file: Optional[str] = None

    table_part: Optional[int] = None
    table_parts: Optional[int] = None


    def to_dict(self):
        return asdict(self)




@dataclass
class EvidenceItem:
    """
    یک شاهد بازیابی‌شده از سیستم RAG.

    هر EvidenceItem در واقع یک chunk است که
    توسط Retriever پیدا شده و قرار است به LLM داده شود.

    این کلاس علاوه بر متن chunk، اطلاعات ساختاری سند
    مانند heading_path، section و اطلاعات منبع را نیز
    نگه می‌دارد تا در مرحله Citation بتوان رابطه زیر را ساخت:

    Concept
        -> Evidence
        -> Section
        -> Document
    """

    evidence_id: str

    rank: int

    chunk_id: str

    document_id: str

    text: str

    chunk_type: str

    heading_path: list[str]

    company: Optional[str] = None

    year: Optional[str] = None

    section: Optional[str] = None

    subsection: Optional[str] = None

    retrieval_score: Optional[float] = None

    # نام فایل اصلی سند
    source_file: Optional[str] = None

    # عنوان قابل نمایش سند
    document_title: Optional[str] = None

    # نوع گزارش، مثلاً:
    # گزارش هیئت‌مدیره
    # صورت‌های مالی
    # گزارش تفسیری مدیریت
    report_type: Optional[str] = None

    def to_dict(self) -> dict:
        """
        تبدیل EvidenceItem به dictionary
        برای ذخیره در JSON.
        """

        return asdict(self)



# =========================================================
# Retrieval result
# =========================================================

@dataclass
class DualRetrievalResult:

    original_query: str

    normalized_query: str

    top_k: int

    baseline: list[dict]

    proposed: list[dict]
