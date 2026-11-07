from __future__ import annotations

from typing import Any

from codal_rag.extraction.evidence import (
    EvidenceItem,
    build_evidence_context,
)

from codal_rag.llm.client import LLMClient
from codal_rag.extraction.prompts import SYSTEM_PROMPT
from codal_rag.extraction.validation import assert_valid_extraction

class ThreeLevelExtractor:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ):

        self.llm_client = llm_client


    def extract(
        self,
        query: str,
        evidence_items: list[EvidenceItem],
    ) -> dict[str, Any]:

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if not evidence_items:

            return {
                "query": query,
                "explicit": [],
                "implicit": [],
                "latent": [],
                "insufficient_evidence": True,
                "notes": (
                    "No retrieved evidence "
                    "was provided."
                ),
            }

        if len({e.evidence_id for e in evidence_items}) != len(evidence_items):
            raise ValueError("Evidence IDs must be unique")
        if any(not e.document_id or not e.chunk_id for e in evidence_items):
            raise ValueError("Evidence must be traceable to a document and chunk")
        evidence_context = (
            build_evidence_context(
                evidence_items
            )
        )

        user_prompt = f"""
سؤال:

{query}


شواهد بازیابی‌شده:

{evidence_context}


اکنون فقط بر اساس همین شواهد،
اطلاعات صریح، ضمنی و نهفته/مفهومی را استخراج کن.

توجه:
هر evidence_id مورد استفاده در پاسخ باید دقیقاً یکی
از شناسه‌های موجود در شواهد بالا باشد.

خروجی فقط JSON معتبر باشد.
"""

        result = (
            (self.llm_client or LLMClient()).generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        )

        assert_valid_extraction(result, evidence_items)
        if result.get("query") != query:
            raise ValueError("Extraction query does not match the retrieval query")
        return result