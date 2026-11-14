from __future__ import annotations

from typing import Any

from codal_rag.extraction.evidence import (
    EvidenceItem,
    build_evidence_context,
)

from codal_rag.llm.client import LLMClient
from codal_rag.extraction.prompts import SYSTEM_PROMPT, V2_EXTENSION
from codal_rag.extraction.validation import assert_valid_extraction

class ThreeLevelExtractor:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *, schema_version: int = 1,
    ):

        self.llm_client = llm_client
        self.schema_version = schema_version


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
                evidence_items, max_chars_per_evidence=None if self.schema_version == 2 else 6000
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
                system_prompt=SYSTEM_PROMPT + (V2_EXTENSION if self.schema_version == 2 else ""),
                user_prompt=user_prompt,
            )
        )

        assert_valid_extraction(result, evidence_items, allow_single_chunk_latent=self.schema_version == 2)
        if self.schema_version == 2:
            for level in ("explicit", "implicit", "latent"):
                for claim in result[level]:
                    labels = claim.get("conceptual_labels")
                    if not isinstance(labels, list) or any(not isinstance(v, str) or not v.strip() for v in labels):
                        raise ValueError("v2 claims require conceptual_labels (a list of nonempty labels, or [])")
        if result.get("query") != query:
            raise ValueError("Extraction query does not match the retrieval query")
        return result
