from __future__ import annotations

from typing import Any

from src.evidence_context import (
    EvidenceItem,
    build_evidence_context,
)

from src.llm_client import LLMClient

class ThreeLevelExtractor:

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ):

        self.llm_client = (
            llm_client
            or LLMClient()
        )


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
            self.llm_client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        )

        return result