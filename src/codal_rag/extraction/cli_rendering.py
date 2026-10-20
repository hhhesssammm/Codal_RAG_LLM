from __future__ import annotations


CONFIDENCE_FA = {
    "high": "بالا",
    "medium": "متوسط",
    "low": "پایین",
    None: "نامشخص",
}


def print_citation_report(
    citation_view: dict,
) -> None:

    print()
    print("=" * 70)
    print("CITATION / EVIDENCE TRACEABILITY")
    print("=" * 70)

    print(
        "Query:",
        citation_view.get(
            "query"
        )
    )

    print(
        "Method:",
        citation_view.get(
            "method"
        )
    )

    print(
        "Relation:",
        citation_view.get(
            "traceability_relation"
        )
    )

    for item in citation_view.get(
        "items",
        [],
    ):

        print()
        print("-" * 70)

        print(
            "Finding ID:",
            item.get(
                "finding_id"
            )
        )

        print(
            "مفهوم:",
            item.get(
                "concept"
            )
        )

        print(
            "نوع:",
            item.get(
                "information_type"
            )
        )

        print(
            "Confidence:",
            CONFIDENCE_FA.get(
                item.get(
                    "confidence"
                ),
                item.get(
                    "confidence"
                ),
            )
        )

        print(
            "نتیجه:",
            item.get(
                "statement"
            )
        )

        evidence_list = item.get(
            "evidence",
            [],
        )

        for index, evidence in enumerate(
            evidence_list,
            start=1,
        ):

            print()
            print(
                f"شاهد {index}:"
            )

            print(
                evidence.get(
                    "snippet"
                )
            )

            print(
                "Evidence ID:",
                evidence.get(
                    "evidence_id"
                )
            )

            print(
                "Chunk:",
                evidence.get(
                    "chunk_id"
                )
            )

            print(
                "منبع:",
                evidence.get(
                    "source_label"
                )
            )

            path_display = (
                evidence.get(
                    "path_display"
                )
            )

            if path_display:

                print(
                    "مسیر:",
                    path_display
                )

            else:

                print(
                    "مسیر:",
                    "ساختار بخش موجود نیست"
                )