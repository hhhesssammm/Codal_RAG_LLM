from __future__ import annotations

import argparse
import json

from datetime import datetime
from pathlib import Path
from typing import Any

from src.citation_builder import build_citation_view
from src.citation_validator import validate_citation_view
from src.citation_renderer import print_citation_report
from src.evidence_context import (
    retrieval_to_evidence,
)

from src.three_level_extractor import (
    ThreeLevelExtractor,
)

from src.extraction_validator import (
    validate_extraction,
)


DEFAULT_INPUT = Path(
    "data/retrieval_results/latest_comparison.json"
)

OUTPUT_DIR = Path(
    "data/extraction_results"
)


def load_json(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def unwrap_results(
    value: Any,
) -> list[dict] | None:
    """
    تلاش می‌کند لیست نتایج Retrieval را
    از چند ساختار متداول JSON استخراج کند.
    """

    if isinstance(
        value,
        list,
    ):

        if not value:
            return []

        if all(
            isinstance(item, dict)
            and "chunk" in item
            for item in value
        ):
            return value

    if isinstance(
        value,
        dict,
    ):

        possible_keys = (
            "results",
            "retrieval_results",
            "hits",
            "items",
        )

        for key in possible_keys:

            if key in value:

                result = unwrap_results(
                    value[key]
                )

                if result is not None:
                    return result

    return None


def find_method_node(
    value: Any,
    method: str,
) -> Any | None:
    """
    به صورت بازگشتی baseline یا proposed
    را در JSON پیدا می‌کند.
    """

    if isinstance(
        value,
        dict,
    ):

        if method in value:
            return value[method]

        alternative_keys = (
            f"{method}_results",
            f"{method}_retrieval",
        )

        for key in alternative_keys:

            if key in value:
                return value[key]

        for child in value.values():

            found = find_method_node(
                child,
                method,
            )

            if found is not None:
                return found

    elif isinstance(
        value,
        list,
    ):

        for child in value:

            found = find_method_node(
                child,
                method,
            )

            if found is not None:
                return found

    return None


def select_retrieval_results(
    payload: Any,
    method: str,
) -> list[dict]:

    # اگر خود فایل مستقیماً لیست نتایج باشد
    direct = unwrap_results(
        payload
    )

    if direct is not None:
        return direct

    method_node = find_method_node(
        payload,
        method,
    )

    if method_node is None:

        if isinstance(
            payload,
            dict,
        ):
            available = list(
                payload.keys()
            )
        else:
            available = []

        raise ValueError(
            f"Could not find '{method}' "
            "retrieval results in JSON.\n"
            f"Top-level keys: {available}"
        )

    results = unwrap_results(
        method_node
    )

    if results is None:
        raise ValueError(
            f"Found '{method}' section "
            "but could not locate a list "
            "of retrieval results containing "
            "a 'chunk' field."
        )

    return results


def save_output(
    data: dict,
    method: str,
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_path = (
        OUTPUT_DIR
        / f"extraction_{method}_{timestamp}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    latest_path = (
        OUTPUT_DIR
        / f"latest_{method}.json"
    )

    with latest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path

def get_retrieval_query(
    payload: Any,
) -> str | None:
    """
    سؤال استفاده‌شده در مرحله Retrieval را
    از فایل JSON استخراج می‌کند.

    برای سازگاری با نسخه‌های مختلف فایل،
    هم original_query و هم query بررسی می‌شوند.
    """

    if not isinstance(
        payload,
        dict,
    ):
        return None

    query = (
        payload.get("original_query")
        or payload.get("query")
    )

    if not isinstance(
        query,
        str,
    ):
        return None

    query = query.strip()

    if not query:
        return None

    return query

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Three-level information extraction "
            "from retrieved financial evidence."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help=(
            "The same question that was used "
            "during retrieval."
        ),
    )

    parser.add_argument(
        "--method",
        choices=(
            "baseline",
            "proposed",
        ),
        default="proposed",
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    print(
        f"Loading retrieval results from: "
        f"{input_path}"
    )

    payload = load_json(
        input_path
    )
    retrieval_query = (
        get_retrieval_query(
            payload
        )
    )

    if retrieval_query is None:
        raise ValueError(
            "The retrieval file does not "
            "contain 'original_query' or 'query'."
        )

    if (
            retrieval_query.strip()
            != args.query.strip()
    ):
        raise ValueError(
            "\nQuery mismatch detected.\n\n"
            f"Retrieval query:\n"
            f"{retrieval_query}\n\n"
            f"Extraction query:\n"
            f"{args.query}\n\n"
            "Run retrieval again using the "
            "same question before extraction."
        )
    retrieval_results = (
        select_retrieval_results(
            payload=payload,
            method=args.method,
        )
    )

    if args.top_k > 0:
        retrieval_results = (
            retrieval_results[
                :args.top_k
            ]
        )

    print(
        "Retrieved items selected:",
        len(retrieval_results),
    )

    evidence_items = (
        retrieval_to_evidence(
            retrieval_results
        )
    )

    print(
        "Evidence items created:",
        len(evidence_items),
    )

    if not evidence_items:

        raise RuntimeError(
            "No valid evidence could be "
            "constructed from retrieval results."
        )

    print("\nEvidence IDs:")

    for evidence in evidence_items:

        print(
            f"  {evidence.evidence_id}"
            f" -> "
            f"{evidence.document_id}"
            f" / "
            f"{evidence.chunk_id}"
        )

    extractor = (
        ThreeLevelExtractor()
    )

    print(
        "\nRunning three-level extraction..."
    )

    extraction = extractor.extract(
        query=args.query,
        evidence_items=evidence_items,
    )

    validation_errors = (
        validate_extraction(
            extraction=extraction,
            evidence_items=evidence_items,
        )
    )

    # ==========================================
    # مرحله 8 - ساخت Citation / Evidence
    # ==========================================

    citation_view = build_citation_view(
        extraction_result=extraction,
        evidence_items=evidence_items,
        query=args.query,
        method=args.method,
    )

    citation_validation = (
        validate_citation_view(
            citation_view
        )
    )

    output = {

        "query": args.query,

        "retrieval_method": args.method,

        "source_retrieval_file": str(
            input_path
        ),

        "top_k": args.top_k,

        "evidence": [
            evidence.to_dict()
            for evidence in evidence_items
        ],

        "extraction": extraction,

        "validation": {
            "valid": (
                    len(validation_errors) == 0
            ),
            "errors": validation_errors,
        },

        # ======================================
        # مرحله 8 - Citation / Evidence
        # ======================================

        "citation_view": citation_view,

        "citation_validation": (
            citation_validation
        ),
    }

    output_path = save_output(
        output,
        method=args.method,
    )

    print(
        f"\nSaved to: {output_path}"
    )

    if validation_errors:

        print(
            "\nWARNING: Extraction validation failed:"
        )

        for error in validation_errors:
            print(
                f"  - {error}"
            )

    else:

        print(
            "\nExtraction validation: OK"
        )

    print("\n============================")
    print("EXPLICIT")
    print("============================")

    for item in extraction.get(
        "explicit",
        [],
    ):

        print(
            "-",
            item.get("claim"),
        )

        print(
            "  Evidence:",
            item.get(
                "evidence_ids"
            ),
        )

    print("\n============================")
    print("IMPLICIT")
    print("============================")

    for item in extraction.get(
        "implicit",
        [],
    ):

        print(
            "-",
            item.get("claim"),
        )

        print(
            "  Evidence:",
            item.get(
                "evidence_ids"
            ),
        )

    print("\n============================")
    print("LATENT / CONCEPTUAL")
    print("============================")

    for item in extraction.get(
        "latent",
        [],
    ):

        print(
            "-",
            item.get("claim"),
        )

        print(
            "  Evidence:",
            item.get(
                "evidence_ids"
            ),
        )

        print(
            "  Disclaimer:",
            item.get(
                "disclaimer"
            ),
        )
    print("\n============================")
    print("EXTRACTION STATUS")
    print("============================")

    print(
        "Insufficient evidence:",
        extraction.get(
            "insufficient_evidence",
            False,
        ),
    )

    print(
        "Notes:",
        extraction.get(
            "notes",
            "",
        ),
    )
    # ==========================================
    # نمایش Citation / Evidence
    # ==========================================

    print_citation_report(
        citation_view
    )

    print("\n============================")
    print("CITATION VALIDATION")
    print("============================")

    print(
        "Valid:",
        citation_validation.get(
            "valid"
        ),
    )

    if citation_validation.get(
            "errors"
    ):

        print("\nErrors:")

        for error in citation_validation[
            "errors"
        ]:
            print(
                "-",
                error,
            )

    if citation_validation.get(
            "warnings"
    ):

        print("\nWarnings:")

        for warning in citation_validation[
            "warnings"
        ]:
            print(
                "-",
                warning,
            )

if __name__ == "__main__":
    main()