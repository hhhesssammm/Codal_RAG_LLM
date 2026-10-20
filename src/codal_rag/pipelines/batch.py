from __future__ import annotations

import json
import shutil
import subprocess
import sys

from pathlib import Path


QUESTIONS_PATH = Path(
    "data/evaluation/questions.json"
)

RUNS_DIR = Path(
    "data/evaluation_runs"
)

LATEST_RETRIEVAL = Path(
    "data/retrieval_results/latest_comparison.json"
)

LATEST_BASELINE = Path(
    "data/extraction_results/latest_baseline.json"
)

LATEST_PROPOSED = Path(
    "data/extraction_results/latest_proposed.json"
)


def load_questions() -> list[dict]:

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def run_command(
    command: list[str],
) -> None:

    print()
    print(
        "Running:",
        " ".join(command),
    )

    subprocess.run(
        command,
        check=True,
    )


def validate_question(
    question: dict,
) -> None:

    required = (
        "id",
        "question",
        "type",
        "expected_answer",
        "gold_evidence",
    )

    for field in required:

        if field not in question:

            raise ValueError(
                f"Question missing field: "
                f"{field}"
            )

    if not question[
        "question"
    ].strip():

        raise ValueError(
            f"{question['id']} "
            "has empty question."
        )

    if question["type"] not in (
        "explicit",
        "implicit",
        "latent",
    ):

        raise ValueError(
            f"Invalid question type in "
            f"{question['id']}"
        )


def copy_file(
    source: Path,
    destination: Path,
) -> None:

    if not source.exists():

        raise FileNotFoundError(
            f"Expected file not found: "
            f"{source}"
        )

    shutil.copy2(
        source,
        destination,
    )


def main():

    questions = (
        load_questions()
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(
        questions
    )

    for index, item in enumerate(
        questions,
        start=1,
    ):

        validate_question(
            item
        )

        question_id = item[
            "id"
        ]

        query = item[
            "question"
        ]

        print()
        print(
            "=" * 70
        )

        print(
            f"[{index}/{total}] "
            f"{question_id}"
        )

        print(
            query
        )

        print(
            "=" * 70
        )

        question_dir = (
            RUNS_DIR
            / question_id
        )

        question_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------
        # ذخیره Gold Question
        # ------------------------------------

        with (
            question_dir
            / "gold.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                item,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ====================================
        # Retrieval Comparison
        # ====================================

        run_command(
            [
                sys.executable,

                "search_compare.py",

                "--query",

                query,
            ]
        )

        copy_file(
            LATEST_RETRIEVAL,
            question_dir
            / "retrieval.json",
        )

        # ====================================
        # Baseline extraction
        # ====================================

        run_command(
            [
                sys.executable,

                "run_extraction.py",

                "--method",

                "baseline",

                "--query",

                query,
            ]
        )

        copy_file(
            LATEST_BASELINE,
            question_dir
            / "baseline.json",
        )

        # ====================================
        # Proposed extraction
        # ====================================

        run_command(
            [
                sys.executable,

                "run_extraction.py",

                "--method",

                "proposed",

                "--query",

                query,
            ]
        )

        copy_file(
            LATEST_PROPOSED,
            question_dir
            / "proposed.json",
        )

        print(
            f"\nCompleted: {question_id}"
        )

    print()
    print("=" * 70)

    print(
        "Evaluation pipeline completed."
    )

    print(
        f"Questions processed: {total}"
    )


if __name__ == "__main__":
    main()