from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = Path(
    "data/evaluation/questions.json"
)


def create_question(
    number: int,
    question_type: str,
) -> dict:

    return {
        "id": f"Q{number:03d}",

        "question": "",

        "type": question_type,

        "expected_answer": "",

        "gold_evidence": [
            {
                "document_id": "",
                "text": ""
            }
        ],

        "notes": ""
    }


def main():

    questions = []

    counter = 1

    # -----------------------------
    # Explicit: 20
    # -----------------------------

    for _ in range(20):

        questions.append(
            create_question(
                counter,
                "explicit",
            )
        )

        counter += 1

    # -----------------------------
    # Implicit: 15
    # -----------------------------

    for _ in range(15):

        questions.append(
            create_question(
                counter,
                "implicit",
            )
        )

        counter += 1

    # -----------------------------
    # Latent: 15
    # -----------------------------

    for _ in range(15):

        questions.append(
            create_question(
                counter,
                "latent",
            )
        )

        counter += 1

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            questions,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Created: {OUTPUT_PATH}"
    )

    print(
        "Total questions:",
        len(questions),
    )

    print(
        "Explicit: 20"
    )

    print(
        "Implicit: 15"
    )

    print(
        "Latent: 15"
    )


if __name__ == "__main__":
    main()