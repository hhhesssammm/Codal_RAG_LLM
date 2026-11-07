from __future__ import annotations

import re


# ============================================
# Persian text normalization
# ============================================


def normalize_text(
    text: str | None,
) -> str:
    """
    نرمال‌سازی سبک برای مقایسه متون فارسی.

    هدف این تابع تغییر محتوای معنایی نیست.
    فقط تفاوت‌های نگارشی ساده کاهش داده می‌شوند.
    """

    if not text:
        return ""

    text = str(text)

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "\u200c": " ",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    # حذف Markdown heading markers
    text = re.sub(
        r"#+",
        " ",
        text,
    )

    # فاصله‌های اضافی
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()

# ============================================
# Tokenization
# ============================================


def tokenize(
    text: str | None,
) -> list[str]:

    text = normalize_text(
        text
    )

    return re.findall(
        r"[\w\u0600-\u06FF]+",
        text,
    )



# ============================================
# Answer correctness
# ============================================


def token_f1(
    expected: str,
    predicted: str,
) -> float:
    """
    F1 ساده در سطح token.

    برای Answer Correctness پایه استفاده می‌شود.
    """

    expected_tokens = (
        tokenize(expected)
    )

    predicted_tokens = (
        tokenize(predicted)
    )

    if (
        not expected_tokens
        or not predicted_tokens
    ):
        return 0.0

    expected_counts = {}

    for token in expected_tokens:

        expected_counts[token] = (
            expected_counts.get(
                token,
                0,
            )
            + 1
        )

    predicted_counts = {}

    for token in predicted_tokens:

        predicted_counts[token] = (
            predicted_counts.get(
                token,
                0,
            )
            + 1
        )

    common = 0

    for token, count in (
        expected_counts.items()
    ):

        common += min(
            count,
            predicted_counts.get(
                token,
                0,
            ),
        )

    if common == 0:
        return 0.0

    precision = (
        common
        / len(predicted_tokens)
    )

    recall = (
        common
        / len(expected_tokens)
    )

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def answer_correctness(
    expected_answer: str,
    candidate_answers: list[str],
) -> dict:
    """
    چون یک سطح استخراج ممکن است چند claim داشته باشد،
    بهترین claim نسبت به جواب Gold انتخاب می‌شود.
    """

    if not candidate_answers:

        return {
            "score": 0.0,
            "best_answer": None,
        }

    best_score = 0.0
    best_answer = None

    for answer in candidate_answers:

        score = token_f1(
            expected=expected_answer,
            predicted=answer,
        )

        if score > best_score:

            best_score = score
            best_answer = answer

    return {
        "score": best_score,
        "best_answer": best_answer,
    }


# ============================================
# Extraction parsing
# ============================================


def get_candidate_answers(
    extraction_output: dict,
    question_type: str,
) -> list[str]:

    extraction = (
        extraction_output.get(
            "extraction"
        )
        or {}
    )

    items = extraction.get(
        question_type,
        [],
    )

    answers = []

    for item in items:

        if isinstance(
            item,
            str,
        ):

            text = item

        elif isinstance(
            item,
            dict,
        ):

            text = (
                item.get(
                    "claim"
                )
                or item.get(
                    "statement"
                )
                or ""
            )

        else:
            continue

        text = str(
            text
        ).strip()

        if text:

            answers.append(
                text
            )

    return answers