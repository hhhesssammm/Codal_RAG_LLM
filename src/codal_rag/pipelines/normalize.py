from pathlib import Path

from src.persian_normalizer import (
    NormalizerConfig,
    normalize_markdown,
    assert_markdown_structure_preserved,
)


INPUT_DIR = Path(
    "data/raw_markdown"
)

OUTPUT_DIR = Path(
    "data/normalized_markdown"
)


config = NormalizerConfig(

    # Recommendation for internal RAG representation
    digit_style="ascii",

    normalize_dates=True,

    normalize_financial_units=True,

    normalize_half_space=True,

    remove_diacritics=True,

    normalize_whitespace=True,
)


def process_file(
    input_path: Path,
    output_path: Path
):

    print(
        f"Processing: {input_path.name}"
    )

    raw_markdown = input_path.read_text(
        encoding="utf-8"
    )

    normalized = normalize_markdown(
        raw_markdown,
        config
    )

    # VERY IMPORTANT
    assert_markdown_structure_preserved(
        raw_markdown,
        normalized
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        normalized,
        encoding="utf-8"
    )

    print(
        f"Saved: {output_path}"
    )


def main():

    files = list(
        INPUT_DIR.rglob("*.md")
    )
    print(files)
    print(
        f"{len(files)} markdown files found."
    )

    for input_path in files:

        relative_path = input_path.relative_to(
            INPUT_DIR
        )

        output_path = (
            OUTPUT_DIR
            / relative_path
        )

        try:

            process_file(
                input_path,
                output_path
            )

        except Exception as e:

            print(
                f"ERROR in {input_path}: {e}"
            )


if __name__ == "__main__":
    main()