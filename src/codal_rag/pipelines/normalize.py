from pathlib import Path

from codal_rag.normalization.persian import normalize_markdown, assert_markdown_structure_preserved


def markdown_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown input not found: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*.md"))
    if not files or any(p.suffix.lower() != ".md" for p in files):
        raise ValueError(f"Expected one or more Markdown (.md) documents: {path}")
    return files


def normalize_documents(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    source = Path(input_path)
    files = markdown_files(source)
    root = source if source.is_dir() else source.parent
    outputs = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        normalized = normalize_markdown(text)
        assert_markdown_structure_preserved(text, normalized)
        destination = Path(output_dir) / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(normalized, encoding="utf-8")
        outputs.append(destination)
    return outputs
