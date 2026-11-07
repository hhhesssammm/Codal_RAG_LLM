"""Injectable tokenization; importing the package never downloads a model."""
from __future__ import annotations


class TokenizerWrapper:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", *,
                 revision: str | None = None, local_files_only: bool = False,
                 tokenizer=None):
        self.model_name = model_name
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install codal-rag[models] to tokenize with transformers") from exc
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only)
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))


class WhitespaceTokenizer:
    """Explicit smoke/test tokenizer; not a replacement for model token counts."""
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


def create_tokenizer(config):
    e = config.embedding
    if e.tokenizer == "whitespace":
        return WhitespaceTokenizer()
    return TokenizerWrapper(e.model, revision=e.revision, local_files_only=e.local_files_only)
