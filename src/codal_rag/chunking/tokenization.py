from transformers import AutoTokenizer
import config




class TokenizerWrapper:

    def __init__(
        self,
        model_name: str = config.TOKENIZER_MODEL
    ):

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )


    def encode(
        self,
        text: str
    ) -> list[int]:

        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )


    def decode(
        self,
        tokens: list[int]
    ) -> str:

        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True
        )


    def count(
        self,
        text: str
    ) -> int:

        return len(
            self.encode(text)
        )