from __future__ import annotations

import json

from codal_rag.config import LLMConfig, load_config


class LLMError(RuntimeError):
    """Provider failure with a safe message (never includes headers or response bodies)."""


class LLMClient:
    def __init__(self, config: LLMConfig | None = None, *, client=None):
        self.config = config or load_config().llm
        if client is None:
            if not self.config.api_key:
                raise LLMError("Set CODAL_LLM_API_KEY or OPENAI_API_KEY for extraction")
            if not self.config.model:
                raise LLMError("Set CODAL_LLM_MODEL for your configured provider")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("Install codal-rag[llm] for live extraction") from exc
            client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                            timeout=self.config.timeout, max_retries=self.config.max_retries)
        self.client = client
        self.model = self.config.model

    @staticmethod
    def _parse_json(text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM response is empty")
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if not lines or lines[-1].strip() != "```":
                raise ValueError("LLM returned an incomplete JSON code fence")
            text = "\n".join(lines[1:-1])
        def reject_constant(_):
            raise ValueError("LLM JSON contains a non-finite number")
        try:
            value = json.loads(text, parse_constant=reject_constant)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned malformed JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("LLM response must be a JSON object")
        return value

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("Prompts cannot be empty")
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}])
        except Exception:
            # Raw provider messages can include sensitive URLs and request contents.
            raise LLMError("LLM request failed; check provider settings, connectivity, quota, and timeout") from None
        if not response.choices:
            raise LLMError("LLM returned no choices")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError("LLM output hit its token limit; increase max_output_tokens or reduce evidence")
        if getattr(choice.message, "refusal", None):
            raise LLMError("LLM provider declined the extraction request")
        return self._parse_json(choice.message.content)
