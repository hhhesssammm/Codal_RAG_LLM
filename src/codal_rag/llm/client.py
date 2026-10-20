from __future__ import annotations

import json

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)


class LLMClient:

    def __init__(self):

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        if not LLM_MODEL:
            raise RuntimeError(
                "LLM_MODEL is not configured."
            )

        client_kwargs = {
            "api_key": OPENAI_API_KEY,
        }

        if OPENAI_BASE_URL:

            client_kwargs[
                "base_url"
            ] = OPENAI_BASE_URL

        self.client = OpenAI(
            **client_kwargs
        )

        self.model = LLM_MODEL


    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:

        if not text:
            raise ValueError(
                "LLM response is empty."
            )

        text = text.strip()

        if text.startswith("```json"):
            text = text[
                len("```json"):
            ]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if (
            start == -1
            or end == -1
            or end <= start
        ):

            raise ValueError(
                "LLM response does not contain "
                "a valid JSON object."
            )

        candidate = text[
            start:end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "LLM returned malformed JSON."
            ) from exc


    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:

        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty."
            )

        if not user_prompt.strip():
            raise ValueError(
                "user_prompt cannot be empty."
            )

        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                temperature=(
                    LLM_TEMPERATURE
                ),

                max_tokens=(
                    LLM_MAX_OUTPUT_TOKENS
                ),

                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_prompt
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            user_prompt
                        ),
                    },
                ],
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise RuntimeError(
                "LLM returned an empty response."
            )

        return self._parse_json(
            content
        )