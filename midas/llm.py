from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str


class _OpenAIMessageAPI:
    def __init__(self, client):
        self._client = client

    def create(self, *, model: str, max_tokens: int, messages: list[dict]):
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


class OpenAICompatClient:
    def __init__(self, client):
        self.messages = _OpenAIMessageAPI(client)


def resolve_llm_config(
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMConfig:
    chosen_provider = (provider or "").strip().lower()

    if not chosen_provider:
        if api_key:
            chosen_provider = "anthropic" if api_key.startswith("sk-ant-") else "openai"
        elif os.environ.get("OPENAI_API_KEY"):
            chosen_provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            chosen_provider = "anthropic"
        else:
            raise ValueError(
                "No LLM credentials found. Pass api_key= or set OPENAI_API_KEY / ANTHROPIC_API_KEY."
            )

    if chosen_provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API key required. Pass api_key= or set OPENAI_API_KEY.")
        return LLMConfig(
            provider="openai",
            api_key=key,
            model=model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        )

    if chosen_provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Anthropic API key required. Pass api_key= or set ANTHROPIC_API_KEY.")
        return LLMConfig(
            provider="anthropic",
            api_key=key,
            model=model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        )

    raise ValueError(f"Unsupported LLM provider: {chosen_provider}")


def create_llm_client(
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    config = resolve_llm_config(api_key=api_key, provider=provider, model=model)

    if config.provider == "openai":
        from openai import OpenAI

        return OpenAICompatClient(OpenAI(api_key=config.api_key)), config

    from anthropic import Anthropic

    return Anthropic(api_key=config.api_key), config
