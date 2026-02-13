"""Base helpers shared across agents."""

from __future__ import annotations

from openai import OpenAI

from daily_research.config import Settings


def get_client(settings: Settings) -> OpenAI:
    """Build an OpenAI client from the current settings."""
    kwargs: dict = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> str:
    """Simple single-turn chat helper."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
