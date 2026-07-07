"""Shared LLM client factory — uses Replit AI Integrations proxy when available."""
import os
import openai

MODEL = os.environ.get("LLM_MODEL", "gpt-5.1")


def get_client() -> openai.OpenAI:
    api_key = (
        os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "no-key"
    )
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def llm_available() -> bool:
    return bool(
        os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
