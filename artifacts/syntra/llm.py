"""Shared LLM client factory — uses Replit AI Integrations proxy when available."""
import contextvars
import datetime
import hashlib
import json
import os
import openai

MODEL = os.environ.get("LLM_MODEL", "gpt-5.1")

# High-volume / low-stakes calls (favorability assessment, party inference)
# ride a cheap model by default.
LIGHT_MODEL = os.environ.get("LLM_LIGHT_MODEL", "gpt-5.4-mini")

# Who model calls are attributed to in the audit chain. Set per thread/request
# (contextvars are per-thread, so the pipeline's background thread sets its own).
_llm_actor: contextvars.ContextVar[str] = contextvars.ContextVar("llm_actor", default="system")


def set_llm_actor(actor_id: str) -> None:
    """Attribute subsequent LLM calls in this thread/context to a user."""
    _llm_actor.set(actor_id or "system")


def audited_chat(stage: str, ref: str = "", **kwargs):
    """chat.completions.create with an audit-chain record of the model action.

    Logs actor, stage, prompt hash, and output hash per call (§12: full audit
    trail of model actions). Audit failures never break the model call.
    """
    resp = get_client().chat.completions.create(**kwargs)
    try:
        from audit import AuditLog
        from models import AuditEvent

        prompt_str = json.dumps(kwargs.get("messages", []), sort_keys=True)
        output = ""
        try:
            output = resp.choices[0].message.content or ""
        except Exception:
            pass
        AuditLog().append(AuditEvent(
            ts=datetime.datetime.utcnow().isoformat(),
            actor_id=_llm_actor.get(),
            action=f"llm_call:{stage}",
            input_hash=hashlib.sha256(ref.encode()).hexdigest()[:16] if ref else "",
            prompt_hash=hashlib.sha256(prompt_str.encode()).hexdigest(),
            output_json=json.dumps({
                "model": kwargs.get("model", ""),
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "output_chars": len(output),
            }),
        ))
    except Exception:
        pass  # the audit record is enrichment — never fail the pipeline on it
    return resp


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
