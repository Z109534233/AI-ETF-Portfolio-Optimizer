"""
Centralized OpenAI Responses API service (Issue #20 -- OpenAI architecture).

Every OpenAI call in this app goes through generate_text() / cached_generate()
here -- no page or module calls the OpenAI SDK directly, and no page uses the
legacy Chat Completions API. This keeps three properties true everywhere the
app touches an LLM:

  1. AI only ever explains/summarizes numbers the app already computed
     deterministically -- it is never asked to compute or recompute a
     financial/ML/risk number itself. Callers pass already-computed values
     in as plain text; nothing here parses a number back out of the model's
     response for use in any calculation.
  2. The app keeps working with no OPENAI_API_KEY configured, or when the
     API call fails/times out/returns nothing usable -- callers get an
     explicit `{"available": False, ...}` result and are expected to fall
     back to their own deterministic/rule-based text, never to block.
  3. The key is read only from Streamlit secrets or the environment, never
     hard-coded, logged, printed, or echoed back in any response/error.
"""

import hashlib
import os

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 1


def _secret(name: str):
    """Read one key from st.secrets, tolerating the case where no
    secrets.toml exists at all (st.secrets raises in that case on some
    Streamlit versions) or Streamlit itself isn't importable (plain
    pytest/CI context)."""
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return value if value else None
    except Exception:
        return None


def get_api_key():
    """OPENAI_API_KEY from Streamlit secrets (Community Cloud deployment)
    first, then the environment (local dev / CI). Returns None -- never an
    empty string -- when unavailable."""
    return _secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or None


def get_model() -> str:
    """Optional OPENAI_MODEL override (secrets, then environment); defaults
    to a cost-conscious current text model."""
    return _secret("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def is_configured() -> bool:
    return bool(get_api_key())


def get_client():
    """Return an OpenAI client if a key is configured and the SDK is
    installed, else None. Never raises -- callers treat None as "fall back
    to rule-based/deterministic output"."""
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        import openai
        return openai.OpenAI(
            api_key=api_key, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=MAX_RETRIES,
        )
    except Exception:
        return None


def generate_text(system_instructions: str, user_content: str, *, max_output_tokens: int = 1400) -> dict:
    """Safe text-generation wrapper around `client.responses.create()` (the
    current OpenAI Responses API -- NOT the legacy `chat.completions.create`).

    Always returns a dict, never raises:
      available -> {"available": True, "text": "...", "source": "ai", "model": "..."}
      unavailable -> {"available": False, "text": None, "source": "rule_based", "error": "..."}

    Any SDK/network/timeout/empty-output error is caught and reported
    through "error" so the caller can fall back cleanly and label the
    result "Rule-Based" instead of "AI-Generated".
    """
    client = get_client()
    if client is None:
        return {
            "available": False, "text": None, "source": "rule_based",
            "error": "OpenAI not configured (no API key, or the openai package is unavailable)",
        }

    model = get_model()
    try:
        response = client.responses.create(
            model=model,
            instructions=system_instructions,
            input=user_content,
            max_output_tokens=max_output_tokens,
        )
        text = getattr(response, "output_text", None)
        if not text:
            return {
                "available": False, "text": None, "source": "rule_based",
                "error": "OpenAI response contained no text output",
            }
        return {"available": True, "text": text, "source": "ai", "model": model}
    except Exception as e:
        # Never surfaces the API key or any secret -- only the exception
        # type/message, which is safe to show in a caption/log.
        return {
            "available": False, "text": None, "source": "rule_based",
            "error": f"{type(e).__name__}: {e}",
        }


def fingerprint(*parts) -> str:
    """Deterministic fingerprint of the inputs behind a cacheable AI result.

    Used so a Streamlit rerun triggered by an UNRELATED widget never
    re-spends an API call (same fingerprint -> reuse cached result), but
    ANY change to a relevant input invalidates the cache and allows a new
    call. Never includes the API key or any other secret -- only the
    already-computed, already-displayed numeric/text inputs themselves.
    """
    raw = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_generate(session_state, cache_key: str, fingerprint_value: str,
                     system_instructions: str, user_content: str,
                     *, max_output_tokens: int = 1400) -> dict:
    """Session-state-cached wrapper around generate_text().

    Reuses the prior result when `fingerprint_value` is unchanged (so
    unrelated widget interactions on the same rerun never trigger a new
    API call) and only calls OpenAI again once the fingerprint changes
    (i.e. a materially different input) or no cached result exists yet.
    `session_state` is any dict-like object (st.session_state in the app,
    a plain dict in tests).
    """
    cached = session_state.get(cache_key)
    if cached and cached.get("fingerprint") == fingerprint_value:
        return cached["result"]

    result = generate_text(system_instructions, user_content, max_output_tokens=max_output_tokens)
    session_state[cache_key] = {"fingerprint": fingerprint_value, "result": result}
    return result
