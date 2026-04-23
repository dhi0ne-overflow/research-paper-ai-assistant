import os
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

# Load project3/.env first, then cwd (so keys work when run from repo root or project3)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()


@dataclass
class LlmError(Exception):
    message: str
    is_rate_limit: bool = False

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def _provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("groq", "gemini"):
        return explicit
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "none"


GEMINI_DEFAULT_MODEL = "gemini-flash-latest"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LlmError(
            "GROQ_API_KEY is not set. Add it to project3/.env or export it in your shell.",
        )
    return Groq(api_key=key)


def _generate_groq(prompt: str, *, temperature: float) -> str:
    model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
    client = _groq_client()
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = (res.choices[0].message.content or "").strip()
        if not text:
            return "No content returned from the model."
        return text
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "429" in msg or "rate limit" in msg:
            raise LlmError("Groq rate limit reached. Wait and try again.", is_rate_limit=True) from e
        raise LlmError(f"Groq error: {e}") from e


def _generate_gemini(prompt: str, *, temperature: float) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LlmError(
            "GEMINI_API_KEY is not set. Add it to project3/.env or switch LLM_PROVIDER=groq with GROQ_API_KEY.",
        )
    genai.configure(api_key=key)
    model_name = os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    model = genai.GenerativeModel(model_name)
    try:
        res = model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        if not res.text:
            return "No content returned from the model."
        return res.text
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "429" in msg or "Resource exhausted" in msg:
            raise LlmError("Rate limit reached. Wait a minute and try again.", is_rate_limit=True) from e
        raise LlmError(f"Model error: {msg}") from e


def generate(prompt: str, *, temperature: float = 0.2) -> str:
    provider = _provider()
    if provider == "none":
        raise LlmError(
            "No LLM API key found. Set GROQ_API_KEY or GEMINI_API_KEY in project3/.env "
            "(see README.md), or set LLM_PROVIDER with the matching key.",
        )
    if provider == "groq":
        return _generate_groq(prompt, temperature=temperature)
    return _generate_gemini(prompt, temperature=temperature)


def active_llm_label() -> str:
    """Short label for UI / debugging (no secrets)."""
    p = _provider()
    if p == "none":
        return "not configured (set GROQ_API_KEY or GEMINI_API_KEY)"
    if p == "groq":
        return f"groq:{os.getenv('GROQ_MODEL', GROQ_DEFAULT_MODEL)}"
    return f"gemini:{os.getenv('GEMINI_MODEL', GEMINI_DEFAULT_MODEL)}"
