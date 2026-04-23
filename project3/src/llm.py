import os
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Align with project1/2: flash model (override with GEMINI_MODEL)
DEFAULT_MODEL = "gemini-flash-latest"
MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


@dataclass
class LlmError(Exception):
    message: str
    is_rate_limit: bool = False

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def get_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(MODEL_NAME)


def generate(prompt: str, *, temperature: float = 0.2) -> str:
    model = get_model()
    try:
        res = model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        if not res.text:
            return "No content returned from the model."
        return res.text
    except Exception as e:  # noqa: BLE001 — surface API errors to user
        msg = str(e)
        if "429" in msg or "Resource exhausted" in msg:
            raise LlmError("Rate limit reached. Wait a minute and try again.", is_rate_limit=True) from e
        raise LlmError(f"Model error: {msg}") from e
