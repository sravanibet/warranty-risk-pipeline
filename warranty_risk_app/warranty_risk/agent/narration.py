# Generates the agent's plain-language narration using Gemini 2.5 — the model
# IntelliMake (the parent project) requires. This is not a permanent model
# lock-in for the rest of the codebase, just what this narration layer uses
# for now, per IntelliMake's current requirement.
#
# Falls back to a plain templated sentence (passed in by the caller) whenever
# no API key is configured, the SDK isn't installed, or the call fails for any
# reason — so the agent's phases never break or block on network/LLM issues,
# including live demos.
import os

DEFAULT_MODEL = "gemini-2.5-flash"

try:
    from google import genai
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

_SYSTEM_INSTRUCTION = (
    "You are the narration voice of a manufacturing Quality Monitoring Agent. "
    "Given structured facts about what just happened in one phase of its "
    "pipeline (Setup, Data, Learning, or Operate), write a short (1-3 "
    "sentence) plain-language summary a plant manager could read at a "
    "glance. State only what the facts support — never invent numbers or "
    "outcomes not given to you. No greetings, no markdown, no bullet points."
)


class Narrator:
    # Wraps Gemini 2.5. If unavailable/unconfigured, `enabled` is False and
    # `narrate()` always returns the caller-supplied fallback untouched.

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        self.enabled = bool(_SDK_AVAILABLE and api_key)
        self._client = None
        if self.enabled:
            try:
                self._client = genai.Client(api_key=api_key)
            except Exception:
                self.enabled = False

    def narrate(self, phase: str, facts: dict, fallback: str) -> str:
        if not self.enabled:
            return fallback

        prompt = f"Phase: {phase}\nFacts: {facts}\n\nWrite the narration."
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"system_instruction": _SYSTEM_INSTRUCTION},
            )
            text = (response.text or "").strip()
            return text if text else fallback
        except Exception:
            return fallback
