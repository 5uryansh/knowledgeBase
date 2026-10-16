"""Optional LLM access (summaries, benchmark judging). Nothing in the core pipeline needs it.

Configure one of:

* any OpenAI-compatible endpoint — OpenAI, vLLM, Ollama, llama.cpp, LM Studio, a Hugging Face Space:
  ``KB_LLM_BASE_URL=http://localhost:11434/v1  KB_LLM_MODEL=qwen2.5:7b  [KB_LLM_API_KEY=...]``
* Google Gemini: ``GEMINI_API_KEY=...`` (rotates through flash-lite models to survive free-tier quotas)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Protocol

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3-flash", "gemini-2.5-flash"]


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, json_mode: bool = False) -> str:  # pragma: no cover
        ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 180):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.name = f"openai-compatible:{model}"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        payload = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user-configured URL
                    return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
            except Exception:  # noqa: BLE001 - transient network / rate limit
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""


class GeminiLLM:
    """Rotates through a model pool so one model's daily quota doesn't stall a long run."""

    def __init__(self, api_key: str, models: list[str] | None = None, retries: int = 2):
        from google import genai  # pip install google-genai

        self.client = genai.Client(api_key=api_key)
        self.models = models or GEMINI_MODELS
        self.idx, self.retries = 0, retries
        self.name = f"gemini:{self.models[0]}"

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        last = None
        config = {"temperature": 0, **({"response_mime_type": "application/json"} if json_mode else {})}
        for _ in range(len(self.models)):
            model = self.models[self.idx]
            for attempt in range(self.retries):
                try:
                    return self.client.models.generate_content(model=model, contents=prompt, config=config).text or ""
                except Exception as exc:  # noqa: BLE001 - quota / transient
                    last = exc
                    time.sleep(min(2 ** attempt, 8))
            self.idx = (self.idx + 1) % len(self.models)
        raise RuntimeError(f"all Gemini models exhausted: {last}")


def build_llm(prefix: str = "KB_LLM") -> LLM | None:
    """From the environment; ``prefix`` lets the benchmark judge use a different model (KB_JUDGE_*)."""
    base = os.environ.get(f"{prefix}_BASE_URL")
    if base:
        return OpenAICompatibleLLM(base, os.environ.get(f"{prefix}_MODEL", "default"), os.environ.get(f"{prefix}_API_KEY", ""))
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiLLM(os.environ["GEMINI_API_KEY"])
    return None


def parse_json(raw: str) -> dict:
    """Lenient JSON extraction (models like to wrap JSON in prose or code fences)."""
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[start:end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        return {}
