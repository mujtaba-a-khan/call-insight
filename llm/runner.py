# llm/runner.py
# Local LLM runner abstraction (safe, optional).
# - The app should work without any model (runner="none").
# - If runner="ollama", we use the local HTTP API. Failures return "" quickly.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LlmConfig:
    """Runtime configuration for the local LLM backend.

    Attributes:
        runner: Backend identifier: "none" or "ollama".
        name: Model name, e.g., "llama3:instruct".
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens to generate.
        timeout_s: HTTP timeout for requests.
        endpoint: Ollama /api/generate endpoint URL.
    """
    runner: str = "none"
    name: str = "llama3:instruct"
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_s: int = 20
    endpoint: str = "http://localhost:11434/api/generate"


class LocalLlm:
    """Thin wrapper around a local text-generation backend."""

    def __init__(self, cfg: Optional[LlmConfig] = None):
        """Create a new LocalLlm wrapper."""
        self.cfg = cfg or LlmConfig()

    # -------------------- UI Health Check ------------------------------------
    def available(self) -> bool:
        """Return True if the configured backend is reachable and enabled."""
        r = (self.cfg.runner or "none").lower()
        if r == "none":
            return False
        if r == "ollama":
            return self._ollama_available()
        return False

    def _ollama_available(self) -> bool:
        """Best-effort check of the local Ollama server via /api/tags."""
        try:
            import requests
        except Exception:
            return False
        try:
            base = self.cfg.endpoint
            if "/api/" in base:
                base = base.split("/api/")[0]
            resp = requests.get(f"{base}/api/tags", timeout=0.5)
            return bool(resp.ok)
        except Exception:
            return False

    # -------------------- Text Completion ------------------------------------
    def complete(self, prompt: str) -> str:
        """Return a short completion; returns '' on errors/disabled."""
        if not isinstance(prompt, str) or not prompt.strip():
            return ""
        runner = (self.cfg.runner or "none").lower()
        if runner == "none":
            return ""
        if runner == "ollama":
            return self._complete_ollama(prompt.strip())
        return ""

    def _complete_ollama(self, prompt: str) -> str:
        try:
            import requests
        except Exception:
            return ""
        try:
            payload = {
                "model": self.cfg.name,
                "prompt": prompt,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
                "stream": False,
            }
            r = requests.post(self.cfg.endpoint, json=payload, timeout=self.cfg.timeout_s)
            if not r.ok:
                return ""
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception:
            return ""
