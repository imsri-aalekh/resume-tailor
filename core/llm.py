"""
Thin Anthropic Messages API client.

Deliberately uses `requests` rather than the official SDK: fewer dependencies
means fewer ways for a Streamlit Cloud build to fail, and the surface we need
here is one POST.

`StubLLM` implements the same interface with canned responses so the whole
pipeline can be exercised without network or an API key (used by the tests).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-sonnet-4-5"
SMART_MODEL = "claude-opus-4-5"


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: Dict[str, Any]) -> None:
        self.input_tokens += int(other.get("input_tokens", 0) or 0)
        self.output_tokens += int(other.get("output_tokens", 0) or 0)
        self.calls += 1

    def estimated_cost_usd(self, in_rate: float = 3.0, out_rate: float = 15.0) -> float:
        """Rough cost at Sonnet list prices (USD per million tokens)."""
        return (self.input_tokens / 1e6) * in_rate + (self.output_tokens / 1e6) * out_rate


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 timeout: int = 180, max_retries: int = 3):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self.transcript: List[Dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, *, max_tokens: int = 4000,
                 temperature: float = 0.2, label: str = "") -> str:
        if not self.api_key:
            raise LLMError(
                "No Anthropic API key configured. Add ANTHROPIC_API_KEY to your "
                "Streamlit secrets (or paste a key in the sidebar)."
            )
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(API_URL, headers=headers, json=payload,
                                  timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = f"network error: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                text = "".join(
                    blk.get("text", "") for blk in data.get("content", [])
                    if blk.get("type") == "text"
                )
                self.usage.add(data.get("usage", {}))
                self.transcript.append({"label": label, "in": user[:4000], "out": text[:8000]})
                return text
            if r.status_code in (429, 500, 502, 503, 529):
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(2.0 * (attempt + 1))
                continue
            raise LLMError(f"Anthropic API error {r.status_code}: {r.text[:500]}")
        raise LLMError(f"Anthropic API unavailable after {self.max_retries} attempts. {last_err}")

    def complete_json(self, system: str, user: str, *, max_tokens: int = 4000,
                      temperature: float = 0.1, label: str = "") -> Any:
        """Ask for JSON and parse it, retrying once with a repair prompt."""
        primed = user + "\n\nRespond with JSON only. No prose, no markdown fence."
        text = self.complete(system, primed, max_tokens=max_tokens,
                             temperature=temperature, label=label)
        parsed = extract_json(text)
        if parsed is not None:
            return parsed
        repair = (
            "The following was supposed to be valid JSON but could not be parsed. "
            "Return the same content as strictly valid JSON, nothing else.\n\n" + text[:6000]
        )
        text2 = self.complete("You repair malformed JSON.", repair,
                              max_tokens=max_tokens, temperature=0.0, label=label + ":repair")
        parsed = extract_json(text2)
        if parsed is None:
            raise LLMError("Model did not return parseable JSON.")
        return parsed


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return None
    candidates: List[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text)
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        try:
            s = text.index(start_ch)
            e = text.rindex(end_ch)
            if e > s:
                candidates.append(text[s : e + 1])
        except ValueError:
            pass
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            # tolerate trailing commas
            fixed = re.sub(r",(\s*[}\]])", r"\1", cand)
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


# --------------------------------------------------------------------------
# offline stub
# --------------------------------------------------------------------------


class StubLLM(LLMClient):
    """Deterministic offline stand-in used by the test suite.

    Handlers are registered by label prefix and receive the user prompt.
    """

    def __init__(self, handlers: Optional[Dict[str, Any]] = None):
        super().__init__(api_key="stub", model="stub")
        self.handlers: Dict[str, Any] = handlers or {}
        self.seen: List[str] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, max_tokens: int = 4000,
                 temperature: float = 0.2, label: str = "") -> str:
        self.seen.append(label)
        self.usage.add({"input_tokens": len(user) // 4, "output_tokens": 400})
        for prefix, handler in self.handlers.items():
            if label.startswith(prefix):
                out = handler(user) if callable(handler) else handler
                return out if isinstance(out, str) else json.dumps(out)
        raise LLMError(f"StubLLM has no handler for label {label!r}")
