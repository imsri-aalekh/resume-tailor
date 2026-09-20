"""
Thin multi-provider chat client.

Deliberately uses `requests` rather than a vendor SDK: fewer dependencies means
fewer ways for a Streamlit Cloud build to fail, and the surface we need here is
one POST. Supporting several providers costs about thirty lines because
everyone except Anthropic speaks the OpenAI chat-completions shape.

Provider notes that matter for this app:

  * The writer/critic loop leans hard on instruction-following and clean JSON.
    Strong models produce few validator violations; weaker free models produce
    more. That fails *safe* — the guards revert bad rewrites to your original
    text — but you will see more blocked rewrites and a flatter result.
  * Model names change constantly. The defaults below are a starting point, not
    a promise; the UI lets you type any model string the provider accepts.

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

API_VERSION = "2023-06-01"


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    url: str
    fmt: str                     # "anthropic" | "openai"
    auth: str                    # "x-api-key" | "bearer"
    default_model: str
    console: str                 # where to get a key
    free: bool = False
    json_mode: bool = False      # supports response_format=json_object
    in_rate: float = 0.0         # USD per million input tokens
    out_rate: float = 0.0
    note: str = ""


PROVIDERS: Dict[str, Provider] = {
    "anthropic": Provider(
        key="anthropic", label="Anthropic (Claude)",
        url="https://api.anthropic.com/v1/messages",
        fmt="anthropic", auth="x-api-key",
        default_model="claude-sonnet-4-5",
        console="https://console.anthropic.com",
        in_rate=3.0, out_rate=15.0,
        note="Best results. Paid — a Claude subscription does not include API credit.",
    ),
    "groq": Provider(
        key="groq", label="Groq (free tier)",
        url="https://api.groq.com/openai/v1/chat/completions",
        fmt="openai", auth="bearer",
        default_model="llama-3.3-70b-versatile",
        console="https://console.groq.com/keys",
        free=True, json_mode=True,
        note="Free tier, no card. Rate limited per minute and per day. "
             "Expect more blocked rewrites than Claude — the guards catch them.",
    ),
    "gemini": Provider(
        key="gemini", label="Google Gemini (free tier)",
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        fmt="openai", auth="bearer",
        default_model="gemini-2.0-flash",
        console="https://aistudio.google.com/apikey",
        free=True, json_mode=True,
        note="Free tier via AI Studio. Uses Google's OpenAI-compatible endpoint.",
    ),
    "openrouter": Provider(
        key="openrouter", label="OpenRouter (mixed free/paid)",
        url="https://openrouter.ai/api/v1/chat/completions",
        fmt="openai", auth="bearer",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        console="https://openrouter.ai/keys",
        free=True,
        note="Models ending in ':free' cost nothing but queue behind paid traffic.",
    ),
}

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = PROVIDERS["anthropic"].default_model
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
        """Rough cost in USD per million tokens. Zero rates mean a free tier."""
        return (self.input_tokens / 1e6) * in_rate + (self.output_tokens / 1e6) * out_rate


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "",
                 provider: str = DEFAULT_PROVIDER,
                 timeout: int = 180, max_retries: int = 3):
        self.provider: Provider = PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])
        self.api_key = api_key or os.environ.get(self._env_var(), "")
        self.model = model or self.provider.default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        self.transcript: List[Dict[str, Any]] = []

    def _env_var(self) -> str:
        return {
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(self.provider.key, "ANTHROPIC_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # -- request shaping -------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        h = {"content-type": "application/json"}
        if self.provider.auth == "x-api-key":
            h["x-api-key"] = self.api_key
            h["anthropic-version"] = API_VERSION
        else:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.provider.key == "openrouter":
            # OpenRouter asks for these so free-tier traffic is attributable.
            h["HTTP-Referer"] = "https://share.streamlit.io"
            h["X-Title"] = "Resume Tailor"
        return h

    def _payload(self, system: str, user: str, max_tokens: int,
                 temperature: float, json_mode: bool = False) -> Dict[str, Any]:
        if self.provider.fmt == "anthropic":
            return {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode and self.provider.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _extract_text(fmt: str, data: Dict[str, Any]) -> str:
        if fmt == "anthropic":
            return "".join(
                blk.get("text", "") for blk in data.get("content", [])
                if blk.get("type") == "text"
            )
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):      # some gateways return content parts
            return "".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        return str(content or "")

    @staticmethod
    def _extract_usage(fmt: str, data: Dict[str, Any]) -> Dict[str, int]:
        u = data.get("usage") or {}
        if fmt == "anthropic":
            return {"input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0)}
        return {"input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0)}

    # -- the call --------------------------------------------------------
    def complete(self, system: str, user: str, *, max_tokens: int = 4000,
                 temperature: float = 0.2, label: str = "",
                 json_mode: bool = False) -> str:
        if not self.api_key:
            raise LLMError(
                f"No API key for {self.provider.label}. Add "
                f"{self._env_var()} to your Streamlit secrets, or paste a key "
                f"in the sidebar. Get one at {self.provider.console}"
            )
        payload = self._payload(system, user, max_tokens, temperature, json_mode)
        headers = self._headers()
        last_err: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.provider.url, headers=headers,
                                  json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_err = f"network error: {exc}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if r.status_code == 200:
                data = r.json()
                text = self._extract_text(self.provider.fmt, data)
                self.usage.add(self._extract_usage(self.provider.fmt, data))
                self.transcript.append({"label": label, "in": user[:4000],
                                        "out": text[:8000]})
                if not text.strip():
                    last_err = "provider returned an empty completion"
                    continue
                return text

            body = r.text[:400]
            # A provider that rejects response_format still works without it.
            if r.status_code == 400 and "response_format" in body and \
                    "response_format" in payload:
                payload.pop("response_format", None)
                continue
            if r.status_code in (429, 500, 502, 503, 529):
                wait = 2.0 * (attempt + 1)
                retry_after = r.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = min(30.0, float(retry_after))
                    except ValueError:
                        pass
                last_err = f"HTTP {r.status_code}: {body}"
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                raise LLMError(
                    f"{self.provider.label} rejected the key ({r.status_code}). "
                    f"Check it at {self.provider.console}. Details: {body}"
                )
            raise LLMError(f"{self.provider.label} error {r.status_code}: {body}")

        hint = ""
        if last_err and "429" in last_err and self.provider.free:
            hint = (" Free tiers are rate limited per minute — wait a minute "
                    "and run it again, or lower the critique rounds to 1.")
        raise LLMError(
            f"{self.provider.label} unavailable after {self.max_retries} "
            f"attempts. {last_err}{hint}"
        )

    def complete_json(self, system: str, user: str, *, max_tokens: int = 4000,
                      temperature: float = 0.1, label: str = "") -> Any:
        """Ask for JSON and parse it, retrying once with a repair prompt."""
        primed = user + "\n\nRespond with JSON only. No prose, no markdown fence."
        text = self.complete(system, primed, max_tokens=max_tokens,
                             temperature=temperature, label=label, json_mode=True)
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
        super().__init__(api_key="stub", model="stub", provider=DEFAULT_PROVIDER)
        self.handlers: Dict[str, Any] = handlers or {}
        self.seen: List[str] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, max_tokens: int = 4000,
                 temperature: float = 0.2, label: str = "",
                 json_mode: bool = False) -> str:
        self.seen.append(label)
        self.usage.add({"input_tokens": len(user) // 4, "output_tokens": 400})
        for prefix, handler in self.handlers.items():
            if label.startswith(prefix):
                out = handler(user) if callable(handler) else handler
                return out if isinstance(out, str) else json.dumps(out)
        raise LLMError(f"StubLLM has no handler for label {label!r}")
