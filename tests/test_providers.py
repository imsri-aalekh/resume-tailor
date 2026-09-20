"""
Provider wiring: request shape and response parsing for both API families.

We cannot call the real endpoints from the test suite, so this asserts the two
things that actually break when adding a provider: that we send the shape the
provider expects, and that we can read back what it returns.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.llm import (PROVIDERS, LLMClient, extract_json, _pace,
                      reset_pacing, ENV_VARS, DEFAULT_PROVIDER)

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    print("== request shaping ==")
    for key, prov in PROVIDERS.items():
        c = LLMClient(api_key="test-key", provider=key)
        payload = c._payload("SYSTEM TEXT", "USER TEXT", 500, 0.3, json_mode=True)
        headers = c._headers()

        if prov.local:
            check(f"{key}: no auth header for a local provider",
                  "Authorization" not in headers and "x-api-key" not in headers)
            check(f"{key}: needs no API key to be usable",
                  LLMClient(provider=key).available)
            check(f"{key}: url points at localhost",
                  "localhost" in prov.url or "127.0.0.1" in prov.url, prov.url)
            check(f"{key}: not rate limited",
                  prov.rpm == 0)
            roles = [m["role"] for m in payload["messages"]]
            check(f"{key}: system sent as a message", roles == ["system", "user"])
        elif prov.fmt == "anthropic":
            check(f"{key}: system sent as top-level field",
                  payload.get("system") == "SYSTEM TEXT")
            check(f"{key}: single user message",
                  payload["messages"] == [{"role": "user", "content": "USER TEXT"}])
            check(f"{key}: x-api-key header", headers.get("x-api-key") == "test-key")
            check(f"{key}: version header", "anthropic-version" in headers)
        else:
            roles = [m["role"] for m in payload["messages"]]
            check(f"{key}: system sent as a message", roles == ["system", "user"])
            check(f"{key}: bearer auth",
                  headers.get("Authorization") == "Bearer test-key")
            check(f"{key}: no anthropic-only fields",
                  "system" not in payload)
            check(f"{key}: json mode only when supported",
                  ("response_format" in payload) == prov.json_mode)

        check(f"{key}: thinking suppressed only where declared",
              (payload.get("reasoning_effort") == "none") == prov.no_think)
        check(f"{key}: model defaults to something",
              bool(payload.get("model")), payload.get("model"))
        check(f"{key}: env var name resolves",
              bool(c._env_var()) or prov.local)
        check(f"{key}: listed in the shared env-var table", key in ENV_VARS)
        check(f"{key}: free tiers declare a per-minute budget",
              (prov.rpm > 0) or prov.local or not prov.free,
              f"rpm={prov.rpm}")

    print("\n== response parsing ==")
    anthropic_body = {
        "content": [{"type": "text", "text": '{"edits": []}'},
                    {"type": "thinking", "text": "ignored"}],
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }
    openai_body = {
        "choices": [{"message": {"role": "assistant", "content": '{"edits": []}'}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45},
    }
    openai_parts_body = {
        "choices": [{"message": {"content": [{"type": "text", "text": '{"edits": []}'}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    check("anthropic text extracted, non-text blocks dropped",
          LLMClient._extract_text("anthropic", anthropic_body) == '{"edits": []}')
    check("openai text extracted",
          LLMClient._extract_text("openai", openai_body) == '{"edits": []}')
    check("openai content-parts variant extracted",
          LLMClient._extract_text("openai", openai_parts_body) == '{"edits": []}')
    check("openai empty choices does not crash",
          LLMClient._extract_text("openai", {"choices": []}) == "")

    check("anthropic usage mapped",
          LLMClient._extract_usage("anthropic", anthropic_body) ==
          {"input_tokens": 120, "output_tokens": 45})
    check("openai usage mapped from prompt/completion tokens",
          LLMClient._extract_usage("openai", openai_body) ==
          {"input_tokens": 120, "output_tokens": 45})
    check("missing usage defaults to zero",
          LLMClient._extract_usage("openai", {}) ==
          {"input_tokens": 0, "output_tokens": 0})

    print("\n== json tolerance (weak models fence and trail commas) ==")
    check("fenced json parsed",
          extract_json('```json\n{"a":1}\n```') == {"a": 1})
    check("prose-wrapped json parsed",
          extract_json('Sure! Here it is: {"a":1} Hope that helps.') == {"a": 1})
    check("trailing comma tolerated",
          extract_json('{"a":1,}') == {"a": 1})

    print("\n== request pacing (module scope, survives a Streamlit rerun) ==")

    class Clock:
        """A virtual clock. sleep() advances it, so pacing can be tested
        without waiting a real minute."""
        def __init__(self):
            self.t = 1000.0
            self.slept = []

        def now(self):
            return self.t

        def sleep(self, d):
            self.slept.append(d)
            self.t += d

    def pace(key, c):
        return _pace(PROVIDERS[key], sleep=c.sleep, clock=c.now)

    reset_pacing()
    c = Clock()
    check("free budget is 5/min for gemini", PROVIDERS["gemini"].rpm == 5)

    for _ in range(PROVIDERS["gemini"].rpm):
        pace("gemini", c)
    check("calls inside the budget never sleep", c.slept == [], str(c.slept))

    waited = pace("gemini", c)
    check("the call past the budget waits", len(c.slept) == 1, str(c.slept))
    check("it waits out the window, not a token 2s backoff",
          waited > 55, f"waited {waited}")

    # The bug this replaces: a fresh client used to reset the count, because the
    # timestamps lived on the client. Streamlit rebuilds the client on every
    # rerun, so the spent budget vanished and the next run blew straight
    # through it. The timestamps now live at module scope.
    reset_pacing()
    c = Clock()
    pace("gemini", c)                              # e.g. the JD-extraction call
    LLMClient(api_key="k", provider="gemini")      # Streamlit rerun
    LLMClient(api_key="k", provider="gemini")      # ...and another
    for _ in range(PROVIDERS["gemini"].rpm - 1):
        pace("gemini", c)
    check("a rebuilt client does not reset the budget", c.slept == [], str(c.slept))
    pace("gemini", c)
    check("the 6th call across rebuilt clients is paced", len(c.slept) == 1)

    # and once the window really has passed, no waiting
    reset_pacing()
    c = Clock()
    for _ in range(PROVIDERS["gemini"].rpm):
        pace("gemini", c)
    c.t += 61
    pace("gemini", c)
    check("a call after the window costs nothing", c.slept == [], str(c.slept))

    reset_pacing()
    c = Clock()
    for _ in range(12):
        pace("anthropic", c)
    check("unmetered providers are never paced", c.slept == [])

    reset_pacing()
    c = Clock()
    for _ in range(12):
        pace("ollama", c)
    check("local provider is never paced", c.slept == [])

    reset_pacing()
    c = Clock()
    for _ in range(PROVIDERS["gemini"].rpm):
        pace("gemini", c)
    pace("groq", c)
    check("a spent gemini budget does not pace groq", c.slept == [], str(c.slept))
    reset_pacing()

    # a fake sleep that does NOT advance the clock must fail loudly, not hang
    class BadClock(Clock):
        def sleep(self, d):
            self.slept.append(d)      # time never moves
    reset_pacing()
    bad = BadClock()
    for _ in range(PROVIDERS["gemini"].rpm):
        pace("gemini", bad)
    try:
        pace("gemini", bad)
        check("a non-advancing clock raises instead of hanging", False)
    except Exception as exc:                              # noqa: BLE001
        check("a non-advancing clock raises instead of hanging",
              "made no progress" in str(exc), type(exc).__name__)
    reset_pacing()

    print("\n== free-tier model choices ==")
    # Groq decommissioned llama-3.3-70b-versatile on 16 Aug 2026 and moved the
    # rest of the Llama family to Enterprise, which broke the default silently:
    # the key was valid, the model was not. Every free provider must therefore
    # name a model its free tier can actually call, and offer alternates for
    # when that one churns too.
    for _k, _pr in PROVIDERS.items():
        if not _pr.free or _pr.local:
            continue
        check(f"{_k}: names a default model", bool(_pr.default_model))
        # Scoped to Groq on purpose: these IDs are retired *on Groq*.
        # OpenRouter serves its own Llama 3.3 and is unaffected, so a blanket
        # substring ban here would be a false alarm.
        if _k == "groq":
            check(f"{_k}: default is not a model Groq retired",
                  "llama-3.3-70b" not in _pr.default_model
                  and "llama-3.1-70b" not in _pr.default_model,
                  _pr.default_model)
            check(f"{_k}: offers alternates, since this one churned before",
                  len(_pr.alt_models) >= 1, str(_pr.alt_models))
        check(f"{_k}: any alternates are distinct from the default",
              _pr.default_model not in _pr.alt_models)

    print("\n== reasoning models ==")
    # granite4.2 and friends put their deliberation in `reasoning` and can
    # return content="" when max_tokens runs out mid-thought. Retrying that is
    # pointless, so the client must say what actually happened.
    check("ollama asks for no thinking", PROVIDERS["ollama"].no_think)
    _c = LLMClient(provider="ollama", model="granite4.2:8b")
    _p = _c._payload("s", "u", 20, 0.2, json_mode=False)
    check("reasoning_effort=none is sent", _p.get("reasoning_effort") == "none")
    check("hosted providers are untouched",
          "reasoning_effort" not in LLMClient(api_key="k", provider="anthropic")
          ._payload("s", "u", 20, 0.2))

    _thinking_body = {
        "choices": [{"message": {"role": "assistant", "content": "",
                                 "reasoning": "Let me think about this..."},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 25, "completion_tokens": 20},
    }
    check("an all-reasoning response extracts as empty",
          LLMClient._extract_text("openai", _thinking_body) == "")

    print("\n== local provider liveness ==")
    import dataclasses
    import core.llm as _L
    # A local provider whose server is not running must report unavailable,
    # quickly. Callers use `available` to decide whether to take an offline
    # path; a False positive means they block for the full timeout instead.
    _dead = dataclasses.replace(_L.PROVIDERS["ollama"], key="dead",
                                url="http://localhost:59999/v1/chat/completions")
    _L.PROVIDERS["dead"] = _dead
    _L.ENV_VARS["dead"] = ""
    try:
        _L._REACHABLE.pop("dead", None)
        _c = _L.LLMClient(provider="dead")
        _t = time.monotonic()
        _avail = _c.available
        _elapsed = time.monotonic() - _t
        check("a local provider with no server is not 'available'", _avail is False)
        check("and says so promptly, rather than blocking", _elapsed < 5.0,
              f"{_elapsed:.2f}s")
        try:
            _c.complete("s", "u", max_tokens=5, label="x")
            check("a dead local server raises instead of retrying", False)
        except _L.LLMError as exc:
            check("a dead local server raises instead of retrying",
                  "Could not reach" in str(exc))
            check("the error says how to start it", "ollama serve" in str(exc))
    finally:
        _L.PROVIDERS.pop("dead", None)
        _L.ENV_VARS.pop("dead", None)
        _L._REACHABLE.pop("dead", None)

    print("\n== local provider errors ==")
    c = LLMClient(provider="ollama", model="not-a-real-model")
    check("local timeout is raised for slow local generation", c.timeout >= 600,
          str(c.timeout))
    check("default provider is the local one", DEFAULT_PROVIDER == "ollama")

    print("\n== key resolution ==")
    os.environ["GROQ_API_KEY"] = "env-groq-key"
    try:
        c = LLMClient(provider="groq")
        check("provider-specific env var picked up", c.api_key == "env-groq-key")
        check("groq is marked free", PROVIDERS["groq"].free)
        c2 = LLMClient(provider="anthropic")
        check("wrong provider does not borrow the key", c2.api_key != "env-groq-key")
    finally:
        os.environ.pop("GROQ_API_KEY", None)

    print("\n" + "=" * 58)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL PROVIDER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
