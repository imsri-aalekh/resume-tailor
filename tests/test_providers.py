"""
Provider wiring: request shape and response parsing for both API families.

We cannot call the real endpoints from the test suite, so this asserts the two
things that actually break when adding a provider: that we send the shape the
provider expects, and that we can read back what it returns.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.llm import PROVIDERS, LLMClient, extract_json

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

        if prov.fmt == "anthropic":
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

        check(f"{key}: model defaults to something",
              bool(payload.get("model")), payload.get("model"))
        check(f"{key}: env var name resolves", bool(c._env_var()))

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
