#!/usr/bin/env bash
# Run the whole offline suite. No API key, no network, no LaTeX required
# (the PDF check skips itself if no engine is installed).
set -uo pipefail
cd "$(dirname "$0")"

# Prefer the project venv if there is one. Bare `python3` is the system
# interpreter, which has none of the requirements installed.
PY=python3
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
fi
echo "using $PY ($($PY --version 2>&1))"

fail=0
for t in tests/test_parser_templates.py tests/test_providers.py tests/test_pipeline.py tests/test_app_smoke.py; do
  echo ""
  echo "════ $t ════"
  "$PY" "$t" || fail=1
done
echo ""
if [ "$fail" -eq 0 ]; then echo "✅ all suites passed"; else echo "❌ something failed"; fi
exit $fail
